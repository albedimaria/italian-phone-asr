# %% [markdown]
# # Parakeet-TDT-0.6B-v3 fine-tune on 8 kHz G.711 Italian phone speech (setup B12)
#
# Second base, same protocol as the Whisper-turbo rounds: same training data (Common Voice IT subset through
# the G.711 chain + round-trip-filtered synthetic phone clips, same additive noise on the same clips), same
# test sets, checkpoint chosen on dev only, paired bootstrap against its own base.
# Setup chosen from three trial kernels (kaggle/trial_parakeet): the full fine-tune does not fit a T4 even at
# batch 2; B12 = preprocessor, subsampling and the lower 12 of 24 encoder layers frozen, batch 4 x accum 4
# (effective 16), fp16, AdamW lr 1e-4 with 100 warmup steps and cosine decay; 11.4 of 14.56 GiB on 10-11 s
# clips, 1.8 s per optimizer step. One GPU.
# Dev for checkpoint selection: the Common Voice part only, through G.711, scored with the project's
# normalizer. The synthetic dev is reported but not used to select: the synthetic set was filtered with
# Parakeet itself, so it is already easy for it (1.67% WER before training).
#
# Inputs: Kaggle secret MDC_API_KEY (Common Voice cannot be re-hosted), dataset ipasr-train-r3.
# Output in /kaggle/working: parakeet-tdt-it-phone.nemo (best dev checkpoint), dev_metrics.json.
# IPASR_SMOKE=1 runs the same train() on a few local clips on CPU (a pre-flight check, not a result).

# %%
import json
import os
import random
import subprocess
import sys
import time
import zlib
from pathlib import Path

SMOKE = bool(os.environ.get("IPASR_SMOKE"))
if not SMOKE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nemo_toolkit[asr]", "jiwer>=3.1",
                    "num2words>=0.5.14", "git+https://github.com/albedimaria/italian-phone-asr.git"], check=True)

import numpy as np
import soundfile as sf

CFG = {
    "round": "parakeet-1", "base": "nvidia/parakeet-tdt-0.6b-v3", "out_name": "parakeet-tdt-it-phone",
    "freeze_lower_layers": 12, "lr": 1e-4, "weight_decay": 1e-3, "warmup_steps": 100, "min_lr": 1e-6,
    "epochs": 3, "batch_size": 4, "grad_accum": 4, "eval_every": 200, "early_stop_patience": 5, "seed": 0,
    "noise_p": 0.5, "noise_snr_db": [15, 35], "max_duration": 20.0, "max_hours": 10.0, "num_workers": 4,
}
print(CFG, flush=True)
random.seed(CFG["seed"]); np.random.seed(CFG["seed"])


# %% [markdown]
# ## Audio: the same chain and the same noise as the Whisper rounds (kaggle/train_lora.py)

# %%
def g711_16k(path: str) -> np.ndarray:
    """Any file -> 8 kHz mu-law round trip -> 16 kHz float32. Same ffmpeg chain as src/iphasr/data.py."""
    enc = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "8000", "-c:a", "pcm_mulaw",
                          "-f", "mulaw", "-"], capture_output=True, check=True).stdout
    raw = subprocess.run(["ffmpeg", "-v", "error", "-f", "mulaw", "-ar", "8000", "-ac", "1", "-i", "-",
                          "-ar", "16000", "-f", "f32le", "-"], input=enc, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def ulaw_16k(path: str) -> np.ndarray:
    """Stored synth clips are already mu-law 8 kHz: decode + resample only."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ar", "16000", "-ac", "1", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def prepare(rows: list[dict], split: str, out_dir: Path) -> Path:
    """Write 16 kHz wavs + a NeMo manifest. Train clips get the Whisper rounds' noise (same seed per clip)."""
    from concurrent.futures import ThreadPoolExecutor
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(i_r):
        i, r = i_r
        audio = g711_16k(r["audio"]) if r["loader"] == "g711" else ulaw_16k(r["audio"])
        rng = np.random.default_rng([zlib.crc32(r["audio"].encode()), CFG["seed"]])
        if split == "train" and rng.random() < CFG["noise_p"] and audio.size:
            snr = rng.uniform(*CFG["noise_snr_db"])
            p = float(np.mean(audio ** 2)) or 1e-8
            audio = (audio + rng.normal(0, np.sqrt(p / 10 ** (snr / 10)), audio.size)).astype(np.float32)
        if not audio.size or len(audio) > CFG["max_duration"] * 16000:
            return None
        dst = out_dir / f"{split}_{i:05d}.wav"
        sf.write(str(dst), np.clip(audio, -1, 1), 16000, subtype="PCM_16")
        return {"audio_filepath": str(dst), "duration": round(len(audio) / 16000, 3), "text": r["text"],
                "kind": r["kind"]}

    with ThreadPoolExecutor(max_workers=(os.cpu_count() or 2) * 2) as ex:
        items = [x for x in ex.map(one, enumerate(rows), chunksize=32) if x is not None]
    man = out_dir.parent / f"{split}.json"
    man.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n", encoding="utf-8")
    print(split, len(items), "of", len(rows), flush=True)
    return man


def read_manifest(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# %% [markdown]
# ## Training

# %%
def train(train_man: Path, dev_cv: list[dict], dev_synth: list[dict], work: Path, max_steps: int,
          accelerator: str = "gpu", precision: str = "16-mixed") -> dict:
    import lightning.pytorch as pl
    import torch
    from nemo.collections.asr.models import ASRModel

    from iphasr.metrics import corpus_wer

    pl.seed_everything(CFG["seed"])
    t0 = time.time()
    model = ASRModel.from_pretrained(CFG["base"], map_location="cpu")

    def dev_wer(rows: list[dict]) -> float:
        out = model.transcribe([r["audio_filepath"] for r in rows], batch_size=16, verbose=False)
        hyps = [o.text if hasattr(o, "text") else str(o) for o in out]
        return corpus_wer([r["text"] for r in rows], hyps)

    if accelerator == "gpu":
        model = model.cuda()
    model.eval()
    metrics = {"config": CFG, "dev_cv_wer_base": dev_wer(dev_cv), "dev_synth_wer_base": dev_wer(dev_synth),
               "evals": []}
    print("base dev WER cv", metrics["dev_cv_wer_base"], "synth", metrics["dev_synth_wer_base"], flush=True)

    layers = model.encoder.layers
    for m in [model.preprocessor, model.encoder.pre_encode, *layers[: CFG["freeze_lower_layers"]]]:
        for p in m.parameters():
            p.requires_grad = False
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print("trainable params M", round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 1),
          "of encoder layers", len(layers), flush=True)

    best = {"wer": metrics["dev_cv_wer_base"], "step": 0, "state": None}
    if SMOKE:   # 4 dev clips can score 0% at base: force the save/restore path to run
        best["wer"] = float("inf")

    class DevEval(pl.Callback):
        """Every eval_every optimizer steps: dev WER (Common Voice, G.711) with the project's normalizer."""
        def __init__(self):
            self.done, self.bad = set(), 0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            step = trainer.global_step
            if time.time() - t0 > CFG["max_hours"] * 3600:
                print("time limit: stopping training", flush=True)
                trainer.should_stop = True
            if step == 20 and accelerator == "gpu":
                print("peak GPU memory GiB:", round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), flush=True)
            if step == 0 or step % CFG["eval_every"] or step in self.done:
                return
            self.done.add(step)
            w = dev_wer(dev_cv)          # transcribe() switches to eval mode and restores train mode itself
            pl_module.train()
            metrics["evals"].append({"step": step, "dev_cv_wer": w, "elapsed_min": round((time.time() - t0) / 60, 1)})
            print("step", step, "dev cv WER", round(w, 4), flush=True)
            if w < best["wer"]:
                best.update(wer=w, step=step,
                            state={n: p.detach().to("cpu", copy=True) for n, p in pl_module.named_parameters()
                                   if n in trainable})
                self.bad = 0
            else:
                self.bad += 1
                if self.bad >= CFG["early_stop_patience"]:
                    print("early stop", flush=True)
                    trainer.should_stop = True

    trainer = pl.Trainer(devices=1, accelerator=accelerator, precision=precision, max_steps=max_steps,
                         accumulate_grad_batches=CFG["grad_accum"], logger=False, enable_checkpointing=False,
                         limit_val_batches=0, num_sanity_val_steps=0, callbacks=[DevEval()],
                         enable_progress_bar=False, log_every_n_steps=50)
    model.set_trainer(trainer)
    model.setup_training_data({"manifest_filepath": str(train_man), "sample_rate": 16000,
                               "batch_size": CFG["batch_size"], "shuffle": True, "num_workers": CFG["num_workers"],
                               "pin_memory": True, "max_duration": CFG["max_duration"], "trim_silence": False})
    model.setup_optimization({"name": "adamw", "lr": CFG["lr"], "weight_decay": CFG["weight_decay"],
                              "sched": {"name": "CosineAnnealing", "warmup_steps": CFG["warmup_steps"],
                                        "min_lr": CFG["min_lr"], "max_steps": max_steps}})
    if SMOKE:
        # NeMo's TDT loss is CUDA-only. The CPU pre-flight swaps the loss alone (encoder output energy), so the
        # data path, optimizer, scheduler, dev-eval callback, best-state restore and save_to still run.
        # The real TDT loss already trained on a T4 in the trial kernels.
        def smoke_step(batch, batch_idx):
            signal, signal_len = batch[0], batch[1]
            enc, _ = model.forward(input_signal=signal, input_signal_length=signal_len)
            return {"loss": enc.float().pow(2).mean()}
        model.training_step = smoke_step
    trainer.fit(model)
    metrics["steps_done"] = trainer.global_step
    metrics["train_minutes"] = round((time.time() - t0) / 60, 1)

    # best dev checkpoint (or the base, if no eval beat it)
    if best["state"] is not None:
        with torch.no_grad():
            params = dict(model.named_parameters())
            for n, v in best["state"].items():
                params[n].copy_(v.to(params[n].device))
    model.eval()
    metrics.update(best_step=best["step"], dev_cv_wer=dev_wer(dev_cv), dev_synth_wer=dev_wer(dev_synth))
    print(json.dumps({k: v for k, v in metrics.items() if k != "config"}, indent=1), flush=True)
    (work / "dev_metrics.json").write_text(json.dumps(metrics, indent=1))
    model.save_to(str(work / f"{CFG['out_name']}.nemo"))
    print("saved", work / f"{CFG['out_name']}.nemo", flush=True)
    return metrics


# %% [markdown]
# ## Data: Kaggle run, or a few local clips for the CPU pre-flight

# %%
if SMOKE:
    # a handful of local clips through the same prepare() + train() path, 3 optimizer steps, eval every 2
    D = Path(os.environ.get("IPASR_DATA", "D:/data/italian-phone-asr"))
    WORK = Path(os.environ["IPASR_SMOKE_DIR"])
    cv = [json.loads(l) for l in (D / "manifests" / "test_cv.jsonl").read_text(encoding="utf-8").splitlines()][:20]
    sy = [json.loads(l) for l in (D / "manifests" / "train_synth.jsonl").read_text(encoding="utf-8").splitlines()][:20]
    rows = {"train": [{"audio": r["audio"], "text": r["text"], "kind": "cv", "loader": "g711"} for r in cv[:12]]
                     + [{"audio": r["audio"], "text": r["text"], "kind": "synth", "loader": "ulaw"} for r in sy[:12]],
            "dev": [{"audio": r["audio"], "text": r["text"], "kind": "cv", "loader": "g711"} for r in cv[12:16]]
                   + [{"audio": r["audio"], "text": r["text"], "kind": "synth", "loader": "ulaw"} for r in sy[12:16]]}
    CFG.update(batch_size=2, grad_accum=2, eval_every=2, warmup_steps=1, num_workers=0)  # Windows: no fork
    tm = prepare(rows["train"], "train", WORK / "wav")
    dm = read_manifest(prepare(rows["dev"], "dev", WORK / "wav"))
    train(tm, [r for r in dm if r["kind"] == "cv"], [r for r in dm if r["kind"] == "synth"], WORK, max_steps=4,
          accelerator="cpu", precision="32")
    sys.exit(0)

# %% Kaggle: Common Voice download (only the subset), synthetic clips from the dataset
import tarfile

import requests

IN = next(p.parent for p in Path("/kaggle/input").rglob("cv_subset.json"))
WORK = Path("/kaggle/working")
TMP = Path("/kaggle/tmp")
TMP.mkdir(parents=True, exist_ok=True)
CV_ID = "cmu5xagm300dyo1075qdhf20m"  # Common Voice Scripted Speech 27.0 - Italian (CC0-1.0)
subset = json.loads((IN / "cv_subset.json").read_text())
wanted = {p for s in ("train", "dev") for p in subset[s]}
CV_DIR = TMP / "cv"
if not (CV_DIR / "done").exists():
    from kaggle_secrets import UserSecretsClient
    key = UserSecretsClient().get_secret("MDC_API_KEY")
    r = requests.post(f"https://mozilladatacollective.com/api/datasets/{CV_ID}/download",
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, timeout=60)
    r.raise_for_status()
    CV_DIR.mkdir(exist_ok=True)
    with requests.get(r.json()["downloadUrl"], stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
            n = 0
            for m in tar:
                if m.isfile() and Path(m.name).name in wanted:
                    (CV_DIR / Path(m.name).name).write_bytes(tar.extractfile(m).read())
                    n += 1
    (CV_DIR / "done").write_text(str(n))
print("cv clips:", len(list(CV_DIR.glob("*.mp3"))), "of", len(wanted), flush=True)

cv_meta = {r["path"]: r for r in subset["meta"]}
rows = {"train": [], "dev": []}
for split in ("train", "dev"):
    for p in subset[split]:
        if (CV_DIR / p).exists():
            rows[split].append({"audio": str(CV_DIR / p), "text": cv_meta[p]["sentence"], "kind": "cv", "loader": "g711"})
synth = [json.loads(l) for l in (IN / "train_synth.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
WAV = {w.name: w for w in Path("/kaggle/input").rglob("synth*.wav")}
assert all(Path(r["audio"]).name in WAV for r in synth), "synth clips missing from the dataset"
random.Random(0).shuffle(synth)   # same split as the Whisper rounds: first 200 = synthetic dev
for k, r in enumerate(synth):
    rows["dev" if k < 200 else "train"].append(
        {"audio": str(WAV[Path(r["audio"]).name]), "text": r["text"], "kind": "synth", "loader": "ulaw"})
print({k: len(v) for k, v in rows.items()}, flush=True)

train_man = prepare(rows["train"], "train", TMP / "wav")
dev = read_manifest(prepare(rows["dev"], "dev", TMP / "wav"))
n_train = len(read_manifest(train_man))
steps = CFG["epochs"] * -(-n_train // (CFG["batch_size"] * CFG["grad_accum"]))
print("optimizer steps", steps, flush=True)
train(train_man, [r for r in dev if r["kind"] == "cv"], [r for r in dev if r["kind"] == "synth"], WORK, steps)
