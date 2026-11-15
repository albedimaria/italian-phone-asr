# %% [markdown]
# # Whisper LoRA on 8 kHz G.711 Italian phone speech
#
# Kaggle script kernel (GPU T4 x1 or P100). Inputs:
# - Kaggle secret `MDC_API_KEY`: Mozilla Data Collective key. Common Voice cannot be re-hosted, so the
#   kernel downloads it itself and keeps only the subset listed in `cv_subset.json`.
# - Kaggle dataset `ipasr-train` (private, our own data): `train_synth/` mu-law WAVs + manifests +
#   `cv_subset.json` + `config.json` (hyper-parameters of this round).
#
# Output in /kaggle/working: `<out_name>/` (LoRA merged into the base weights, ready for
# `pipeline("automatic-speech-recognition")`), `adapter/`, `dev_metrics.json`, checkpoints for resume.

# %%
import os
import subprocess
import sys

# the image ships torchao 0.10, which recent PEFT rejects at LoRA injection: remove it (not used here)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft>=0.13", "jiwer>=3.1", "num2words>=0.5.14",
                "git+https://github.com/albedimaria/italian-phone-asr.git"], check=True)

# %%
import json
import random
import tarfile
import zlib
from pathlib import Path

import numpy as np
import requests
import torch

# the mount path of an attached dataset varies between Kaggle images: locate it by content
IN = next(p.parent for p in Path("/kaggle/input").rglob("cv_subset.json"))
WORK = Path("/kaggle/working")
TMP = Path("/kaggle/tmp")
TMP.mkdir(parents=True, exist_ok=True)
# The config lives here, not in the dataset: the API token cannot version datasets (403), so a
# dataset config.json goes stale. It is printed and saved with the metrics instead.
CFG = {
    "round": 4, "base": "openai/whisper-large-v3-turbo", "out_name": "whisper-turbo-it-phone",
    # Rounds 1-3 (whisper-small, LoRA on q/v of encoder AND decoder, lr 1e-3) drifted on spelling
    # ("bongiorno", "perfecto"): the problem is acoustic, so adapt the encoder only.
    "lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.05,
    "target_modules": r"model\.encoder\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|fc1|fc2)",
    "lr": 2e-4, "epochs": 3, "batch_size": 4, "grad_accum": 4, "synth_fraction": 1.0, "warmup_steps": 100,
    "eval_steps": 100, "save_steps": 100, "early_stop_patience": 5, "seed": 0,
    # keep the checkpoint with the lowest dev WER (generated text), not the lowest eval loss
    "select_by": "wer", "eval_subset": 400,
    # augmentation: SpecAugment in the encoder + additive noise after the codec on half the train clips
    "spec_augment": True, "noise_p": 0.5, "noise_snr_db": [15, 35],
    "max_hours": 10.5,
    # ablation arm: "both" | "cv" (Common Voice through G.711 only) | "synth" (synthetic phone set only)
    "data_mix": "both",   # Kaggle stops at 12 h: stop training early enough to evaluate and save
}
print(CFG)
random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])

# %% [markdown]
# ## Common Voice IT: download once per session, extract only the subset

# %%
CV_ID = "cmu5xagm300dyo1075qdhf20m"  # Common Voice Scripted Speech 27.0 - Italian (CC0-1.0)
subset = json.loads((IN / "cv_subset.json").read_text())   # {"train": [...paths], "dev": [...paths]}
wanted = {p: s for s in ("train", "dev") for p in subset[s]}
CV_DIR = TMP / "cv"
if not (CV_DIR / "done").exists():
    from kaggle_secrets import UserSecretsClient
    key = UserSecretsClient().get_secret("MDC_API_KEY")
    r = requests.post(f"https://mozilladatacollective.com/api/datasets/{CV_ID}/download",
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, timeout=60)
    r.raise_for_status()
    url = r.json()["downloadUrl"]
    CV_DIR.mkdir(exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:   # single streaming pass
            n = 0
            for m in tar:
                name = Path(m.name).name
                if m.isfile() and name in wanted:
                    (CV_DIR / name).write_bytes(tar.extractfile(m).read())
                    n += 1
    (CV_DIR / "done").write_text(str(n))
print("cv clips:", len(list(CV_DIR.glob("*.mp3"))), "of", len(wanted))

# %% [markdown]
# ## The G.711 chain: 8 kHz mono -> mu-law -> decode -> 16 kHz

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


# %%
def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]

cv_meta = {r["path"]: r for r in subset["meta"]}
rows = {"train": [], "dev": []}
for split in ("train", "dev"):
    for p in subset[split]:
        f = CV_DIR / p
        if f.exists():
            rows[split].append({"audio": str(f), "text": cv_meta[p]["sentence"], "kind": "cv", "loader": "g711"})
synth = read_jsonl(IN / "train_synth.jsonl")  # round-trip-filtered clips only
# audio may sit in train_synth/ or in several uploaded parts (web upload caps file size): find it anywhere
WAV = {w.name: w for w in Path("/kaggle/input").rglob("synth*.wav")}
missing = [r["audio"] for r in read_jsonl(IN / "train_synth.jsonl") if Path(r["audio"]).name not in WAV]
assert not missing, f"{len(missing)} synth clips not found, e.g. {missing[:3]}"
random.Random(0).shuffle(synth)
n_dev = 200
for r in synth[:n_dev]:
    rows["dev"].append({"audio": str(WAV[Path(r["audio"]).name]), "text": r["text"], "kind": "synth",
                        "loader": "ulaw"})
keep = int(len(synth[n_dev:]) * CFG["synth_fraction"])
for r in synth[n_dev:n_dev + keep]:
    rows["train"].append({"audio": str(WAV[Path(r["audio"]).name]), "text": r["text"], "kind": "synth",
                          "loader": "ulaw"})
if CFG["data_mix"] != "both":   # dev keeps both kinds, so every arm is scored on the same dev set
    rows["train"] = [r for r in rows["train"] if r["kind"] == CFG["data_mix"]]
print({k: len(v) for k, v in rows.items()})

# %% [markdown]
# ## Features (cached to disk so a resumed session skips this)

# %%
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE = CFG["base"]
processor = WhisperProcessor.from_pretrained(BASE, language="italian", task="transcribe")

# build the LoRA model before any data work, so a library problem fails in a minute, not after featurizing
from peft import LoraConfig, get_peft_model
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

model = WhisperForConditionalGeneration.from_pretrained(BASE)
model.generation_config.language = "italian"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.use_cache = False
if CFG["spec_augment"]:
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = 0.05
    model.config.mask_feature_prob = 0.05
lora = LoraConfig(r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"], lora_dropout=CFG["lora_dropout"],
                  target_modules=CFG["target_modules"], bias="none")
model = get_peft_model(model, lora)
model.print_trainable_parameters()



FEAT = TMP / "features"
FEAT.mkdir(exist_ok=True)

def _one(r, split="train"):
    audio = g711_16k(r["audio"]) if r["loader"] == "g711" else ulaw_16k(r["audio"])
    rng = np.random.default_rng([zlib.crc32(r["audio"].encode()), CFG["seed"]])
    if split == "train" and rng.random() < CFG["noise_p"] and audio.size:
        snr = rng.uniform(*CFG["noise_snr_db"])
        p = float(np.mean(audio ** 2)) or 1e-8
        audio = (audio + rng.normal(0, np.sqrt(p / 10 ** (snr / 10)), audio.size)).astype(np.float32)
    if len(audio) > 30 * 16000:
        return None
    x = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="np").input_features[0]
    return {"input_features": torch.tensor(x, dtype=torch.float16), "labels": processor.tokenizer(r["text"]).input_ids,
            "kind": r["kind"], "text": r["text"]}


def featurize(split):
    """ffmpeg per clip is the bottleneck (47 min serial for 12k clips): fan out over all CPUs."""
    from concurrent.futures import ThreadPoolExecutor
    out = FEAT / f"{split}.pt"
    if out.exists():
        return torch.load(out, weights_only=False)
    feats = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as ex:
        for k, f in enumerate(ex.map(lambda r: _one(r, split), rows[split], chunksize=32)):
            if f is not None:
                feats.append(f)
            if k % 2000 == 0:
                print(split, k, len(rows[split]), flush=True)
    torch.save(feats, out)
    return feats

train_feats, dev_feats = featurize("train"), featurize("dev")

# %% [markdown]
# ## Training

# %%
class Collator:
    def __call__(self, batch):
        x = torch.stack([b["input_features"].float() for b in batch])
        lab = processor.tokenizer.pad({"input_ids": [b["labels"] for b in batch]}, return_tensors="pt")
        y = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
        if (y[:, 0] == model.config.decoder_start_token_id).all():
            y = y[:, 1:]
        return {"input_features": x, "labels": y}


class DS(torch.utils.data.Dataset):
    def __init__(self, f): self.f = f
    def __len__(self): return len(self.f)
    def __getitem__(self, i): return self.f[i]


from iphasr.metrics import corpus_wer

by_wer = CFG["select_by"] == "wer"
# a fixed random slice of dev keeps each in-training generate pass to a few minutes
eval_feats = random.Random(1).sample(dev_feats, min(CFG["eval_subset"], len(dev_feats))) if by_wer else dev_feats


def compute_metrics(pred):
    ids = pred.label_ids.copy()
    ids[ids == -100] = processor.tokenizer.pad_token_id
    refs = processor.batch_decode(ids, skip_special_tokens=True)
    hyps = processor.batch_decode(pred.predictions, skip_special_tokens=True)
    return {"wer": corpus_wer(refs, hyps)}


args = Seq2SeqTrainingArguments(
    output_dir=str(WORK / "ckpt"), per_device_train_batch_size=CFG["batch_size"],
    gradient_accumulation_steps=CFG["grad_accum"], learning_rate=CFG["lr"], warmup_steps=CFG["warmup_steps"],
    num_train_epochs=CFG["epochs"], fp16=True, eval_strategy="steps", eval_steps=CFG["eval_steps"],
    save_steps=CFG["save_steps"], save_total_limit=2, logging_steps=50, per_device_eval_batch_size=8,
    remove_unused_columns=False, label_names=["labels"], report_to=[], dataloader_num_workers=2,
    load_best_model_at_end=True, metric_for_best_model="eval_wer" if by_wer else "eval_loss",
    greater_is_better=False, predict_with_generate=by_wer, generation_max_length=200,
    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
)
import time
from transformers import EarlyStoppingCallback, TrainerCallback

T0 = time.time()


class TimeLimit(TrainerCallback):
    def on_step_end(self, args, state, control, **kw):
        if time.time() - T0 > CFG["max_hours"] * 3600:
            print("time limit: stopping training", flush=True)
            control.should_training_stop = True
        if state.global_step == 20 and torch.cuda.is_available():
            print("peak GPU memory GB:", round(torch.cuda.max_memory_allocated() / 1e9, 1), flush=True)


trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=DS(train_feats), eval_dataset=DS(eval_feats),
                         data_collator=Collator(), compute_metrics=compute_metrics if by_wer else None,
                         callbacks=[EarlyStoppingCallback(CFG["early_stop_patience"]), TimeLimit()])
last = None
if (WORK / "ckpt").exists():
    cks = sorted((WORK / "ckpt").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    last = str(cks[-1]) if cks else None
trainer.train(resume_from_checkpoint=last)

# %% [markdown]
# ## Dev WER with the project's normalizer, then merge and save

# %%
model.eval()
model.config.use_cache = True

def transcribe(feats, bs=16):
    out = []
    for k in range(0, len(feats), bs):
        x = torch.stack([f["input_features"].float() for f in feats[k:k + bs]]).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            ids = model.generate(input_features=x, language="italian", task="transcribe", max_new_tokens=200)
        out += processor.batch_decode(ids, skip_special_tokens=True)
    return out

metrics = {"config": CFG, "log_history": trainer.state.log_history}
for kind in ("cv", "synth"):
    sub = [f for f in dev_feats if f["kind"] == kind]
    hyps = transcribe(sub)
    metrics[f"dev_{kind}_wer"] = corpus_wer([f["text"] for f in sub], hyps)
with model.disable_adapter():
    for kind in ("cv", "synth"):
        sub = [f for f in dev_feats if f["kind"] == kind]
        metrics[f"dev_{kind}_wer_base"] = corpus_wer([f["text"] for f in sub], transcribe(sub))
print(json.dumps(metrics, indent=1))
(WORK / "dev_metrics.json").write_text(json.dumps(metrics, indent=1))

model.save_pretrained(WORK / "adapter")
merged = model.merge_and_unload()
OUT = WORK / CFG["out_name"]
merged.save_pretrained(OUT, safe_serialization=True)
processor.save_pretrained(OUT)
print("saved", list(OUT.iterdir()))
