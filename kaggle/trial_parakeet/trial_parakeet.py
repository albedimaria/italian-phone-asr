# %% [markdown]
# # Trial run 3: memory margin of the Parakeet setup that fits a Kaggle T4
#
# Trial 1: full fine-tune OOM (batch 8 x 2, 16 x 1). Trial 2 (clips 8-14 s): only B fits, lower 12 of 24
# encoder layers frozen, batch 4 x accum 4, 14.34 GB of 14.56, 2.1 s/step; full fine-tune OOM even at
# batch 2 (weights + grads + AdamW states of 627 M params dominate). This kernel measures the margin:
#   B12        = trial-2 B, on clips at the real maximum (10-11 s: Common Voice tops out at 10.6 s)
#   B12_stress = B12 on 13-14 s clips (worse than any training clip)
#   B16        = 16 of 24 layers frozen
#   B12_8bit   = B12 with 8-bit AdamW (bitsandbytes): optimizer states 4x smaller
#   FULL_8bit  = full fine-tune with 8-bit AdamW, batch 4 x 4
# Same effective batch 16, fp16. Each config runs in its own process, so one OOM cannot leak into the next.
# The final setup is Alberto's call. Data: synthetic phone clips concatenated in pairs; no Common Voice
# download, no secret.
# Output: /kaggle/working/trial.json

# %%
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

t_install = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nemo_toolkit[asr]", "bitsandbytes"], check=True)
t_install = time.time() - t_install

import numpy as np
import soundfile as sf

IN = next(p.parent for p in Path("/kaggle/input").rglob("train_synth.jsonl"))
WORK = Path("/kaggle/working")
AUD = Path("/kaggle/tmp/wav16")
AUD.mkdir(parents=True, exist_ok=True)
MEASURED_STEPS, WARMUP_STEPS, EFFECTIVE = 8, 2, 16

WAV = {w.name: w for w in Path("/kaggle/input").rglob("synth*.wav")}
rows = [json.loads(l) for l in (IN / "train_synth.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
random.Random(0).shuffle(rows)
cache: dict[str, np.ndarray] = {}


def load16k(r) -> np.ndarray:
    """Stored synth clips are mu-law 8 kHz: decode + resample to 16 kHz PCM, the rate Parakeet expects."""
    if r["audio"] not in cache:
        src = WAV[Path(r["audio"]).name]
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-ar", "16000", "-ac", "1", "-f", "s16le",
                              "-"], check=True, capture_output=True).stdout
        cache[r["audio"]] = np.frombuffer(raw, dtype=np.int16)
    return cache[r["audio"]]


n_clips = EFFECTIVE * (MEASURED_STEPS + WARMUP_STEPS)
gap = np.zeros(4000, dtype=np.int16)   # 0.25 s of silence between the two sentences


def build(name: str, lo: float, hi: float) -> tuple[Path, list[float]]:
    """Pairs of clips padded with trailing silence to at least `lo` s; pairs longer than `hi` are skipped."""
    out, k = [], 0
    while len(out) < n_clips:
        a, b = rows[k % len(rows)], rows[(k + 1) % len(rows)]
        k += 2
        audio = np.concatenate([load16k(a), gap, load16k(b)])
        if len(audio) < lo * 16000:
            audio = np.concatenate([audio, np.zeros(int(lo * 16000) - len(audio), dtype=np.int16)])
        dur = len(audio) / 16000
        if dur > hi:
            continue
        dst = AUD / f"{name}{len(out):04d}.wav"
        sf.write(str(dst), audio, 16000, subtype="PCM_16")
        out.append({"audio_filepath": str(dst), "duration": round(dur, 2), "text": f"{a['text']} {b['text']}"})
    man = WORK / f"manifest_{name}.json"
    man.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    durs = [r["duration"] for r in out]
    print(name, "clips", len(out), "dur min/mean/max", min(durs), round(float(np.mean(durs)), 1), max(durs),
          flush=True)
    return man, durs


MAN = {"real_max": build("real_max", 10.0, 11.0), "stress": build("stress", 13.0, 14.0)}

CONFIGS = {
    "B12": dict(batch_size=4, grad_accum=4, freeze_n=12, data="real_max"),
    "B12_stress": dict(batch_size=4, grad_accum=4, freeze_n=12, data="stress"),
    "B16": dict(batch_size=4, grad_accum=4, freeze_n=16, data="real_max"),
    "B12_8bit": dict(batch_size=4, grad_accum=4, freeze_n=12, opt8bit=True, data="real_max"),
    "FULL_8bit": dict(batch_size=4, grad_accum=4, opt8bit=True, data="real_max"),
}

WORKER = r'''
import json, sys, time
import numpy as np
import torch
import lightning.pytorch as pl
from nemo.collections.asr.models import ASRModel

name, c, man, out, steps, warm = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
res = dict(c)


class StepTimer(pl.Callback):
    def __init__(self): self.t = []
    def on_before_zero_grad(self, trainer, pl_module, optimizer):   # once per optimizer step
        torch.cuda.synchronize(); self.t.append(time.time())


try:
    torch.cuda.reset_peak_memory_stats()
    model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    if c.get("freeze_n"):
        layers = model.encoder.layers
        for m in [model.preprocessor, model.encoder.pre_encode, *layers[: c["freeze_n"]]]:
            for p in m.parameters():
                p.requires_grad = False
        res["encoder_layers"] = len(layers)
    res["trainable_params_M"] = round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 1)
    model.setup_training_data({"manifest_filepath": man, "sample_rate": 16000, "batch_size": c["batch_size"],
                               "shuffle": True, "num_workers": 2, "pin_memory": True, "max_duration": 30.0,
                               "trim_silence": False})
    model.setup_optimization({"name": "adamw", "lr": 1e-4, "weight_decay": 1e-3})
    if c.get("opt8bit"):
        import bitsandbytes as bnb
        model._optimizer = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad],
                                               lr=1e-4, weight_decay=1e-3)
        model._scheduler = None
    res["optimizer"] = type(model._optimizer).__name__
    timer = StepTimer()
    trainer = pl.Trainer(devices=1, accelerator="gpu", precision="16-mixed", max_steps=steps + warm,
                         accumulate_grad_batches=c["grad_accum"], logger=False, enable_checkpointing=False,
                         limit_val_batches=0, num_sanity_val_steps=0, callbacks=[timer], enable_progress_bar=False)
    model.set_trainer(trainer)
    trainer.fit(model)
    s = np.diff(timer.t[warm - 1:])
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    res.update(s_per_step=round(float(np.mean(s)), 2), optimizer_steps=len(timer.t), peak_gb_train=round(peak, 2),
               margin_gb=round(total - peak, 2), fits=True)
except torch.cuda.OutOfMemoryError as e:
    res.update(error="OOM", detail=str(e)[:300], peak_gb_train=round(torch.cuda.max_memory_allocated() / 1e9, 2),
               fits=False)
except Exception as e:   # a NeMo/Lightning/bitsandbytes API mismatch is a finding too: record it
    res.update(error=type(e).__name__, detail=str(e)[:500], fits=None)
open(out, "w").write(json.dumps(res))
'''
Path("/kaggle/tmp/worker.py").write_text(WORKER)

results = {"install_s": round(t_install), "measured_steps": MEASURED_STEPS, "effective_batch": EFFECTIVE,
           "clip_seconds": {k: {"min": min(d), "mean": round(float(np.mean(d)), 1), "max": max(d)}
                            for k, (_, d) in MAN.items()}}
env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
for name, c in CONFIGS.items():
    out = f"/kaggle/tmp/{name}.json"
    t0 = time.time()
    p = subprocess.run([sys.executable, "/kaggle/tmp/worker.py", name, json.dumps(c), str(MAN[c["data"]][0]), out,
                        str(MEASURED_STEPS), str(WARMUP_STEPS)], env=env, capture_output=True, text=True)
    r = json.loads(Path(out).read_text()) if Path(out).exists() else {
        **c, "error": f"worker exit {p.returncode}", "detail": p.stderr[-800:], "fits": None}
    r["wall_s"] = round(time.time() - t0)
    results[name] = r
    print(name, json.dumps(r), flush=True)
    (WORK / "trial.json").write_text(json.dumps(results, indent=1))

import torch
results["gpu"] = torch.cuda.get_device_name(0)
(WORK / "trial.json").write_text(json.dumps(results, indent=1))
print(json.dumps(results, indent=1))
