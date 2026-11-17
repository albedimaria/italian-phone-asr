# %% [markdown]
# # Trial run 2: which lighter Parakeet fine-tune fits a Kaggle T4, and how fast?
#
# Trial 1 (full fine-tune, batch 8 x 2 and 16 x 1, fp16) ran out of memory: 14.2-14.5 GB of 14.6.
# This kernel measures seconds per optimizer step and peak memory for lighter setups, from the one that
# changes least. It does not pick the final setup: that is Alberto's call.
#   A = fp16, batch 4 x accum 4 (effective 16), full fine-tune
#   B = A with the preprocessor, subsampling and lower half of the encoder layers frozen
#   C = A with clips capped at 10 s (max_duration)
#   D = batch 8 x accum 2 with NeMo's fused joint (loss computed in sub-batches of 4): the TDT joint
#       tensor (batch x frames x tokens x vocab) is the usual memory peak
#   E = batch 2 x accum 8, full fine-tune (fallback if A does not fit)
# Each config runs in its own process, so one OOM cannot leak memory into the next.
# Data: synthetic phone clips concatenated in pairs to 8-15 s, so memory is measured at Common Voice
# lengths, not on short synthetic clips. No Common Voice download, no secret.
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
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nemo_toolkit[asr]"], check=True)
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


def load16k(r) -> np.ndarray:
    """Stored synth clips are mu-law 8 kHz: decode + resample to 16 kHz PCM, the rate Parakeet expects."""
    src = WAV[Path(r["audio"]).name]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-ar", "16000", "-ac", "1", "-f", "s16le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16)


n_clips = EFFECTIVE * (MEASURED_STEPS + WARMUP_STEPS)
train, k = [], 0
gap = np.zeros(4000, dtype=np.int16)   # 0.25 s of silence between the two sentences
while len(train) < n_clips:
    a, b = rows[k], rows[k + 1]; k += 2
    audio = np.concatenate([load16k(a), gap, load16k(b)])
    dur = len(audio) / 16000
    if not 8.0 <= dur <= 15.0:
        continue
    dst = AUD / f"pair{len(train):04d}.wav"
    sf.write(str(dst), audio, 16000, subtype="PCM_16")
    train.append({"audio_filepath": str(dst), "duration": round(dur, 2), "text": f"{a['text']} {b['text']}"})
man = WORK / "train_manifest.json"
man.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n", encoding="utf-8")
durs = [r["duration"] for r in train]
print("clips", len(train), "dur min/mean/max", min(durs), round(float(np.mean(durs)), 1), max(durs), flush=True)

CONFIGS = {
    "A_bs4_acc4": dict(batch_size=4, grad_accum=4),
    "B_bs4_acc4_freeze_lower_half": dict(batch_size=4, grad_accum=4, freeze_lower=True),
    "C_bs4_acc4_max10s": dict(batch_size=4, grad_accum=4, max_duration=10.0),
    "D_bs8_acc2_fused_joint4": dict(batch_size=8, grad_accum=2, fused_batch=4),
    "E_bs2_acc8": dict(batch_size=2, grad_accum=8),
}

WORKER = r'''
import json, sys, time
import numpy as np
import torch
import lightning.pytorch as pl
from nemo.collections.asr.models import ASRModel
from omegaconf import open_dict

name, c, man, out, steps, warm = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
res = dict(c)


class StepTimer(pl.Callback):
    def __init__(self): self.t = []
    def on_before_zero_grad(self, trainer, pl_module, optimizer):   # once per optimizer step
        torch.cuda.synchronize(); self.t.append(time.time())


try:
    torch.cuda.reset_peak_memory_stats()
    model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    if c.get("fused_batch"):
        with open_dict(model.cfg):
            model.cfg.joint.fuse_loss_wer = True
            model.cfg.joint.fused_batch_size = c["fused_batch"]
        model.joint.set_fuse_loss_wer(True)
        model.joint.set_fused_batch_size(c["fused_batch"])
        model.joint.set_loss(model.loss)
        model.joint.set_wer(model.wer)
    if c.get("freeze_lower"):
        layers = model.encoder.layers
        frozen = [model.preprocessor, model.encoder.pre_encode, *layers[: len(layers) // 2]]
        for m in frozen:
            for p in m.parameters():
                p.requires_grad = False
        res["encoder_layers"] = len(layers)
        res["frozen_layers"] = len(layers) // 2
    res["trainable_params_M"] = round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 1)
    model.setup_training_data({"manifest_filepath": man, "sample_rate": 16000, "batch_size": c["batch_size"],
                               "shuffle": True, "num_workers": 2, "pin_memory": True,
                               "max_duration": c.get("max_duration", 30.0), "trim_silence": False})
    model.setup_optimization({"name": "adamw", "lr": 1e-4, "weight_decay": 1e-3})
    timer = StepTimer()
    trainer = pl.Trainer(devices=1, accelerator="gpu", precision="16-mixed", max_steps=steps + warm,
                         accumulate_grad_batches=c["grad_accum"], logger=False, enable_checkpointing=False,
                         limit_val_batches=0, num_sanity_val_steps=0, callbacks=[timer], enable_progress_bar=False)
    model.set_trainer(trainer)
    trainer.fit(model)
    s = np.diff(timer.t[warm - 1:])
    res.update(s_per_step=round(float(np.mean(s)), 2), s_per_step_all=[round(float(x), 2) for x in s],
               optimizer_steps=len(timer.t), peak_gb_train=round(torch.cuda.max_memory_allocated() / 1e9, 2),
               fits=True)
except torch.cuda.OutOfMemoryError as e:
    res.update(error="OOM", detail=str(e)[:300], peak_gb_train=round(torch.cuda.max_memory_allocated() / 1e9, 2),
               fits=False)
except Exception as e:   # a NeMo/Lightning API mismatch is a finding too: record it
    res.update(error=type(e).__name__, detail=str(e)[:500], fits=None)
open(out, "w").write(json.dumps(res))
'''
Path("/kaggle/tmp/worker.py").write_text(WORKER)

results = {"install_s": round(t_install), "measured_steps": MEASURED_STEPS, "effective_batch": EFFECTIVE,
           "clip_seconds": {"min": min(durs), "mean": round(float(np.mean(durs)), 1), "max": max(durs)}}
env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
for name, c in CONFIGS.items():
    out = f"/kaggle/tmp/{name}.json"
    t0 = time.time()
    p = subprocess.run([sys.executable, "/kaggle/tmp/worker.py", name, json.dumps(c), str(man), out,
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
