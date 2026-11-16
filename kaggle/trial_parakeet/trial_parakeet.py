# %% [markdown]
# # Trial run: can parakeet-tdt-0.6b-v3 be fine-tuned on a Kaggle T4, and how fast?
#
# Parakeet is the strongest local system on the bench (4.02 % WER on test_cv_g711 vs 5.47 % for the
# round-4 Whisper LoRA) and 26x faster on CPU. Before spending a 10 h session on it, this kernel answers:
# does NeMo install on the Kaggle image, does a full fine-tune fit in 16 GB, and how many seconds per
# optimizer step. Italian is already in v3's tokenizer, so tokenizer, decoder and joint are kept.
# Configs (effective batch 16 for both, fp16 mixed precision, AdamW lr 1e-4 as in the NeMo docs):
#   A = batch 8 x accum 2      B = batch 16 x accum 1
# Data: a few hundred synthetic phone clips from the attached dataset (no Common Voice, no secret).
# Also: base WER on 200 held-out synthetic clips, so a later round has its starting point.
# Output: /kaggle/working/trial.json

# %%
import subprocess
import sys
import time

t_install = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nemo_toolkit[asr]", "jiwer>=3.1",
                "num2words>=0.5.14", "git+https://github.com/albedimaria/italian-phone-asr.git"], check=True)
t_install = time.time() - t_install

# %%
import gc
import json
import random
from pathlib import Path

import lightning.pytorch as pl
import torch
from nemo.collections.asr.models import ASRModel

from iphasr.metrics import corpus_wer

IN = next(p.parent for p in Path("/kaggle/input").rglob("train_synth.jsonl"))
WORK = Path("/kaggle/working")
AUD = Path("/kaggle/tmp/wav16")
AUD.mkdir(parents=True, exist_ok=True)
MEASURED_STEPS, WARMUP_STEPS = 8, 2
CONFIGS = {"A_bs8_acc2": dict(batch_size=8, grad_accum=2), "B_bs16_acc1": dict(batch_size=16, grad_accum=1)}

WAV = {w.name: w for w in Path("/kaggle/input").rglob("synth*.wav")}
rows = [json.loads(l) for l in (IN / "train_synth.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
random.Random(0).shuffle(rows)


def to16k(r) -> dict:
    """Stored synth clips are mu-law 8 kHz: decode + resample to 16 kHz PCM, the rate Parakeet expects."""
    src = WAV[Path(r["audio"]).name]
    dst = AUD / (src.stem + ".wav")
    if not dst.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                        str(dst)], check=True)
    import soundfile as sf
    return {"audio_filepath": str(dst), "duration": sf.info(str(dst)).duration, "text": r["text"]}


n_train = 16 * (MEASURED_STEPS + WARMUP_STEPS) * 2   # accum 2 for A: enough batches for both configs
train = [to16k(r) for r in rows[:n_train]]
dev = [to16k(r) for r in rows[n_train:n_train + 200]]
man = WORK / "train_manifest.json"
man.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n", encoding="utf-8")
print("clips", len(train), len(dev), "install s", round(t_install), flush=True)

results = {"gpu": torch.cuda.get_device_name(0), "install_s": round(t_install), "measured_steps": MEASURED_STEPS}

# %% base WER on the held-out synthetic clips
model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3").cuda().eval()
t0 = time.time()
out = model.transcribe([d["audio_filepath"] for d in dev], batch_size=16)
hyps = [o.text if hasattr(o, "text") else str(o) for o in out]
results["base_dev_synth_wer"] = round(corpus_wer([d["text"] for d in dev], hyps), 4)
results["base_dev_transcribe_200_s"] = round(time.time() - t0, 1)
print("base dev synth WER", results["base_dev_synth_wer"], flush=True)
del model; gc.collect(); torch.cuda.empty_cache()


class StepTimer(pl.Callback):
    def __init__(self): self.t = []
    def on_before_zero_grad(self, trainer, pl_module, optimizer):   # once per optimizer step
        torch.cuda.synchronize(); self.t.append(time.time())


for name, c in CONFIGS.items():
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
        model.setup_training_data({"manifest_filepath": str(man), "sample_rate": 16000, "batch_size": c["batch_size"],
                                   "shuffle": True, "num_workers": 2, "pin_memory": True, "max_duration": 30,
                                   "trim_silence": False})
        model.setup_optimization({"name": "adamw", "lr": 1e-4, "weight_decay": 1e-3})
        timer = StepTimer()
        trainer = pl.Trainer(devices=1, accelerator="gpu", precision="16-mixed", max_steps=MEASURED_STEPS + WARMUP_STEPS,
                             accumulate_grad_batches=c["grad_accum"], logger=False, enable_checkpointing=False,
                             limit_val_batches=0, num_sanity_val_steps=0, callbacks=[timer],
                             enable_progress_bar=False)
        model.set_trainer(trainer)
        trainer.fit(model)
        import numpy as np
        steps = np.diff(timer.t[WARMUP_STEPS - 1:])
        results[name] = {**c, "s_per_step": round(float(np.mean(steps)), 2),
                         "s_per_step_all": [round(float(s), 2) for s in steps],
                         "peak_gb_train": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    except torch.cuda.OutOfMemoryError as e:
        results[name] = {**c, "error": "OOM", "detail": str(e)[:300]}
    except Exception as e:   # a NeMo/Lightning API mismatch is a finding too: record it, keep going
        results[name] = {**c, "error": type(e).__name__, "detail": str(e)[:500]}
    print(name, json.dumps(results[name]), flush=True)
    for v in ("trainer", "model"):
        globals().pop(v, None)
    gc.collect(); torch.cuda.empty_cache()

(WORK / "trial.json").write_text(json.dumps(results, indent=1))
print(json.dumps(results, indent=1))
