# %% [markdown]
# # Trial run (~20 min): seconds per step and peak memory of two training configs, before round 5
#
# Round 4 assumed a speed it never measured and reached 0.8 epochs in 10.5 h. This kernel measures it.
# Same model, LoRA modules, precision and collator as kaggle/train_lora.py; only the batch layout differs:
#   A = current:  batch 4 x accum 4, gradient checkpointing on
#   B = proposed: batch 2 x accum 8, gradient checkpointing off  (same effective batch, 16)
# Whisper pads every clip to 30 s, so the encoder cost per clip does not depend on the clip: a few hundred
# synthetic clips (already in the attached dataset, no Common Voice download, no secret) measure it.
# Also timed: one in-training eval pass (generate) on 200 dev clips, as round 5 would run every 200 steps.
# Output: /kaggle/working/trial.json

# %%
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft>=0.13", "jiwer>=3.1", "num2words>=0.5.14",
                "git+https://github.com/albedimaria/italian-phone-asr.git"], check=True)

# %%
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback,
                          WhisperForConditionalGeneration, WhisperProcessor)

IN = next(p.parent for p in Path("/kaggle/input").rglob("train_synth.jsonl"))
WORK = Path("/kaggle/working")
BASE = "openai/whisper-large-v3-turbo"
TARGETS = r"model\.encoder\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|fc1|fc2)"
MEASURED_STEPS, WARMUP_STEPS = 8, 2       # optimizer steps; the first ones include CUDA warm-up
CONFIGS = {"A_bs4_acc4_ckpt": dict(batch_size=4, grad_accum=4, ckpt=True),
           "B_bs2_acc8_nockpt": dict(batch_size=2, grad_accum=8, ckpt=False)}

processor = WhisperProcessor.from_pretrained(BASE, language="italian", task="transcribe")
WAV = {w.name: w for w in Path("/kaggle/input").rglob("synth*.wav")}
rows = [json.loads(l) for l in (IN / "train_synth.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
random.Random(0).shuffle(rows)


def ulaw_16k(path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ar", "16000", "-ac", "1", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def feat(r):
    x = processor.feature_extractor(ulaw_16k(WAV[Path(r["audio"]).name]), sampling_rate=16000,
                                    return_tensors="np").input_features[0]
    return {"input_features": torch.tensor(x, dtype=torch.float16), "labels": processor.tokenizer(r["text"]).input_ids}


n_train = 16 * (MEASURED_STEPS + WARMUP_STEPS)
train = [feat(r) for r in rows[:n_train]]
dev = [feat(r) for r in rows[n_train:n_train + 200]]
print("clips", len(train), len(dev), flush=True)


class Collator:
    def __init__(self, start): self.start = start
    def __call__(self, batch):
        x = torch.stack([b["input_features"].float() for b in batch])
        lab = processor.tokenizer.pad({"input_ids": [b["labels"] for b in batch]}, return_tensors="pt")
        y = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
        if (y[:, 0] == self.start).all():
            y = y[:, 1:]
        return {"input_features": x, "labels": y}


class DS(torch.utils.data.Dataset):
    def __init__(self, f): self.f = f
    def __len__(self): return len(self.f)
    def __getitem__(self, i): return self.f[i]


class StepTimer(TrainerCallback):
    def __init__(self): self.t = []
    def on_step_end(self, args, state, control, **kw):
        torch.cuda.synchronize(); self.t.append(time.time())


results = {"gpu": torch.cuda.get_device_name(0), "measured_steps": MEASURED_STEPS}
for name, c in CONFIGS.items():
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        model = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float32)
        model.generation_config.language, model.generation_config.task = "italian", "transcribe"
        model.generation_config.forced_decoder_ids = None
        model.config.use_cache = False
        model.config.apply_spec_augment, model.config.mask_time_prob, model.config.mask_feature_prob = True, 0.05, 0.05
        model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, target_modules=TARGETS,
                                                 bias="none"))
        args = Seq2SeqTrainingArguments(
            output_dir=str(WORK / "tmp" / name), per_device_train_batch_size=c["batch_size"],
            gradient_accumulation_steps=c["grad_accum"], learning_rate=2e-4, warmup_steps=0,
            max_steps=MEASURED_STEPS + WARMUP_STEPS, fp16=True, logging_steps=1, save_strategy="no",
            per_device_eval_batch_size=8, remove_unused_columns=False, label_names=["labels"], report_to=[],
            dataloader_num_workers=2, predict_with_generate=True, generation_max_length=200,
            gradient_checkpointing=c["ckpt"], gradient_checkpointing_kwargs={"use_reentrant": False})
        timer = StepTimer()
        trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=DS(train), eval_dataset=DS(dev),
                                 data_collator=Collator(model.config.decoder_start_token_id), callbacks=[timer])
        trainer.train()
        steps = np.diff(timer.t[WARMUP_STEPS - 1:])
        train_peak = torch.cuda.max_memory_allocated() / 1e9
        t0 = time.time()
        trainer.evaluate()
        results[name] = {**c, "s_per_step": round(float(np.mean(steps)), 1),
                         "s_per_step_all": [round(float(s), 1) for s in steps],
                         "peak_gb_train": round(train_peak, 1),
                         "peak_gb_with_eval": round(torch.cuda.max_memory_allocated() / 1e9, 1),
                         "eval_200_s": round(time.time() - t0, 1)}
    except torch.cuda.OutOfMemoryError as e:
        results[name] = {**c, "error": "OOM", "detail": str(e)[:300]}
    print(name, json.dumps(results[name]), flush=True)
    for v in ("trainer", "model"):
        globals().pop(v, None)
    gc.collect(); torch.cuda.empty_cache()

(WORK / "trial.json").write_text(json.dumps(results, indent=1))
print(json.dumps(results, indent=1))
