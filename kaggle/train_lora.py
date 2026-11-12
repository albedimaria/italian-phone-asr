# %% [markdown]
# # Whisper-small LoRA on 8 kHz G.711 Italian phone speech
#
# Kaggle script kernel (GPU T4 x1 or P100). Inputs:
# - Kaggle secret `MDC_API_KEY`: Mozilla Data Collective key. Common Voice cannot be re-hosted, so the
#   kernel downloads it itself and keeps only the subset listed in `cv_subset.json`.
# - Kaggle dataset `ipasr-train` (private, our own data): `train_synth/` mu-law WAVs + manifests +
#   `cv_subset.json` + `config.json` (hyper-parameters of this round).
#
# Output in /kaggle/working: `whisper-small-it-phone/` (LoRA merged into the base weights, ready for
# `pipeline("automatic-speech-recognition")`), `adapter/`, `dev_metrics.json`, checkpoints for resume.

# %%
import os
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft>=0.13", "jiwer>=3.1", "num2words>=0.5.14",
                "git+https://github.com/albedimaria/italian-phone-asr.git"], check=True)

# %%
import json
import random
import tarfile
from pathlib import Path

import numpy as np
import requests
import torch

IN = Path("/kaggle/input/ipasr-train")
WORK = Path("/kaggle/working")
TMP = Path("/kaggle/tmp")
TMP.mkdir(parents=True, exist_ok=True)
CFG = {
    "round": 1, "lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"],
    "lr": 1e-3, "epochs": 3, "batch_size": 16, "grad_accum": 1, "synth_fraction": 1.0, "warmup_steps": 100,
    "eval_steps": 300, "save_steps": 300, "seed": 0,
}
if (IN / "config.json").exists():
    CFG.update(json.loads((IN / "config.json").read_text()))
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
synth = read_jsonl(IN / "train_synth.jsonl")
random.Random(0).shuffle(synth)
n_dev = 200
for r in synth[:n_dev]:
    rows["dev"].append({"audio": str(IN / "train_synth" / Path(r["audio"]).name), "text": r["text"], "kind": "synth",
                        "loader": "ulaw"})
keep = int(len(synth[n_dev:]) * CFG["synth_fraction"])
for r in synth[n_dev:n_dev + keep]:
    rows["train"].append({"audio": str(IN / "train_synth" / Path(r["audio"]).name), "text": r["text"], "kind": "synth",
                          "loader": "ulaw"})
print({k: len(v) for k, v in rows.items()})

# %% [markdown]
# ## Features (cached to disk so a resumed session skips this)

# %%
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(BASE, language="italian", task="transcribe")

FEAT = TMP / "features"
FEAT.mkdir(exist_ok=True)

def featurize(split):
    out = FEAT / f"{split}.pt"
    if out.exists():
        return torch.load(out)
    feats = []
    for k, r in enumerate(rows[split]):
        audio = g711_16k(r["audio"]) if r["loader"] == "g711" else ulaw_16k(r["audio"])
        if len(audio) > 30 * 16000:
            continue
        x = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="np").input_features[0]
        y = processor.tokenizer(r["text"]).input_ids
        feats.append({"input_features": torch.tensor(x, dtype=torch.float16), "labels": y, "kind": r["kind"],
                      "text": r["text"]})
        if k % 1000 == 0:
            print(split, k, len(rows[split]), flush=True)
    torch.save(feats, out)
    return feats

train_feats, dev_feats = featurize("train"), featurize("dev")

# %% [markdown]
# ## LoRA on q/v projections

# %%
from peft import LoraConfig, get_peft_model
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

model = WhisperForConditionalGeneration.from_pretrained(BASE)
model.generation_config.language = "italian"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.use_cache = False
lora = LoraConfig(r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"], lora_dropout=CFG["lora_dropout"],
                  target_modules=CFG["target_modules"], bias="none")
model = get_peft_model(model, lora)
model.print_trainable_parameters()


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


args = Seq2SeqTrainingArguments(
    output_dir=str(WORK / "ckpt"), per_device_train_batch_size=CFG["batch_size"],
    gradient_accumulation_steps=CFG["grad_accum"], learning_rate=CFG["lr"], warmup_steps=CFG["warmup_steps"],
    num_train_epochs=CFG["epochs"], fp16=True, eval_strategy="steps", eval_steps=CFG["eval_steps"],
    save_steps=CFG["save_steps"], save_total_limit=2, logging_steps=50, per_device_eval_batch_size=16,
    remove_unused_columns=False, label_names=["labels"], report_to=[], dataloader_num_workers=2,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
)
trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=DS(train_feats), eval_dataset=DS(dev_feats),
                         data_collator=Collator())
last = None
if (WORK / "ckpt").exists():
    cks = sorted((WORK / "ckpt").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    last = str(cks[-1]) if cks else None
trainer.train(resume_from_checkpoint=last)

# %% [markdown]
# ## Dev WER with the project's normalizer, then merge and save

# %%
from iphasr.metrics import corpus_wer

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

metrics = {"config": CFG}
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
merged.save_pretrained(WORK / "whisper-small-it-phone", safe_serialization=True)
processor.save_pretrained(WORK / "whisper-small-it-phone")
print("saved", list((WORK / "whisper-small-it-phone").iterdir()))
