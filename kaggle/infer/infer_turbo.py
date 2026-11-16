# %% [markdown]
# # Inference only: Whisper-large-v3-turbo base vs round-4 LoRA on test_cv and test_cv_g711
#
# Kaggle script kernel (GPU T4). The same 661 Common Voice clips the local bench uses, transcribed on
# GPU instead of the local CPU (about 3 h per model per set there, minutes here).
# Inputs:
# - Kaggle secret `MDC_API_KEY`: Common Voice cannot be re-hosted, so the kernel downloads it itself.
# - Kernel source `whisper-small-it-phone-lora` (round 4 output): the merged `whisper-turbo-it-phone/`.
# - `kaggle/infer/test_cv_clips.json` from the public repo: the 661 test clip file names.
# Decoding, identical for both models: Whisper's standard guards against repetition loops
# (temperature fallback, compression_ratio_threshold 1.35 on zlib-based ratio, logprob_threshold -1.0,
# no_speech_threshold 0.6). Output: /kaggle/working/transcripts/<system>/<set>.jsonl, same ids as the
# local bench, plus timing.json.

# %%
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import requests
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

print("transformers", __import__("transformers").__version__, "torch", torch.__version__)
WORK = Path("/kaggle/working")
TMP = Path("/kaggle/tmp")
CV_DIR = TMP / "cv_test"
CV_DIR.mkdir(parents=True, exist_ok=True)
REPO_RAW = "https://raw.githubusercontent.com/albedimaria/italian-phone-asr/main"
CLIPS = requests.get(f"{REPO_RAW}/kaggle/infer/test_cv_clips.json", timeout=60).json()
assert len(CLIPS) == 661, len(CLIPS)

# %% [markdown]
# ## Common Voice IT 27.0: one streaming pass, keep only the 661 test clips

# %%
CV_ID = "cmu5xagm300dyo1075qdhf20m"  # Common Voice Scripted Speech 27.0 - Italian (CC0-1.0)
wanted = set(CLIPS)
if not (CV_DIR / "done").exists():
    from kaggle_secrets import UserSecretsClient
    key = UserSecretsClient().get_secret("MDC_API_KEY")
    r = requests.post(f"https://mozilladatacollective.com/api/datasets/{CV_ID}/download",
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, timeout=60)
    r.raise_for_status()
    with requests.get(r.json()["downloadUrl"], stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
            for m in tar:
                name = Path(m.name).name
                if m.isfile() and name in wanted:
                    (CV_DIR / name).write_bytes(tar.extractfile(m).read())
    (CV_DIR / "done").write_text("ok")
missing = [c for c in CLIPS if not (CV_DIR / c).exists()]
assert not missing, f"{len(missing)} clips missing, e.g. {missing[:3]}"

# %% [markdown]
# ## Audio: the same ffmpeg chains as src/iphasr (load_16k and g711_chain)

# %%
def load_16k(path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def g711_16k(path) -> np.ndarray:
    """8 kHz mono -> mu-law -> decode -> 16 kHz 16-bit, as the local test_cv_g711 WAVs."""
    enc = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "8000", "-c:a", "pcm_mulaw",
                          "-f", "mulaw", "-"], capture_output=True, check=True).stdout
    raw = subprocess.run(["ffmpeg", "-v", "error", "-f", "mulaw", "-ar", "8000", "-ac", "1", "-i", "-",
                          "-ar", "16000", "-f", "s16le", "-"], input=enc, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


SETS = {
    "test_cv": [(f"cv_{Path(c).stem}", load_16k(CV_DIR / c)) for c in CLIPS],
    "test_cv_g711": [(f"cv_{Path(c).stem}_g711", g711_16k(CV_DIR / c)) for c in CLIPS],
}
print({k: len(v) for k, v in SETS.items()})

# %% [markdown]
# ## Transcribe: both models, same decoding

# %%
GEN = dict(language="italian", task="transcribe", max_new_tokens=200,
           temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=1.35,
           logprob_threshold=-1.0, no_speech_threshold=0.6, condition_on_prev_tokens=False)
ROUND4 = next(p.parent for p in Path("/kaggle/input").rglob("whisper-turbo-it-phone/config.json"))
MODELS = {"whisper-large-v3-turbo": "openai/whisper-large-v3-turbo", "whisper-turbo-it-phone": str(ROUND4)}
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
timing = {}

for system, src in MODELS.items():
    model = WhisperForConditionalGeneration.from_pretrained(src, torch_dtype=torch.float16).cuda().eval()
    for set_name, items in SETS.items():
        out_dir = WORK / "transcripts" / system
        out_dir.mkdir(parents=True, exist_ok=True)
        t0, rows, bs = time.time(), [], 16
        for k in range(0, len(items), bs):
            chunk = items[k:k + bs]
            feats = processor([a for _, a in chunk], sampling_rate=16000, return_tensors="pt",
                              return_attention_mask=True)
            with torch.no_grad():
                ids = model.generate(input_features=feats.input_features.half().cuda(),
                                     attention_mask=feats.attention_mask.cuda(), **GEN)
            hyps = processor.batch_decode(ids, skip_special_tokens=True)
            rows += [{"id": i, "hyp": h.strip()} for (i, _), h in zip(chunk, hyps)]
        timing[f"{system}/{set_name}"] = round(time.time() - t0, 1)
        (out_dir / f"{set_name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                                   encoding="utf-8")
        print(system, set_name, len(rows), timing[f"{system}/{set_name}"], "s", flush=True)
    del model
    torch.cuda.empty_cache()

(WORK / "timing.json").write_text(json.dumps({"gen": {k: str(v) for k, v in GEN.items()}, "seconds": timing,
                                               "gpu": torch.cuda.get_device_name(0)}, indent=1))
print("done")
