# %% [markdown]
# # test_synth_entities: held-out entity sentences read by a voice that is not in training
#
# train_synth uses every open Italian voice we had (3 Piper, 2 Kokoro). This set is read by
# Chatterbox multilingual (resemble-ai/chatterbox, MIT), default voice, on a Kaggle GPU.
# Sentences: iphasr.synth.generate_heldout() from the public repo (held-out templates, cars, names).
# The TTS reads `tts_text` (numbers spelled out); the reference is the written `text`.
# Output: /kaggle/working/wav/<id>.wav (model rate, 16-bit) + sentences.jsonl. The G.711 chain is
# applied locally, identical to test_cv_g711.

# %%
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "chatterbox-tts", "num2words>=0.5.14", "jiwer>=3.1",
                "git+https://github.com/albedimaria/italian-phone-asr.git"], check=True)

# %%
import json
import time
from pathlib import Path

import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

from iphasr.synth import generate_heldout

WORK = Path("/kaggle/working")
OUT = WORK / "wav"
OUT.mkdir(parents=True, exist_ok=True)
items = generate_heldout()
torch.manual_seed(0)
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model="v3")
print("sr", model.sr, "items", len(items), flush=True)
t0 = time.time()
for k, it in enumerate(items):
    wav = model.generate(it["tts_text"], language_id="it")
    ta.save(str(OUT / f"{it['id']}.wav"), wav.cpu(), model.sr, bits_per_sample=16)
    it["sr"] = model.sr
    if k % 20 == 0:
        print(k, round(time.time() - t0), "s", flush=True)
(WORK / "sentences.jsonl").write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n",
                                       encoding="utf-8")
print("done", round(time.time() - t0), "s")
