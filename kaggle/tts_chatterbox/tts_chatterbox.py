# %% [markdown]
# # test_synth_entities: held-out entity sentences read by a voice that is not in training
#
# train_synth uses every open Italian voice we had (3 Piper, 2 Kokoro). This set is read by
# Chatterbox multilingual (resemble-ai/chatterbox, MIT), default voice, on a Kaggle GPU.
# Sentences: iphasr.synth.generate_heldout() from the public repo (held-out templates, cars, names).
# The TTS reads `tts_text` (numbers spelled out); the reference is the written `text`.
# Chatterbox pins torch 2.6, which breaks the image's torchvision: it runs in its own uv venv.
# Output: /kaggle/working/wav/<id>.wav (model rate, 16-bit) + sentences.jsonl. The G.711 chain is
# applied locally, identical to test_cv_g711.

# %%
import subprocess
import sys
from pathlib import Path

VENV = Path("/kaggle/tmp/cb")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
subprocess.run(["uv", "venv", str(VENV), "--python", "3.11"], check=True)
py = str(VENV / "bin" / "python")
# editable clone, not a wheel: the generator reads data/script/copione.txt (test values it must avoid)
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/albedimaria/italian-phone-asr.git",
                "/kaggle/tmp/repo"], check=True)
subprocess.run(["uv", "pip", "install", "--python", py, "-q", "chatterbox-tts", "num2words>=0.5.14", "jiwer>=3.1",
                "-e", "/kaggle/tmp/repo"], check=True)

WORKER = r'''
import json, time
from pathlib import Path
import torch, torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from iphasr.synth import generate_heldout

WORK = Path("/kaggle/working"); OUT = WORK / "wav"; OUT.mkdir(parents=True, exist_ok=True)
items = generate_heldout()
torch.manual_seed(0)
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")  # PyPI release: no t3_model arg
import importlib.metadata as md; print("chatterbox", md.version("chatterbox-tts"), "torch", torch.__version__, "sr", model.sr, "items", len(items), flush=True)
t0 = time.time()
for k, it in enumerate(items):
    wav = model.generate(it["tts_text"], language_id="it")
    ta.save(str(OUT / f"{it['id']}.wav"), wav.cpu(), model.sr, bits_per_sample=16)
    it["sr"] = model.sr
    if k % 20 == 0:
        print(k, round(time.time() - t0), "s", flush=True)
(WORK / "sentences.jsonl").write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n",
                                       encoding="utf-8")
print("done", round(time.time() - t0), "s", flush=True)
'''
Path("/kaggle/tmp/worker.py").write_text(WORKER)
subprocess.run([py, "/kaggle/tmp/worker.py"], check=True)
