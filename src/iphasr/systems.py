"""The six benchmarked systems. Transcripts are cached per (system, set) and resumable.

Cloud providers receive the original file bytes (Twilio 8 kHz WAV, Common Voice MP3...).
Local models get the same audio decoded by ffmpeg to 16 kHz mono float32.

Model choices (checked 2026-11-12):
- OpenAI: gpt-4o-transcribe shuts down 2027-02-26; its named successor gpt-transcribe is used.
- ElevenLabs: scribe_v3 is the newest model the API accepts (IPASR_SCRIBE_MODEL=scribe_v2 to compare).
- Parakeet-TDT-0.6B-v3 (CC-BY-4.0) runs through onnx-asr (ONNX export of the NVIDIA checkpoint).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .paths import REPO, data_root, manifest_path, read_jsonl, transcript_path

# USD per audio minute, for COSTS.md bookkeeping (local models: 0)
PRICE_PER_MIN = {
    "deepgram-nova-3": 0.0043,
    "elevenlabs-scribe": 0.22 / 60,
    "openai-gpt-transcribe": 0.0045,
}
ELEVENLABS_MODEL = os.environ.get("IPASR_SCRIBE_MODEL", "scribe_v3")
MODELS_DIR = Path("D:/cache/models")


def load_16k(path: str) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


# --- cloud ---------------------------------------------------------------------------

def _retry(fn, tries: int = 5):
    import httpx
    for k in range(tries):
        try:
            return fn()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 500, 502, 503, 504) and k < tries - 1:
                time.sleep(2 ** k)
                continue
            raise
        except (httpx.TransportError,):
            if k < tries - 1:
                time.sleep(2 ** k)
                continue
            raise


def _mime(path: str) -> str:
    return {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac"}.get(path.rsplit(".", 1)[-1], "audio/wav")


def deepgram(path: str) -> str:
    import httpx

    def go():
        r = httpx.post("https://api.deepgram.com/v1/listen",
                       params={"model": "nova-3", "language": "it", "punctuate": "true", "smart_format": "true",
                               "mip_opt_out": "true"},
                       headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}", "Content-Type": _mime(path)},
                       content=Path(path).read_bytes(), timeout=120)
        r.raise_for_status()
        return r.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
    return _retry(go)


def elevenlabs(path: str) -> str:
    import httpx

    def go():
        with open(path, "rb") as f:
            r = httpx.post("https://api.elevenlabs.io/v1/speech-to-text",
                           headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
                           data={"model_id": ELEVENLABS_MODEL, "language_code": "it", "tag_audio_events": "false"},
                           files={"file": (Path(path).name, f, _mime(path))}, timeout=180)
        r.raise_for_status()
        return r.json()["text"]
    return _retry(go)


def openai(path: str) -> str:
    import httpx

    def go():
        with open(path, "rb") as f:
            r = httpx.post("https://api.openai.com/v1/audio/transcriptions",
                           headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                           data={"model": "gpt-transcribe", "languages[]": "it"},
                           files={"file": (Path(path).name, f, _mime(path))}, timeout=180)
        r.raise_for_status()
        return r.json()["text"]
    return _retry(go)


CLOUD = {"deepgram-nova-3": deepgram, "elevenlabs-scribe": elevenlabs, "openai-gpt-transcribe": openai}


# --- local ---------------------------------------------------------------------------

class Whisper:
    def __init__(self, model: str):
        import torch
        from transformers import pipeline
        torch.set_num_threads(max(1, os.cpu_count() - 1))
        self.pipe = pipeline("automatic-speech-recognition", model=model, device="cpu", torch_dtype=torch.float32)

    def batch(self, paths: list[str]) -> list[str]:
        audio = [{"raw": load_16k(p), "sampling_rate": 16000} for p in paths]
        out = self.pipe(audio, batch_size=len(audio), chunk_length_s=30,
                        generate_kwargs={"language": "italian", "task": "transcribe"})
        return [o["text"].strip() for o in out]


class Parakeet:
    def __init__(self):
        import onnx_asr
        self.m = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", str(MODELS_DIR / "parakeet-tdt-0.6b-v3"))

    def batch(self, paths: list[str]) -> list[str]:
        return [self.m.recognize(load_16k(p), sample_rate=16000) for p in paths]


def local_model(system: str, model_path: str | None = None):
    if system == "whisper-small":
        return Whisper("openai/whisper-small")
    if system == "whisper-small-it-phone":
        return Whisper(model_path or os.environ.get("IPASR_FT_MODEL", str(MODELS_DIR / "whisper-small-it-phone")))
    if system == "parakeet-tdt-0.6b-v3":
        return Parakeet()
    raise KeyError(system)


# --- runner --------------------------------------------------------------------------

def run(system: str, set_name: str, limit: int | None = None, workers: int = 4, model_path: str | None = None) -> dict:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
    manifest = read_jsonl(manifest_path(set_name))[:limit]
    if not manifest:
        raise SystemExit(f"{set_name}: manifest empty ({manifest_path(set_name)})")
    out = transcript_path(system, set_name)
    done = {r["id"]: r for r in read_jsonl(out)}
    todo = [r for r in manifest if r["id"] not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        def save(row, hyp):
            f.write(json.dumps({"id": row["id"], "hyp": hyp}, ensure_ascii=False) + "\n")
            f.flush()
        if system in CLOUD:
            fn = CLOUD[system]
            with ThreadPoolExecutor(workers) as ex:
                for row, hyp in zip(todo, ex.map(lambda r: fn(r["audio"]), todo)):
                    save(row, hyp)
        else:
            m = local_model(system, model_path)
            bs = 8
            for k in range(0, len(todo), bs):
                chunk = todo[k:k + bs]
                for row, hyp in zip(chunk, m.batch([r["audio"] for r in chunk])):
                    save(row, hyp)
                print(f"  {system} {set_name}: {k + len(chunk)}/{len(todo)}", flush=True)
    minutes = sum(r.get("duration", 0.0) for r in todo) / 60
    cost = minutes * PRICE_PER_MIN.get(system, 0.0)
    usage = {"at": time.strftime("%Y-%m-%d %H:%M"), "system": system, "set": set_name, "utts": len(todo),
             "minutes": round(minutes, 2), "est_usd": round(cost, 4), "seconds": round(time.time() - t0, 1)}
    with (data_root() / "usage.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(usage) + "\n")
    return usage
