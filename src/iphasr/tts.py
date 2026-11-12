"""Synthesize train_synth with open voices, then store as 8 kHz G.711 mu-law WAV.

Voices and licenses (checked 2026-11-12):
- Kokoro-82M if_sara, im_nicola            Apache-2.0
- Piper it_IT-paola-medium                 dataset CC0-1.0
- Piper it_IT-serena-medium                dataset CC-BY-4.0
- Piper it_IT-riccardo-x_low               M-AILABS (permissive)

Stored audio is the codec output itself (mu-law in a WAV container); the training
loader decodes it and resamples to 16 kHz, so the chain is exact and 4x smaller.
"""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import numpy as np

from .paths import data_root, manifest_path, read_jsonl, write_jsonl

PIPER_DIR = Path("D:/cache/tts")
PIPER_VOICES = ["it_IT-paola-medium", "it_IT-serena-medium", "it_IT-riccardo-x_low"]
KOKORO_VOICES = ["if_sara", "im_nicola"]
VOICES = [f"piper:{v}" for v in PIPER_VOICES] + [f"kokoro:{v}" for v in KOKORO_VOICES]


def to_ulaw_wav(audio: np.ndarray, sr: int, dst: Path, gain_db: float = 0.0) -> float:
    """float mono -> 8 kHz mu-law WAV via ffmpeg. Returns duration (s)."""
    x = np.clip(audio * (10 ** (gain_db / 20)), -1, 1)
    pcm = (x * 32767).astype("<i2").tobytes()
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "-",
                    "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", str(dst)], input=pcm, check=True)
    return len(x) / sr


class Synth:
    def __init__(self):
        self._piper: dict = {}
        self._kokoro = None

    def __call__(self, voice: str, text: str, rng: random.Random) -> tuple[np.ndarray, int]:
        kind, name = voice.split(":")
        if kind == "piper":
            from piper import PiperVoice, SynthesisConfig
            if name not in self._piper:
                self._piper[name] = PiperVoice.load(str(PIPER_DIR / f"{name}.onnx"))
            v = self._piper[name]
            cfg = SynthesisConfig(length_scale=rng.uniform(0.9, 1.15), noise_scale=rng.uniform(0.5, 0.8))
            chunks = [c.audio_float_array for c in v.synthesize(text, syn_config=cfg)]
            return np.concatenate(chunks), v.config.sample_rate
        from kokoro import KPipeline
        if self._kokoro is None:
            self._kokoro = KPipeline(lang_code="i", repo_id="hexgrad/Kokoro-82M")
        parts = [r.audio.numpy() for r in self._kokoro(text, voice=name, speed=rng.uniform(0.9, 1.1))]
        return np.concatenate(parts), 24000


def synthesize(limit: int | None = None, seed: int = 0) -> list[dict]:
    """Render data_root()/synth/sentences.jsonl round-robin over voices; resumable."""
    sentences = read_jsonl(data_root() / "synth" / "sentences.jsonl")[:limit]
    out_dir = data_root() / "train_synth" / "audio"
    done = {r["id"]: r for r in read_jsonl(manifest_path("train_synth"))}
    rng = random.Random(seed)
    synth = Synth()
    rows = []
    for k, s in enumerate(sentences):
        voice = VOICES[k % len(VOICES)]
        gain = rng.uniform(-6, 0)
        if s["id"] in done and Path(done[s["id"]]["audio"]).exists():
            rows.append(done[s["id"]])
            continue
        audio, sr = synth(voice, s["tts_text"], rng)
        dst = out_dir / f"{s['id']}.wav"
        dur = to_ulaw_wav(audio, sr, dst, gain)
        rows.append({"id": s["id"], "audio": str(dst), "text": s["text"], "entities": s["entities"],
                     "voice": voice, "duration": dur})
        if len(rows) % 100 == 0:
            write_jsonl(manifest_path("train_synth"), rows)
            print(f"{len(rows)}/{len(sentences)}  {sum(r['duration'] for r in rows) / 3600:.2f} h", flush=True)
    write_jsonl(manifest_path("train_synth"), rows)
    return rows
