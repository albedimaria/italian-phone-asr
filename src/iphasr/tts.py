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


def roundtrip_filter(max_wer: float = 0.10) -> dict:
    """Transcribe every synth clip with Parakeet; keep clips whose entities all survive and WER <= max_wer.

    TTS voices garble some inputs (long compound numbers, names): a wrong label teaches the model
    the wrong thing, so those clips are dropped. Writes manifests/train_synth_clean.jsonl.
    """
    from .metrics import entity_accuracy, utt_counts
    from .systems import Parakeet

    rows = read_jsonl(manifest_path("train_synth"))
    cache = data_root() / "synth" / "roundtrip.jsonl"
    done = {r["id"]: r["hyp"] for r in read_jsonl(cache)}
    p = Parakeet()
    with cache.open("a", encoding="utf-8") as f:
        import json
        for r in rows:
            if r["id"] not in done:
                done[r["id"]] = p.batch([r["audio"]])[0]
                f.write(json.dumps({"id": r["id"], "hyp": done[r["id"]]}, ensure_ascii=False) + "\n")
    keep, stats = [], {}
    for r in rows:
        c = utt_counts(r["text"], done[r["id"]])
        acc = entity_accuracy([(r["entities"], done[r["id"]])]).get("all", (0, 0))
        ok = c.ref_len > 0 and c.errors / c.ref_len <= max_wer and acc[0] == acc[1]
        s = stats.setdefault(r["voice"], [0, 0])
        s[0] += ok; s[1] += 1
        if ok:
            keep.append({**r, "roundtrip_wer": round(c.errors / c.ref_len, 3)})
    write_jsonl(manifest_path("train_synth_clean"), keep)
    return {"kept": len(keep), "of": len(rows), "hours": round(sum(r["duration"] for r in keep) / 3600, 2),
            "by_voice": {v: f"{a}/{b}" for v, (a, b) in stats.items()}}
