"""Audio chain and dataset preparation (everything lands under IPASR_DATA).

G.711 chain (training audio): any input -> 8 kHz mono -> mu-law encode -> decode -> 16 kHz PCM.
test_collaudio: caller channel of Collaudio runs, silences trimmed, reference = the lines the
tester actually spoke (events.json, customer_speech_start).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .paths import data_root, manifest_path, write_jsonl

COLLAUDIO_RUNS = Path("D:/Projects/collaudio/runs")
CUSTOMER_CH = 1  # Twilio dual-channel recording: 0 = agent, 1 = caller (verified in Collaudio)


def g711_chain(src: Path, dst: Path) -> None:
    """src -> 8 kHz mu-law round trip -> 16 kHz 16-bit mono WAV at dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-ac", "1", "-ar", "8000", "-c:a", "pcm_mulaw", "-f", "mulaw", "-"],
        capture_output=True, check=True).stdout
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "mulaw", "-ar", "8000", "-ac", "1", "-i", "-",
         "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
        input=enc, check=True)


def speech_regions(x: np.ndarray, sr: int, frame_ms: int = 20, thr_db: float = 15.0,
                   pad_s: float = 0.25, min_gap_s: float = 0.6) -> list[tuple[int, int]]:
    """Energy VAD with an adaptive threshold (thr_db above the channel's noise floor)."""
    n = int(sr * frame_ms / 1000)
    frames = x[: len(x) // n * n].reshape(-1, n).astype(np.float64)
    db = 10 * np.log10((frames ** 2).mean(1) + 1e-10)
    floor = np.percentile(db, 10)
    active = db > floor + thr_db
    regions, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            regions.append([start, i]); start = None
    if start is not None:
        regions.append([start, len(active)])
    merged: list[list[int]] = []
    gap = int(min_gap_s * 1000 / frame_ms)
    for r in regions:
        if merged and r[0] - merged[-1][1] <= gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    pad = int(pad_s * sr)
    return [(max(0, a * n - pad), min(len(x), b * n + pad)) for a, b in merged if (b - a) * frame_ms >= 150]


def prepare_collaudio(runs: Path = COLLAUDIO_RUNS) -> list[dict]:
    import soundfile as sf

    out_dir = data_root() / "test_collaudio" / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in sorted(p for p in runs.iterdir() if p.is_dir()):
        ev_p, wav_p = run / "events.json", run / "call.wav"
        if not (ev_p.exists() and wav_p.exists()):
            continue
        events = json.loads(ev_p.read_text(encoding="utf-8"))
        said = [e["text"] for e in events if e.get("type") == "customer_speech_start" and e.get("text")]
        if not said:
            continue
        x, sr = sf.read(str(wav_p), dtype="int16")
        if x.ndim != 2:
            continue
        ch = x[:, CUSTOMER_CH]
        regions = speech_regions(ch, sr)
        if not regions:
            continue
        gap = np.zeros(int(0.3 * sr), dtype=np.int16)
        parts = []
        for a, b in regions:
            parts += [ch[a:b], gap]
        y = np.concatenate(parts[:-1])
        dst = out_dir / f"{run.name}.wav"
        sf.write(str(dst), y, sr, subtype="PCM_16")
        rows.append({"id": run.name, "audio": str(dst), "text": " ".join(said), "duration": len(y) / sr,
                     "sample_rate": sr, "variant": run.name.split("-")[2]})
    write_jsonl(manifest_path("test_collaudio"), rows)
    return rows
