"""Audio chain and dataset preparation (everything lands under IPASR_DATA).

G.711 chain (training audio): any input -> 8 kHz mono -> mu-law encode -> decode -> 16 kHz PCM.
test_collaudio: caller channel of Collaudio runs, silences trimmed, reference = the lines the
tester actually spoke (events.json, customer_speech_start). The Twilio trial preamble at the start
of the caller channel is cut.
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
        # The caller channel starts with the Twilio trial preamble ("Thank you for using your demo
        # account..."), before the tester's media stream opens. Map stream time to recording time
        # (both end at hangup) and drop every region that ends before the first scripted line.
        stop_ms = next((e["t"] for e in events if e.get("type") == "stream_stop"), None)
        first_ms = next(e["t"] for e in events if e.get("type") == "customer_speech_start")
        if stop_ms is None:
            continue
        offset_s = len(ch) / sr - stop_ms / 1000
        first_line_at = int((offset_s + first_ms / 1000) * sr)
        regions = [(a, b) for a, b in speech_regions(ch, sr) if b > first_line_at]
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


# --- Common Voice IT 27.0 ------------------------------------------------------------------

CV_TAR = Path("D:/data/italian-phone-asr/raw/cv-it-27/common-voice-scripted-speech-27-0-italia-5ca1c1b9.tar.gz")
TRAIN_HOURS, DEV_CLIPS, TEST_HOURS, MAX_PER_SPEAKER = 15.0, 600, 1.0, 25
AVG_CLIP_S = 5.444  # from the MDC datasheet; real durations are measured on the clips we keep


def _read_tsv(data: bytes) -> list[dict]:
    import csv
    import io
    return list(csv.DictReader(io.StringIO(data.decode("utf-8")), delimiter="	", quoting=csv.QUOTE_NONE))


def _pick(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Up to n clips, at most MAX_PER_SPEAKER per speaker, spread over as many speakers as possible."""
    import random
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    per: dict[str, int] = {}
    out = []
    for r in rows:
        if per.get(r["client_id"], 0) >= MAX_PER_SPEAKER:
            continue
        per[r["client_id"]] = per.get(r["client_id"], 0) + 1
        out.append(r)
        if len(out) >= n:
            break
    return out


def prepare_cv(tar_path: Path = CV_TAR) -> dict:
    """One streaming pass over the tar: read the split TSVs, extract test_cv clips (kept local),
    write cv_subset.json (train/dev clip lists for the Kaggle kernel) and the test_cv manifest."""
    import tarfile

    import soundfile as sf

    tsv: dict[str, list[dict]] = {}
    with tarfile.open(tar_path, "r|gz") as tar:
        for m in tar:
            name = Path(m.name).name
            if m.isfile() and name in ("train.tsv", "dev.tsv", "test.tsv"):
                tsv[name[:-4]] = _read_tsv(tar.extractfile(m).read())
            if len(tsv) == 3:
                break
    train = _pick(tsv["train"], int(TRAIN_HOURS * 3600 / AVG_CLIP_S), seed=0)
    dev = _pick(tsv["dev"], DEV_CLIPS, seed=1)
    test = _pick(tsv["test"], int(TEST_HOURS * 3600 / AVG_CLIP_S), seed=2)
    out_dir = data_root() / "test_cv" / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    want = {r["path"] for r in test}
    with tarfile.open(tar_path, "r|gz") as tar:
        for m in tar:
            name = Path(m.name).name
            if m.isfile() and name in want and not (out_dir / name).exists():
                (out_dir / name).write_bytes(tar.extractfile(m).read())
    rows = []
    for r in test:
        f = out_dir / r["path"]
        if f.exists():
            rows.append({"id": f"cv_{Path(r['path']).stem}", "audio": str(f), "text": r["sentence"],
                         "duration": sf.info(str(f)).duration, "speaker": r["client_id"][:12]})
    write_jsonl(manifest_path("test_cv"), rows)
    subset = {"train": [r["path"] for r in train], "dev": [r["path"] for r in dev],
              "meta": [{"path": r["path"], "sentence": r["sentence"]} for r in train + dev]}
    dst = data_root() / "kaggle" / "cv_subset.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")
    return {"train": len(train), "dev": len(dev), "test_cv": len(rows),
            "test_cv_min": round(sum(r["duration"] for r in rows) / 60, 1),
            "train_speakers": len({r["client_id"] for r in train})}
