"""Where data lives. Audio and transcripts stay outside the repo (IPASR_DATA)."""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

SETS = ["test_human", "test_cv", "test_collaudio"]
SYSTEMS = {
    "deepgram-nova-3": "Deepgram Nova-3",
    "elevenlabs-scribe": "ElevenLabs Scribe v3",
    "openai-gpt-transcribe": "OpenAI gpt-transcribe",
    "parakeet-tdt-0.6b-v3": "NVIDIA Parakeet-TDT-0.6B-v3",
    "whisper-small": "Whisper-small (base)",
    "whisper-small-it-phone": "Whisper-small + LoRA (ours)",
}
BASE, OURS = "whisper-small", "whisper-small-it-phone"


def data_root() -> Path:
    return Path(os.environ.get("IPASR_DATA", "D:/data/italian-phone-asr"))


def manifest_path(set_name: str) -> Path:
    return data_root() / "manifests" / f"{set_name}.jsonl"


def transcript_path(system: str, set_name: str) -> Path:
    return data_root() / "transcripts" / system / f"{set_name}.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
