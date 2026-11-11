"""Pre-push sweep: no audio, no test_human data, no secrets in tracked files.

Checks every file `git ls-files` would publish (tracked + staged):
- audio extensions and anything under a test_human / sessions path
- known secret shapes (OpenAI, HF, ElevenLabs, AWS, private keys, Twilio API keys)
- the literal values found in local .env files (the strongest check: a real key
  can have any shape)
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from .paths import REPO

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".ulaw", ".m4a", ".webm", ".aac", ".amr"}
FORBIDDEN_PATH = re.compile(r"(^|/)(test_human|sessions|consents?)(/|$)", re.I)
SECRET_PATTERNS = {
    "openai": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}"),
    "huggingface": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "aws": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "twilio api key": re.compile(r"\bSK[0-9a-f]{32}\b"),
    "elevenlabs": re.compile(r"\bsk_[0-9a-f]{40,}\b"),
}
ENV_FILES = [REPO / ".env", Path("D:/Projects/collaudio/.env")]


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [l for l in out.splitlines() if l.strip()]


def env_secret_values() -> set[str]:
    vals = set()
    for f in ENV_FILES:
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                if len(v) >= 16:
                    vals.add(v)
    for k, v in os.environ.items():
        if any(t in k for t in ("KEY", "TOKEN", "SECRET")) and len(v) >= 16:
            vals.add(v)
    return vals


def sweep() -> list[str]:
    problems = []
    secrets = env_secret_values()
    for rel in tracked_files():
        p = REPO / rel
        if p.suffix.lower() in AUDIO_EXT:
            problems.append(f"audio file would be published: {rel}")
        if FORBIDDEN_PATH.search(rel):
            problems.append(f"private path would be published: {rel}")
        if not p.is_file() or p.stat().st_size > 5_000_000:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for name, rx in SECRET_PATTERNS.items():
            if rx.search(text):
                problems.append(f"{name}-like secret in {rel}")
        for v in secrets:
            if v in text:
                problems.append(f"value from a local .env appears in {rel}")
                break
    return problems


def main() -> int:
    files = tracked_files()
    problems = sweep()
    if problems:
        print("PREFLIGHT FAILED:")
        for p in problems:
            print(" -", p)
        return 1
    print(f"preflight ok: {len(files)} files, no audio, no private paths, no secrets")
    return 0
