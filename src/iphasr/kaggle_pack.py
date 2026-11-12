"""Build the private Kaggle dataset `ipasr-train` (our own data only; Common Voice is fetched in-kernel)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .paths import data_root, manifest_path, read_jsonl, write_jsonl

DATASET_SLUG = "ipasr-train"


def pack(username: str, config: dict | None = None) -> Path:
    out = data_root() / "kaggle" / DATASET_SLUG
    audio = out / "train_synth"
    audio.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(manifest_path("train_synth_clean"))
    if not rows:
        raise SystemExit("train_synth_clean is empty: run the round-trip filter first")
    for r in rows:
        dst = audio / Path(r["audio"]).name
        if not dst.exists():
            shutil.copy2(r["audio"], dst)
    write_jsonl(out / "train_synth.jsonl", [{**r, "audio": Path(r["audio"]).name} for r in rows])
    shutil.copy2(data_root() / "kaggle" / "cv_subset.json", out / "cv_subset.json")
    (out / "config.json").write_text(json.dumps(config or {}, indent=1), encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "ipasr-train", "id": f"{username}/{DATASET_SLUG}", "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
    }, indent=1), encoding="utf-8")
    return out
