"""Load the annotated test script (data/script/copione.txt)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO / "data" / "script" / "copione.txt"
ENTITY_TYPES = {"prezzo", "data", "orario", "auto", "nome", "telefono", "numero"}
_MARK = re.compile(r"\[([^\[\]|]+)\|([a-z]+)\]")


@dataclass
class Line:
    id: str
    text: str
    entities: list[dict] = field(default_factory=list)


def parse_line(raw: str) -> tuple[str, list[dict]]:
    ents = []
    for m in _MARK.finditer(raw):
        if m.group(2) not in ENTITY_TYPES:
            raise ValueError(f"unknown entity type {m.group(2)!r} in: {raw}")
        ents.append({"type": m.group(2), "text": m.group(1)})
    return _MARK.sub(lambda m: m.group(1), raw), ents


def load_script(path: Path = SCRIPT_PATH) -> list[Line]:
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        text, ents = parse_line(raw)
        out.append(Line(id=f"s{len(out) + 1:03d}", text=text, entities=ents))
    return out
