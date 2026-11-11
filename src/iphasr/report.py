"""Build results/table.md from cached transcripts. No network.

Gate (exit 1, "an empty gate is not green"):
- test_human has < 5 speakers or < 40 minutes
- any of the 6 systems lacks a complete transcript for any of the 3 test sets
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .metrics import entity_accuracy, paired_bootstrap, utt_counts
from .paths import BASE, OURS, RESULTS, SETS, SYSTEMS, manifest_path, read_jsonl, transcript_path

MIN_SPEAKERS = 5
MIN_MINUTES = 40.0
TARGET_REL = 0.15
ENTITY_TYPES = ["prezzo", "data", "orario", "auto", "nome", "telefono", "numero"]


@dataclass
class Scored:
    ids: list[str]
    errors: np.ndarray
    ref_lens: np.ndarray

    @property
    def wer(self) -> float:
        return float(self.errors.sum() / self.ref_lens.sum())


def _score(manifest: list[dict], hyps: dict[str, str], keep=lambda r: True) -> Scored:
    rows = [r for r in manifest if keep(r)]
    cs = [utt_counts(r["text"], hyps[r["id"]]) for r in rows]
    return Scored([r["id"] for r in rows], np.array([c.errors for c in cs]), np.array([c.ref_len for c in cs]))


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def build(n_resamples: int = 10_000) -> tuple[str, list[str]]:
    """Return (markdown, gate_failures)."""
    fails: list[str] = []
    manifests = {s: read_jsonl(manifest_path(s)) for s in SETS}
    hyps: dict[tuple[str, str], dict[str, str]] = {}

    human = manifests["test_human"]
    speakers = {r["speaker"] for r in human}
    minutes = sum(r.get("duration", 0.0) for r in human) / 60
    if len(speakers) < MIN_SPEAKERS:
        fails.append(f"test_human has {len(speakers)} speakers (< {MIN_SPEAKERS})")
    if minutes < MIN_MINUTES:
        fails.append(f"test_human has {minutes:.1f} min (< {MIN_MINUTES:.0f})")
    for s, m in manifests.items():
        if not m:
            fails.append(f"{s}: manifest missing or empty ({manifest_path(s)})")
    for sysname in SYSTEMS:
        for s in SETS:
            rows = {r["id"]: r["hyp"] for r in read_jsonl(transcript_path(sysname, s))}
            missing = [r["id"] for r in manifests[s] if r["id"] not in rows]
            if not rows or missing:
                fails.append(f"{sysname} on {s}: {len(missing) if rows else 'all'} transcripts missing")
            else:
                hyps[(sysname, s)] = rows

    md = ["# Results", "",
          "WER (%) after the shared Italian normalizer (`src/iphasr/normalize.py`). Lower is better.", ""]
    sizes = {s: f"{len(manifests[s])} utts" for s in SETS}
    sizes["test_human"] += f", {len(speakers)} speakers, {minutes:.0f} min"
    md += ["| System | " + " | ".join(f"{s} ({sizes[s]})" for s in SETS) + " |",
           "|---|" + "---:|" * len(SETS)]
    scored: dict[tuple[str, str], Scored] = {}
    for sysname, label in SYSTEMS.items():
        cells = []
        for s in SETS:
            if (sysname, s) in hyps:
                scored[(sysname, s)] = _score(manifests[s], hyps[(sysname, s)])
                cells.append(_pct(scored[(sysname, s)].wer))
            else:
                cells.append("—")
        md.append(f"| {label} | " + " | ".join(cells) + " |")

    # entity accuracy on test_human
    md += ["", "## Entity accuracy on test_human (%)", "",
           "An entity is correct when its normalized form appears contiguously in the normalized hypothesis.", "",
           "| System | " + " | ".join(ENTITY_TYPES) + " | all |", "|---|" + "---:|" * (len(ENTITY_TYPES) + 1)]
    for sysname, label in SYSTEMS.items():
        if (sysname, "test_human") not in hyps:
            md.append(f"| {label} | " + " | ".join("—" for _ in range(len(ENTITY_TYPES) + 1)) + " |")
            continue
        h = hyps[(sysname, "test_human")]
        acc = entity_accuracy([(r.get("entities", []), h[r["id"]]) for r in human])
        md.append(f"| {label} | " + " | ".join(
            f"{_pct(acc[t][0] / acc[t][1])}" if t in acc else "—" for t in ENTITY_TYPES + ["all"]) + " |")

    # fine-tuned vs base: paired bootstrap
    md += ["", "## Fine-tuned vs base on test_human", ""]
    if (BASE, "test_human") in scored and (OURS, "test_human") in scored:
        a, b = scored[(BASE, "test_human")], scored[(OURS, "test_human")]
        r = paired_bootstrap(a.errors, b.errors, a.ref_lens, n_resamples=n_resamples)
        passed = r.rel_reduction >= TARGET_REL and r.ci_low > 0
        md += [f"- WER base {_pct(r.wer_a)}% → fine-tuned {_pct(r.wer_b)}%",
               f"- relative reduction **{_pct(r.rel_reduction)}%**, 95% CI [{_pct(r.ci_low)}%, {_pct(r.ci_high)}%] "
               f"(paired bootstrap over {r.n_utts} utterances, {r.n_resamples} resamples)",
               f"- target ≥ {_pct(TARGET_REL)}% with CI lower bound > 0: **{'MET' if passed else 'NOT MET'}**"]
    else:
        md.append("- not available (missing transcripts)")

    # slices
    md += ["", "## Slices of test_human (WER %)", ""]
    slices = defaultdict(list)
    for r in human:
        slices[f"speaker {r['speaker']}"].append(r["id"])
        slices[f"channel: {r.get('channel', 'browser')}"].append(r["id"])
        if r.get("r_moscia"):
            slices["uvular /r/ speaker (n=1)"].append(r["id"])
    if human and slices:
        names = sorted(slices)
        md += ["| System | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
        for sysname, label in SYSTEMS.items():
            if (sysname, "test_human") not in hyps:
                continue
            h = hyps[(sysname, "test_human")]
            cells = []
            for name in names:
                ids = set(slices[name])
                cells.append(_pct(_score(human, h, keep=lambda r, ids=ids: r["id"] in ids).wer))
            md.append(f"| {label} | " + " | ".join(cells) + " |")
    else:
        md.append("- not available")

    if fails:
        md += ["", "## Gate: FAILED", ""] + [f"- {f}" for f in fails]
    return "\n".join(md) + "\n", fails


def main(n_resamples: int = 10_000) -> int:
    md, fails = build(n_resamples)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "table.md").write_text(md, encoding="utf-8")
    print(md)
    if fails:
        print(f"GATE FAILED ({len(fails)} problems): results/table.md is incomplete.")
        return 1
    return 0
