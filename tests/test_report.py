"""The report gate: empty or partial data must FAIL, complete data passes."""

import random

import pytest

from iphasr import report
from iphasr.paths import SETS, SYSTEMS, manifest_path, transcript_path, write_jsonl
from iphasr.script import load_script


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("IPASR_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(report, "RESULTS", tmp_path / "results")
    return tmp_path


def _fill(n_speakers=5, secs=30.0, systems=SYSTEMS):
    rng = random.Random(0)
    script = load_script()
    human = [{"id": f"spk{s}_{l.id}", "speaker": f"spk{s}", "text": l.text, "entities": l.entities,
              "duration": secs, "channel": "browser", "r_moscia": s == 0}
             for s in range(n_speakers) for l in script[:20]]
    write_jsonl(manifest_path("test_human"), human)
    for s in ("test_cv", "test_collaudio"):
        write_jsonl(manifest_path(s), [{"id": f"{s}{i}", "text": "buongiorno vorrei un appuntamento"} for i in range(10)])
    for sysname in systems:
        p_err = 0.05 if sysname == "whisper-small-it-phone" else 0.3
        for s in SETS:
            rows = []
            for r in (human if s == "test_human" else [{"id": f"{s}{i}", "text": "buongiorno vorrei un appuntamento"} for i in range(10)]):
                words = r["text"].split()
                hyp = " ".join(w if rng.random() > p_err else "xx" for w in words)
                rows.append({"id": r["id"], "hyp": hyp})
            write_jsonl(transcript_path(sysname, s), rows)


def test_empty_gate_fails(root):
    md, fails = report.build(n_resamples=200)
    assert fails and any("speakers" in f for f in fails) and any("minutes" in f or "min" in f for f in fails)
    assert report.main(200) == 1
    assert (root / "results" / "table.md").exists()


def test_too_few_speakers_or_minutes_fails(root):
    _fill(n_speakers=4, secs=30.0)       # 4 speakers x 20 x 30 s = 40 min, but 4 speakers
    _, fails = report.build(n_resamples=200)
    assert any("speakers" in f for f in fails)


def test_missing_system_fails(root):
    _fill(systems=[s for s in SYSTEMS if s != "parakeet-tdt-0.6b-v3"])
    _, fails = report.build(n_resamples=200)
    assert any("parakeet" in f for f in fails)


def test_complete_data_passes(root):
    _fill(n_speakers=5, secs=30.0)       # 5 x 20 x 30 s = 50 min
    md, fails = report.build(n_resamples=500)
    assert fails == []
    for label in SYSTEMS.values():
        assert label in md
    assert "relative reduction" in md and "MET" in md
    assert report.main(500) == 0
