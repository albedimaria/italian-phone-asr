import pytest

from iphasr import review
from iphasr.paths import manifest_path, read_jsonl, write_jsonl

ROW = {"id": "spk01_browser_l007", "speaker": "spk01", "audio": "x.wav",
       "text": "Ci vediamo alle 15:30 da Mario.",
       "entities": [{"type": "orario", "text": "15:30"}, {"type": "nome", "text": "Mario"}]}


def test_unverified_row_keeps_script_text():
    r = review.apply_verified(ROW, {})
    assert r["text"] == ROW["text"] and r["verified"] is False and len(r["entities"]) == 2


def test_verified_row_uses_what_was_said_and_drops_unsaid_entities():
    r = review.apply_verified(ROW, {ROW["id"]: "ci vediamo alle tre e mezza da Mario"})
    assert r["text"] == "ci vediamo alle tre e mezza da Mario" and r["script_text"] == ROW["text"]
    assert [e["type"] for e in r["entities"]] == ["nome"]       # "15:30" was not said: not scorable
    assert r["verified"] is True


def test_same_reading_keeps_the_entity():
    r = review.apply_verified(ROW, {ROW["id"]: "Ci vediamo alle quindici e trenta da Mario."})
    assert {e["type"] for e in r["entities"]} == {"orario", "nome"}


def test_apply_is_idempotent_and_reversible():
    v = {ROW["id"]: "ci vediamo alle tre e mezza da Mario"}
    once = review.apply_verified(ROW, v)
    assert review.apply_verified(once, v) == once
    back = review.apply_verified(once, {})        # verdict withdrawn: back to the script
    assert back["text"] == ROW["text"] and len(back["entities"]) == 2


def test_review_app_saves_verdict_and_rewrites_manifest(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("IPASR_DATA", str(tmp_path))
    wav = tmp_path / "a.wav"; wav.write_bytes(b"RIFF")
    write_jsonl(manifest_path("test_human"), [{**ROW, "audio": str(wav)}])
    c = TestClient(review.create_review_app("spk01"))
    assert c.get("/api/rows").json()[0]["said"] is None
    assert c.get(f"/audio/{ROW['id']}").status_code == 200
    assert c.get("/audio/other").status_code == 404
    assert c.post("/api/verify", json={"id": ROW["id"], "said": "ci vediamo alle tre e mezza da Mario"}).json()["ok"]
    m = read_jsonl(manifest_path("test_human"))[0]
    assert m["verified"] and m["text"].startswith("ci vediamo") and m["script_text"] == ROW["text"]
    assert c.post("/api/verify", json={"id": "nope", "said": "x"}).status_code == 400


def test_synth_entities_review_is_separate_and_worst_first(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from iphasr.paths import transcript_path
    monkeypatch.setenv("IPASR_DATA", str(tmp_path))
    wav = tmp_path / "a.wav"; wav.write_bytes(b"RIFF")
    rows = [{"id": f"se_{k}", "speaker": "chatterbox-default", "audio": str(wav), "text": "costa 240,99 euro",
             "entities": [{"type": "prezzo", "text": "240,99 euro"}]} for k in ("a", "b")]
    write_jsonl(manifest_path("test_synth_entities"), rows)
    write_jsonl(transcript_path("whisper-large-v3-turbo", "test_synth_entities"),
                [{"id": "se_a", "hyp": "costa 240,99 euro"}, {"id": "se_b", "hyp": "costa duecento"}])
    c = TestClient(review.create_review_app(None, "test_synth_entities"))
    got = c.get("/api/rows").json()
    assert [r["id"] for r in got] == ["se_b", "se_a"] and got[0]["worst"] > got[1]["worst"]
    assert c.post("/api/verify", json={"id": "se_b", "said": "costa 240,99"}).json()["ok"]
    m = {r["id"]: r for r in read_jsonl(manifest_path("test_synth_entities"))}
    assert m["se_b"]["verified"] and m["se_b"]["entities"] == []     # "euro" not said: entity not scorable
    assert review.load_verified("test_human") == {}                   # test_human verdicts untouched
