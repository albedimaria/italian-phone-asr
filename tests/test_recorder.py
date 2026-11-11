"""Recording flow with signed fake Twilio webhooks: next, redo, resume, final takes."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("twilio")

from fastapi.testclient import TestClient  # noqa: E402
from twilio.request_validator import RequestValidator  # noqa: E402

PUBLIC = "https://rec.example.test"
AUTH = "f" * 32


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("IPASR_DATA", str(tmp_path))
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", AUTH)
    monkeypatch.setenv("RECORDER_PUBLIC_URL", PUBLIC)
    from iphasr import recorder
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    return TestClient(recorder.create_app()), recorder


def post(c, path, form):
    sig = RequestValidator(AUTH).compute_signature(PUBLIC + path, form)
    return c.post(path, data=form, headers={"X-Twilio-Signature": sig})


def test_unsigned_webhook_rejected(client):
    c, rec = client
    s = rec.new_session("spk01")
    r = c.post("/twiml/start", data={"session": s["token"]}, headers={"X-Twilio-Signature": "nope"})
    assert r.status_code == 403


def test_flow_next_redo_resume(client):
    c, rec = client
    from iphasr.record_cli import final_takes
    s = rec.new_session("spk01")
    tok = s["token"]
    r = post(c, "/twiml/start", {"session": tok, "CallSid": "CA1"})
    assert r.status_code == 200 and "<Record" in r.text and 'finishOnKey="#*"' in r.text
    assert c.get(f"/api/{tok}/status").json()["current"]["i"] == 0

    post(c, f"/twiml/done?session={tok}&i=0", {"Digits": "#", "RecordingSid": "RE0", "RecordingDuration": "4"})
    assert c.get(f"/api/{tok}/status").json()["current"]["i"] == 1
    post(c, f"/twiml/done?session={tok}&i=1", {"Digits": "*", "RecordingSid": "RE1a", "RecordingDuration": "2"})
    assert c.get(f"/api/{tok}/status").json()["current"]["i"] == 1          # redo keeps the line
    post(c, f"/twiml/done?session={tok}&i=1", {"Digits": "#", "RecordingSid": "RE1b", "RecordingDuration": "3"})
    post(c, f"/twiml/done?session={tok}&i=2", {"Digits": "hangup", "RecordingSid": "RE2", "RecordingDuration": "1"})

    # new call resumes at the first line without a confirmed take
    post(c, "/twiml/start", {"session": tok, "CallSid": "CA2"})
    assert c.get(f"/api/{tok}/status").json()["current"]["i"] == 2
    finals = final_takes(rec.load_session(tok))
    assert {i: t["recording_sid"] for i, t in finals.items()} == {0: "RE0", 1: "RE1b"}


def test_page_and_unknown_session(client):
    c, rec = client
    s = rec.new_session("spk02", channel="mobile")
    page = c.get(f"/s/{s['token']}").text
    assert 'codecPreferences: ["pcmu"]' in page and "{{" not in page
    assert c.get("/s/doesnotexist").status_code == 404
    assert len(c.get(f"/api/{s['token']}/script").json()) >= 100
