"""Recording server for test_human: one Twilio <Record> per script line.

Flow
  browser (Twilio Voice JS SDK, codec forced to PCMU = G.711 mu-law)
    -> TwiML App voice URL  /twiml/start      (params: session)
    -> /twiml/line?i=N      <Record finishOnKey="#*">
  the speaker presses "Avanti" -> call.sendDigits("#") -> Record ends -> next line
                      "Ripeti" -> call.sendDigits("*") -> same line again (new take)
  /twiml/rec  (recordingStatusCallback) logs RecordingSid per take.
  `bench record fetch` downloads the takes as WAV and deletes them from Twilio.

Each line is its own recording, so segmentation is exact and needs no alignment.
The mobile path reuses the same TwiML: `bench record call` dials the speaker's
phone (trial accounts: verified numbers only), keys # and * on the keypad.

State lives in IPASR_DATA/test_human/sessions/<token>.json, never in the repo.
"""


import json
import os
import secrets
import time
from pathlib import Path

from .paths import REPO, data_root
from .script import load_script

STATIC = REPO / "recorder" / "static"
SDK_URL = "https://cdn.jsdelivr.net/npm/@twilio/voice-sdk@2.18.5/dist/twilio.min.js"


def sessions_dir() -> Path:
    d = data_root() / "test_human" / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_session(token: str) -> dict:
    p = sessions_dir() / f"{token}.json"
    if not token.isalnum() or not p.exists():
        raise KeyError(token)
    return json.loads(p.read_text(encoding="utf-8"))


def save_session(s: dict) -> None:
    p = sessions_dir() / f"{s['token']}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def new_session(speaker: str, channel: str = "browser", r_moscia: bool = False) -> dict:
    s = {"token": secrets.token_hex(8), "speaker": speaker, "channel": channel, "r_moscia": r_moscia,
         "created": time.time(), "current": None, "takes": []}
    save_session(s)
    return s


def env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"missing {name} in .env")
    return v


def create_app():
    from dotenv import load_dotenv
    from fastapi import FastAPI, Form, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from twilio.jwt.access_token import AccessToken
    from twilio.jwt.access_token.grants import VoiceGrant
    from twilio.request_validator import RequestValidator
    from twilio.twiml.voice_response import VoiceResponse

    load_dotenv(REPO / ".env")
    lines = load_script()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    validator = RequestValidator(env("TWILIO_AUTH_TOKEN"))

    async def check_twilio(req: Request) -> dict:
        form = dict(await req.form())
        url = env("RECORDER_PUBLIC_URL").rstrip("/") + req.url.path + (f"?{req.url.query}" if req.url.query else "")
        if not validator.validate(url, form, req.headers.get("X-Twilio-Signature", "")):
            raise HTTPException(403, "bad signature")
        return form

    def xml(vr: VoiceResponse) -> Response:
        return Response(str(vr), media_type="application/xml")

    def record_line(token: str, i: int) -> Response:
        s = load_session(token)
        vr = VoiceResponse()
        if i >= len(lines):
            s["current"] = None
            s["finished"] = time.time()
            save_session(s)
            vr.say("Registrazione completata. Grazie mille!", language="it-IT")
            vr.hangup()
            return xml(vr)
        s["current"] = {"i": i, "since": time.time()}
        save_session(s)
        q = f"session={token}&i={i}"
        vr.record(action=f"/twiml/done?{q}", method="POST", finish_on_key="#*", max_length=60, timeout=30,
                  play_beep=False, trim="do-not-trim",
                  recording_status_callback=f"/twiml/rec?{q}", recording_status_callback_event="completed")
        # Record without input (30 s silence) falls through here: keep the same line
        vr.redirect(f"/twiml/line?{q}", method="POST")
        return xml(vr)

    @app.get("/s/{token}", response_class=HTMLResponse)
    def page(token: str):
        try:
            s = load_session(token)
        except KeyError:
            raise HTTPException(404)
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        return html.replace("{{SDK_URL}}", SDK_URL).replace("{{TOKEN}}", token).replace(
            "{{CHANNEL}}", s["channel"])

    @app.get("/api/{token}/script")
    def script(token: str):
        load_session(token)
        return [{"id": l.id, "text": l.text} for l in lines]

    @app.get("/api/{token}/status")
    def status(token: str):
        s = load_session(token)
        done = {t["i"] for t in s["takes"] if t.get("digits") == "#"}
        return {"current": s["current"], "done": len(done), "total": len(lines), "finished": "finished" in s}

    @app.get("/api/{token}/voice-token")
    def voice_token(token: str):
        s = load_session(token)
        at = AccessToken(env("TWILIO_ACCOUNT_SID"), env("TWILIO_API_KEY"), env("TWILIO_API_SECRET"),
                         identity=f"rec-{s['token']}", ttl=3600)
        at.add_grant(VoiceGrant(outgoing_application_sid=env("TWILIO_TWIML_APP_SID"), incoming_allow=False))
        return JSONResponse({"token": at.to_jwt()})

    @app.post("/twiml/start")
    async def start(req: Request):
        form = await check_twilio(req)
        token = form.get("session") or req.query_params.get("session", "")
        s = load_session(token)
        s.setdefault("calls", []).append({"call_sid": form.get("CallSid"), "at": time.time()})
        save_session(s)
        # resume where the speaker left off
        done = {t["i"] for t in s["takes"] if t.get("digits") == "#"}
        first = next((i for i in range(len(lines)) if i not in done), len(lines))
        return record_line(token, first)

    @app.post("/twiml/line")
    async def line(req: Request, session: str, i: int):
        await check_twilio(req)
        return record_line(session, i)

    @app.post("/twiml/done")
    async def done(req: Request, session: str, i: int):
        form = await check_twilio(req)
        s = load_session(session)
        digits = form.get("Digits", "")
        s["takes"].append({"i": i, "line_id": lines[i].id, "digits": digits,
                           "recording_sid": form.get("RecordingSid"), "duration": form.get("RecordingDuration"),
                           "at": time.time()})
        save_session(s)
        return record_line(session, i + 1 if digits == "#" else i)

    @app.post("/twiml/rec")
    async def rec(req: Request, session: str, i: int):
        form = await check_twilio(req)
        s = load_session(session)
        s.setdefault("recordings", []).append({"i": i, "recording_sid": form.get("RecordingSid"),
                                               "duration": form.get("RecordingDuration"), "at": time.time()})
        save_session(s)
        return Response(status_code=204)

    return app
