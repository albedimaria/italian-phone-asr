"""`bench record ...`: sessions, server, outbound mobile call, fetch, manifest."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from .paths import REPO, data_root, manifest_path, write_jsonl
from .recorder import load_session, new_session, save_session, sessions_dir
from .script import load_script


def _env():
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")


def _client():
    from twilio.rest import Client
    _env()
    return Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])


def ngrok_url() -> str | None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as r:
            tunnels = json.load(r)["tunnels"]
        return next(t["public_url"] for t in tunnels if t["public_url"].startswith("https"))
    except Exception:
        return None


def _set_env_var(key: str, value: str) -> None:
    p = REPO / ".env"
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    lines = [l for l in lines if not l.startswith(f"{key}=")] + [f"{key}={value}"]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ[key] = value


def setup() -> int:
    """One-off: create a Twilio API key and a TwiML App (Alberto's OK required)."""
    c = _client()
    if not os.environ.get("TWILIO_API_KEY"):
        k = c.new_keys.create(friendly_name="italian-phone-asr recorder")
        _set_env_var("TWILIO_API_KEY", k.sid)
        _set_env_var("TWILIO_API_SECRET", k.secret)
        print("created API key", k.sid)
    if not os.environ.get("TWILIO_TWIML_APP_SID"):
        app = c.applications.create(friendly_name="italian-phone-asr recorder",
                                    voice_url="https://example.invalid/twiml/start", voice_method="POST")
        _set_env_var("TWILIO_TWIML_APP_SID", app.sid)
        print("created TwiML app", app.sid)
    return 0


def serve(port: int = 8731, local: bool = False) -> int:
    """Start ngrok (if not running), point the TwiML App at it, run the server.

    --local: page preview only, no tunnel and no Twilio changes (calls will not work).
    """
    import uvicorn

    _env()
    if local:
        from .recorder import create_app
        os.environ.setdefault("RECORDER_PUBLIC_URL", f"http://127.0.0.1:{port}")
        uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="warning")
        return 0
    url = ngrok_url()
    proc = None
    if not url:
        proc = subprocess.Popen(["ngrok", "http", str(port), "--log", "stdout"], stdout=subprocess.DEVNULL)
        for _ in range(40):
            time.sleep(0.25)
            if url := ngrok_url():
                break
    if not url:
        print("ngrok did not start")
        return 1
    os.environ["RECORDER_PUBLIC_URL"] = url
    c = _client()
    c.applications(os.environ["TWILIO_TWIML_APP_SID"]).update(voice_url=f"{url}/twiml/start", voice_method="POST")
    print(f"public URL: {url}  (TwiML App updated)")
    for p in sorted(sessions_dir().glob("*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        if "finished" not in s:
            print(f"  {s['speaker']:<8} {s['channel']:<8} {url}/s/{s['token']}")
    from .recorder import create_app
    try:
        uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="warning")
    finally:
        if proc:
            proc.terminate()
    return 0


def new(speaker: str, channel: str, r_moscia: bool) -> int:
    s = new_session(speaker, channel, r_moscia)
    url = ngrok_url() or "<public-url>"
    print(f"session {s['token']} for {speaker} ({channel}): {url}/s/{s['token']}")
    return 0


def call(token: str, to: str) -> int:
    """Mobile path: dial the speaker's phone, same TwiML as the browser."""
    _env()
    s = load_session(token)
    url = ngrok_url()
    if not url:
        print("server/ngrok not running: start `bench record serve` first")
        return 1
    c = _client()
    cl = c.calls.create(to=to, from_=os.environ["TWILIO_PHONE_NUMBER"],
                        url=f"{url}/twiml/start?session={s['token']}", method="POST")
    print("calling", to, cl.sid)
    return 0


def final_takes(s: dict) -> dict[int, dict]:
    """Last take ended with '#' per line index (earlier takes were redone)."""
    out = {}
    for t in s["takes"]:
        if t.get("digits") == "#" and t.get("recording_sid"):
            out[t["i"]] = t
    return out


def fetch(delete_remote: bool = True) -> int:
    """Download final takes as WAV, delete every take from Twilio, rebuild the manifest."""
    import soundfile as sf

    c = _client()
    sid, tok = os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"]
    import httpx

    script = {l.id: l for l in load_script()}
    lines = list(script.values())
    rows = []
    for p in sorted(sessions_dir().glob("*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        out_dir = data_root() / "test_human" / "audio" / s["speaker"] / s["channel"]
        out_dir.mkdir(parents=True, exist_ok=True)
        finals = final_takes(s)
        for i, t in sorted(finals.items()):
            wav = out_dir / f"{lines[i].id}.wav"
            if not wav.exists():
                r = httpx.get(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Recordings/{t['recording_sid']}.wav",
                              auth=(sid, tok), timeout=60, follow_redirects=True)
                r.raise_for_status()
                wav.write_bytes(r.content)
            info = sf.info(str(wav))
            line = lines[i]
            rows.append({"id": f"{s['speaker']}_{s['channel']}_{line.id}", "audio": str(wav), "text": line.text,
                         "entities": line.entities, "speaker": s["speaker"], "channel": s["channel"],
                         "r_moscia": s.get("r_moscia", False), "duration": info.duration,
                         "sample_rate": info.samplerate})
        if delete_remote:
            for t in s["takes"] + s.get("recordings", []):
                rs = t.get("recording_sid")
                if rs and not t.get("deleted_remote"):
                    try:
                        c.recordings(rs).delete()
                    except Exception as e:  # already deleted
                        if "20404" not in str(e):
                            raise
                    t["deleted_remote"] = True
            save_session(s)
    write_jsonl(manifest_path("test_human"), rows)
    spk = {r["speaker"] for r in rows}
    print(f"test_human: {len(rows)} utts, {len(spk)} speakers, {sum(r['duration'] for r in rows) / 60:.1f} min")
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("record", help="test_human recording sessions (Twilio)")
    rs = p.add_subparsers(dest="rcmd", required=True)
    rs.add_parser("setup", help="create Twilio API key + TwiML App (one-off)")
    q = rs.add_parser("serve"); q.add_argument("--port", type=int, default=8731); q.add_argument("--local", action="store_true")
    q = rs.add_parser("new"); q.add_argument("speaker"); q.add_argument("--channel", choices=["browser", "mobile"], default="browser")
    q.add_argument("--r-moscia", action="store_true")
    q = rs.add_parser("call"); q.add_argument("token"); q.add_argument("to")
    q = rs.add_parser("fetch"); q.add_argument("--keep-remote", action="store_true")


def run(args) -> int:
    if args.rcmd == "setup":
        return setup()
    if args.rcmd == "serve":
        return serve(args.port, args.local)
    if args.rcmd == "new":
        return new(args.speaker, args.channel, args.r_moscia)
    if args.rcmd == "call":
        return call(args.token, args.to)
    if args.rcmd == "fetch":
        return fetch(not args.keep_remote)
    return 2
