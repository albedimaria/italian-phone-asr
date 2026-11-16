"""Listening check of test_human references (`bench record review`).

The reference of a recorded line is the text that was shown, not what was said: "15:30" can be read
"quindici e trenta", "quindici trenta" or "tre e mezza", and a slip ("il" for "un") is the speaker's,
not the ASR's. After each session someone listens to every take and confirms or corrects the text.

- Verdicts live in IPASR_DATA/test_human/verified.json ({utt_id: said}), never in the repo.
- `apply_verified` rewrites a manifest row: `text` becomes what was said, the shown text moves to
  `script_text`, and entities whose normalized form is no longer in the reference are dropped
  (an entity that was not said cannot be scored). Idempotent: it always starts from `script_text`.
- The review page binds to 127.0.0.1 only: the recordings never leave the machine.
"""

from __future__ import annotations

import json
from pathlib import Path

from .metrics import entity_hit
from .paths import data_root, manifest_path, read_jsonl, write_jsonl


def verified_path() -> Path:
    return data_root() / "test_human" / "verified.json"


def load_verified() -> dict[str, str]:
    p = verified_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_verified(v: dict[str, str]) -> None:
    p = verified_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")


def apply_verified(row: dict, verified: dict[str, str]) -> dict:
    shown = row.get("script_text", row["text"])
    entities = row.get("script_entities", row.get("entities", []))
    if row["id"] not in verified:
        return {**row, "text": shown, "entities": entities, "verified": False}
    said = verified[row["id"]].strip()
    kept = [e for e in entities if entity_hit(e["text"], said)]
    return {**row, "text": said, "script_text": shown, "entities": kept, "script_entities": entities,
            "verified": True}


def apply_to_manifest() -> tuple[int, int]:
    rows = read_jsonl(manifest_path("test_human"))
    v = load_verified()
    rows = [apply_verified(r, v) for r in rows]
    write_jsonl(manifest_path("test_human"), rows)
    return sum(r["verified"] for r in rows), len(rows)


PAGE = """<!doctype html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Verifica riferimenti</title>
<style>
 :root{--bg:#f6f5f2;--ink:#1d1d1b;--muted:#5b5a55;--line:#d9d6cf;--ok:#1f6b45;--chg:#8a5a00}
 body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.5 system-ui,"Segoe UI",sans-serif}
 main{max-width:860px;margin:0 auto;padding:24px 16px 80px}
 h1{font-size:24px;margin:0 0 4px} .muted{color:var(--muted)}
 .row{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px;margin:10px 0}
 .row.cur{outline:3px solid var(--ink)} .row.done{border-left:6px solid var(--ok)} .row.changed{border-left:6px solid var(--chg)}
 .shown{color:var(--muted);font-size:15px} input{width:100%;box-sizing:border-box;font:inherit;padding:8px;
 border:1px solid var(--line);border-radius:6px;margin-top:6px} audio{width:100%;margin-top:6px}
 kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;background:#fff;font-size:14px}
</style></head><body><main>
<h1>Verifica dei riferimenti · __SPK__</h1>
<p class="muted">Ascolta e scrivi quello che è stato <b>detto</b>, con le cifre come le ha lette (es. "quindici e trenta").
<kbd>Invio</kbd> conferma e passa alla successiva (che parte da sola) · <kbd>Ctrl</kbd>+<kbd>Spazio</kbd> riascolta.
Verificate: <b id="n">0</b> / <span id="tot">0</span></p>
<div id="list"></div></main>
<script>
const SPK = "__SPK__"; let rows = [], cur = 0;
const $ = (s, el=document) => el.querySelector(s);
async function load() {
  rows = await (await fetch(`/api/rows?speaker=${encodeURIComponent(SPK)}`)).json();
  $("#tot").textContent = rows.length;
  $("#list").innerHTML = rows.map((r, i) => `<div class="row" id="r${i}">
    <div class="shown">${i + 1}. mostrata: ${r.shown.replace(/</g, "&lt;")}</div>
    <audio controls preload="none" src="/audio/${encodeURIComponent(r.id)}"></audio>
    <input value="${(r.said ?? r.shown).replace(/"/g, "&quot;")}" data-i="${i}"></div>`).join("");
  rows.forEach((r, i) => mark(i));
  document.querySelectorAll("input").forEach(el => {
    el.addEventListener("focus", () => go(+el.dataset.i, false));
    el.addEventListener("keydown", async e => {
      if (e.key === "Enter") { e.preventDefault(); await save(+el.dataset.i); go(+el.dataset.i + 1, true); }
      if (e.key === " " && e.ctrlKey) { e.preventDefault(); const a = $(`#r${cur} audio`); a.currentTime = 0; a.play(); }
    });
  });
  const first = rows.findIndex(r => r.said === null);
  go(first < 0 ? 0 : first, false);
}
function mark(i) {
  const r = rows[i], el = $(`#r${i}`);
  el.classList.toggle("done", r.said !== null && r.said === r.shown);
  el.classList.toggle("changed", r.said !== null && r.said !== r.shown);
  $("#n").textContent = rows.filter(x => x.said !== null).length;
}
function go(i, play) {
  if (i >= rows.length) return;
  document.querySelectorAll(".row").forEach(x => x.classList.remove("cur"));
  cur = i; const el = $(`#r${i}`); el.classList.add("cur");
  el.scrollIntoView({block: "center"}); const inp = $("input", el);
  if (document.activeElement !== inp) inp.focus();
  if (play) { const a = $("audio", el); a.currentTime = 0; a.play(); }
}
async function save(i) {
  const said = $(`#r${i} input`).value.trim();
  await fetch("/api/verify", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({id: rows[i].id, said})});
  rows[i].said = said; mark(i);
}
load();
</script></body></html>"""


def create_review_app(speaker: str):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse

    rows = [r for r in read_jsonl(manifest_path("test_human")) if r["speaker"] == speaker]
    if not rows:
        raise SystemExit(f"no test_human rows for speaker {speaker!r}: run `bench record fetch` first")
    by_id = {r["id"]: r for r in rows}
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE.replace("__SPK__", speaker)

    @app.get("/api/rows")
    def api_rows():
        v = load_verified()
        return [{"id": r["id"], "shown": r.get("script_text", r["text"]), "said": v.get(r["id"])} for r in rows]

    @app.get("/audio/{utt_id}")
    def audio(utt_id: str):
        if utt_id not in by_id:
            raise HTTPException(404)
        return FileResponse(by_id[utt_id]["audio"], media_type="audio/wav")

    @app.post("/api/verify")
    def verify(body: dict):
        if body.get("id") not in by_id or not str(body.get("said", "")).strip():
            raise HTTPException(400)
        v = load_verified()
        v[body["id"]] = str(body["said"]).strip()
        save_verified(v)
        apply_to_manifest()
        return {"ok": True}

    return app


def serve(speaker: str, port: int = 8732) -> int:
    import uvicorn
    print(f"review page for {speaker}: http://127.0.0.1:{port}/  (local only)")
    uvicorn.run(create_review_app(speaker), host="127.0.0.1", port=port, log_level="warning")
    done, total = apply_to_manifest()
    print(f"test_human: {done}/{total} utterances verified by listening")
    return 0
