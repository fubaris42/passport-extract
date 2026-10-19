"""
Passport extraction web service (production-grade, OS-agnostic).

Run (dev):   uv run flask --app web run --port 5000
Run (prod):  uv run python web.py            # serves via waitress (cross-platform)

Endpoints:
    GET  /            -> upload page (drag & drop)
    GET  /health      -> liveness/readiness probe (no PII)
    POST /api/extract -> multipart 'image' (or several 'images') => JSON

Design notes for production:
  * PII never touches disk — uploads are decoded in memory.
  * OCR runs on a thread-safe pool of engines (PPX_POOL_SIZE), so the
    multithreaded WSGI server handles concurrent uploads without sharing a
    non-thread-safe RapidOCR instance.
  * Nothing PII is logged — only counts, latency and review outcome.

Config via environment:
    PPX_POOL_SIZE     OCR engines in the pool (default 2)
    PPX_MAX_UPLOAD_MB max request size in MB (default 20)
    PPX_MAX_FILES     max images per request (default 10)
    PPX_PORT          port (default 5000)
    PPX_THREADS       waitress threads (default = 4 * pool size)
    PPX_MAX_EDGE      image downscale long-edge (see passport_extract)
"""
from __future__ import annotations

import os
import time
import logging

from flask import Flask, request, jsonify, render_template_string, g

import passport_extract as pe

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
POOL_SIZE = int(os.getenv("PPX_POOL_SIZE", "2"))
MAX_UPLOAD_MB = int(os.getenv("PPX_MAX_UPLOAD_MB", "20"))
MAX_FILES = int(os.getenv("PPX_MAX_FILES", "10"))
PORT = int(os.getenv("PPX_PORT", "5000"))
THREADS = int(os.getenv("PPX_THREADS", str(POOL_SIZE * 4)))
ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("passport-service")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Thread-safe pool of OCR engines, built once at startup (also warms the model
# so the first real request isn't slow).
_POOL: pe.EnginePool | None = None


def _pool() -> pe.EnginePool:
    global _POOL
    if _POOL is None:
        log.info("building OCR pool size=%d", POOL_SIZE)
        _POOL = pe.EnginePool(POOL_SIZE)
    return _POOL


def _extract_upload(file_storage) -> dict:
    filename = file_storage.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED:
        return {"_file": filename, "_error": "unsupported file type"}
    data = file_storage.read()
    if not data:
        return {"_file": filename, "_error": "empty file"}
    try:
        with _pool().borrow() as eng:          # in-memory: no PII on disk
            result = pe.extract(data, engine=eng, name=filename)
        return result
    except Exception as e:  # noqa
        # log the failure type, never the image or field values
        log.warning("extract_failed file_ext=%s err=%s", ext, type(e).__name__)
        return {"_file": filename, "_error": "could not process image"}


@app.before_request
def _start_timer():
    g._t0 = time.time()


@app.post("/api/extract")
def api_extract():
    files = request.files.getlist("images") or []
    if "image" in request.files:
        files = [request.files["image"]] + files
    if not files:
        return jsonify({"error": "no file uploaded (field 'image')"}), 400
    if len(files) > MAX_FILES:
        return jsonify({"error": f"too many files (max {MAX_FILES})"}), 400

    results = [_extract_upload(f) for f in files]
    ok = sum(1 for r in results if not r.get("_error"))
    review = sum(1 for r in results if r.get("_needs_review"))
    dt = time.time() - g.get("_t0", time.time())
    # metrics only — no PII
    log.info("extract n=%d ok=%d review=%d ms=%d", len(results), ok, review,
             int(dt * 1000))
    return jsonify(results if len(results) > 1 else results[0])


@app.get("/health")
def health():
    return jsonify({"status": "ok", "pool_size": POOL_SIZE})


@app.errorhandler(413)
def _too_large(_e):
    return jsonify({"error": f"request too large (max {MAX_UPLOAD_MB} MB)"}), 413


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Passport Extractor</title>
<style>
  :root{--bg:#0f1220;--card:#1a1f36;--fg:#e8ebf5;--mut:#9aa4c0;--acc:#5b8cff;--ok:#3ddc97;--err:#ff6b6b}
  *{box-sizing:border-box}
  body{margin:0;font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg)}
  .wrap{max-width:820px;margin:0 auto;padding:32px 16px}
  h1{font-size:22px;margin:0 0 4px}
  p.sub{color:var(--mut);margin:0 0 24px}
  .drop{display:block;border:2px dashed #33406b;border-radius:14px;padding:40px;
        text-align:center;cursor:pointer;transition:.15s;background:var(--card)}
  .drop.hover{border-color:var(--acc);background:#20264a}
  .drop input{display:none}
  .btn{background:var(--acc);color:#fff;border:0;border-radius:9px;padding:10px 18px;
       font-size:15px;cursor:pointer;margin-top:14px}
  .card{background:var(--card);border-radius:14px;padding:18px 20px;margin-top:20px}
  .row{display:flex;justify-content:space-between;gap:12px;padding:7px 0;border-bottom:1px solid #262c4d}
  .row:last-child{border-bottom:0}
  .k{color:var(--mut)} .v{font-weight:600;text-align:right}
  .name{font-size:17px;margin:0 0 8px}
  .flag{font-size:12px;color:var(--ok)} .flag.warn{color:var(--err)}
  .muted{color:var(--mut);font-size:13px}
  .review{margin-top:8px;font-size:13px}
</style></head>
<body><div class="wrap">
  <h1>Passport Data Extractor</h1>
  <p class="sub">Upload an Indonesian passport photo. Data is read from the MRZ (check-digit verified) plus the printed zone.</p>
  <label class="drop" id="drop">
    <div>📷 Drop a passport image here, or click to choose</div>
    <div class="muted" style="margin-top:6px">JPG / PNG · up to __MAXMB__ MB · multiple allowed</div>
    <input type="file" id="file" accept="image/*" multiple>
    <div><button class="btn" type="button" onclick="document.getElementById('file').click()">Choose file</button></div>
  </label>
  <div id="out"></div>
</div>
<script>
const LABELS={jenis_type:"Jenis / Type",kode_negara:"Kode Negara / Country",
no_paspor:"No. Paspor",nama_lengkap:"Nama Lengkap",kewarganegaraan:"Kewarganegaraan",
tanggal_lahir:"Tanggal Lahir",sex:"Sex",tanggal_terbit:"Tanggal Terbit / Date of Issue",
tempat_lahir:"Tempat Lahir",tanggal_habis_berlaku:"Tanggal Habis Berlaku",
kantor_yang_mengeluarkan:"Kantor yang Mengeluarkan"};
const ORDER=Object.keys(LABELS);
const drop=document.getElementById('drop'),file=document.getElementById('file'),out=document.getElementById('out');
;['dragenter','dragover'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('hover')}));
;['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('hover')}));
drop.addEventListener('drop',ev=>{send(ev.dataTransfer.files)});
file.addEventListener('change',()=>send(file.files));
function card(d){
  if(d._error) return `<div class="card"><div class="name">${d._file||''}</div><div class="flag warn">Error: ${d._error}</div></div>`;
  const v=d._mrz_valid||{};
  const rows=ORDER.map(k=>{
    let val=d[k]==null?'<span class="muted">—</span>':d[k];
    let chk=v[k]===true?' <span class="flag">✓ MRZ</span>':v[k]===false?' <span class="flag warn">✗ check</span>':'';
    return `<div class="row"><span class="k">${LABELS[k]}</span><span class="v">${val}${chk}</span></div>`;
  }).join('');
  let foot='';
  if(d._scanner_watermark) foot+='<div class="review flag warn">⚠ scanner watermark — re-scan before strict submission</div>';
  if(d._needs_review) foot+=`<div class="review flag warn">Needs review: ${(d._review_reasons||[]).join(', ')}</div>`;
  else foot+='<div class="review flag">✓ auto-clean</div>';
  return `<div class="card"><div class="name">${d.nama_lengkap||d._file||''}</div>${rows}${foot}</div>`;
}
async function send(files){
  if(!files||!files.length)return;
  out.innerHTML='<div class="card muted">Extracting…</div>';
  const fd=new FormData();
  for(const f of files) fd.append('images',f);
  try{
    const r=await fetch('/api/extract',{method:'POST',body:fd});
    const d=await r.json();
    if(d.error){out.innerHTML='<div class="card flag warn">'+d.error+'</div>';return;}
    const arr=Array.isArray(d)?d:[d];
    out.innerHTML=arr.map(card).join('');
  }catch(e){out.innerHTML='<div class="card flag warn">Request failed: '+e+'</div>'}
}
</script></body></html>""".replace("__MAXMB__", str(MAX_UPLOAD_MB))


@app.get("/")
def index():
    return render_template_string(PAGE)


if __name__ == "__main__":
    _pool()  # warm the model before accepting traffic
    try:
        from waitress import serve
        log.info("serving on 0.0.0.0:%d threads=%d", PORT, THREADS)
        serve(app, host="0.0.0.0", port=PORT, threads=THREADS)
    except ImportError:
        log.warning("waitress not installed; falling back to Flask dev server")
        app.run(host="0.0.0.0", port=PORT)
