#!/usr/bin/env python3
"""Flask demo portal for the fashion retrieval system.

    ./.venv/bin/python webapp/app.py         # then open http://127.0.0.1:5000

Design: a THIN layer over the exact same `Retriever` the CLI uses — the browser shows
precisely what the ML system returns, including how the query was parsed and each result's
per-signal score breakdown (so the retrieval logic is transparent and checkable).

Routes
  GET /                     search page
  GET /evaluate             live run of the 5 official queries + how-scoring-works notes
  GET /api/search?q=&k=     JSON: parsed query + ranked results (+ signal breakdown)
  GET /api/evaluate         JSON: the 5 official queries run live
  GET /img/<image_id>       serves the underlying image file
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, request, send_file

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluate import EVAL_QUERIES
from src.retrieval import Retriever
from src.utils.config import load_config

app = Flask(__name__)

# The Retriever holds GPU models; build once, serialize access with a lock.
_cfg = load_config()
_retriever = Retriever(_cfg)
_lock = threading.Lock()


def _parsed(spec):
    return {
        "attributes": spec.attributes,
        "bindings": [
            {"color": b.color, "garment": b.garment_type}
            for b in spec.bindings if b.is_bound()
        ],
    }


def _run(query: str, k: int):
    """Parse + retrieve under the GPU lock; return a JSON-friendly dict."""
    with _lock:
        spec = _retriever.parser.parse(query)
        t0 = time.perf_counter()
        hits = _retriever.retrieve(query, k=k)
        dt = (time.perf_counter() - t0) * 1000.0
    return {
        "query": query,
        "parsed": _parsed(spec),
        "latency_ms": round(dt, 1),
        "results": [
            {"image_id": h.image_id, "rank": i + 1, "score": round(h.score, 3),
             "caption": h.caption, "signals": {k2: round(v, 3) for k2, v in h.signals.items()}}
            for i, h in enumerate(hits)
        ],
    }


# --------------------------------------------------------------------------- #
@app.route("/api/search")
def api_search():
    q = (request.args.get("q") or "").strip()
    k = min(max(int(request.args.get("k", 12)), 1), 48)
    if not q:
        return jsonify({"error": "empty query"}), 400
    return jsonify(_run(q, k))


@app.route("/api/evaluate")
def api_evaluate():
    out = []
    for item in EVAL_QUERIES:
        row = _run(item["query"], 6)
        row["probes"] = item["probes"]
        out.append(row)
    return jsonify(out)


@app.route("/img/<int:image_id>")
def img(image_id: int):
    rec = _retriever.db.get_image(image_id, with_regions=False)
    if rec is None or not Path(rec.image_path).exists():
        abort(404)
    return send_file(rec.image_path)


@app.route("/")
def index():
    return render_template_string(PAGE, active="search", body=SEARCH_BODY)


@app.route("/evaluate")
def evaluate_page():
    return render_template_string(PAGE, active="evaluate", body=EVAL_BODY)


# --------------------------------------------------------------------------- #
# Templates (inline to keep the demo self-contained).
# --------------------------------------------------------------------------- #
STYLE = """
:root{--bg:#0d1117;--card:#161b22;--bd:#21262d;--fg:#e6edf3;--mut:#8b949e;--acc:#2f81f7;
      --g:#3fb950;--y:#d29922;--r:#f85149;--p:#a371f7}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,sans-serif;
  background:var(--bg);color:var(--fg)}
a{color:var(--acc);text-decoration:none}
.nav{display:flex;gap:18px;align-items:center;padding:14px 24px;border-bottom:1px solid var(--bd);
  position:sticky;top:0;background:var(--bg);z-index:5}
.nav b{font-size:16px}.nav a{color:var(--mut);font-weight:600;padding:4px 2px}
.nav a.on{color:var(--fg);border-bottom:2px solid var(--acc)}
.wrap{max-width:1180px;margin:0 auto;padding:24px}
.searchbar{display:flex;gap:10px;margin:8px 0 6px}
.searchbar input[type=text]{flex:1;padding:13px 16px;font-size:16px;background:var(--card);
  border:1px solid var(--bd);border-radius:10px;color:var(--fg)}
.searchbar select,.searchbar button{padding:0 16px;border-radius:10px;border:1px solid var(--bd);
  background:var(--card);color:var(--fg);font-size:15px;cursor:pointer}
.searchbar button{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:700}
.hint{color:var(--mut);font-size:13px;margin-bottom:14px}
.chip{display:inline-block;background:var(--card);border:1px solid var(--bd);border-radius:999px;
  padding:3px 11px;margin:3px 5px 3px 0;font-size:12.5px}
.chip .k{color:var(--mut)}.chip.bind{border-color:var(--p);color:#d2b8ff}
.parsed{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:12px 14px;margin:10px 0 18px}
.parsed h4{margin:0 0 6px;font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;overflow:hidden}
.card img{width:100%;height:270px;object-fit:cover;display:block;background:#000}
.card .body{padding:10px 12px}
.badge{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.badge .rk{color:var(--mut);font-size:12px}.badge .sc{font-weight:800;font-size:15px}
.cap{font-size:12.5px;color:var(--mut);min-height:32px;line-height:1.35;margin-bottom:8px}
.sig{display:flex;align-items:center;gap:6px;margin:3px 0;font-size:11px}
.sig .lb{width:74px;color:var(--mut);flex:0 0 auto}
.sig .track{flex:1;height:6px;background:#0d1117;border-radius:4px;overflow:hidden}
.sig .fill{height:100%;border-radius:4px}
.sig .vv{width:30px;text-align:right;color:var(--fg)}
.f-global{background:var(--acc)}.f-caption{background:var(--g)}
.f-attribute{background:var(--y)}.f-region{background:var(--p)}.f-cross{background:var(--r)}
.eqhead{margin:26px 0 4px;font-size:18px;font-weight:700}
.eqsub{color:var(--mut);font-size:13px;margin-bottom:8px}
.explain{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:16px 18px;margin-bottom:8px}
.explain h3{margin:0 0 8px}.explain li{margin:5px 0;color:#cdd9e5;font-size:14px}
.explain code{background:#0d1117;padding:1px 6px;border-radius:5px;color:#d2b8ff}
.spin{color:var(--mut);padding:20px 0}
"""

PAGE = """<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Fashion Scene Search</title><style>""" + STYLE + """</style></head><body>
<div class='nav'><b>👗 Fashion Scene Search</b>
  <a href='/' class='{{ "on" if active=="search" else "" }}'>Search</a>
  <a href='/evaluate' class='{{ "on" if active=="evaluate" else "" }}'>Evaluation</a>
  <span style='margin-left:auto;color:var(--mut);font-size:12px'>3,200 images · FashionCLIP + regions</span>
</div>
<div class='wrap'>""" + "{{ body|safe }}" + """</div>
<script>
function sigRow(name,val){
  const key = name.split('_')[0];
  return `<div class='sig'><span class='lb'>${name.replace('_',' ')}</span>
    <span class='track'><span class='fill f-${key}' style='width:${Math.round(val*100)}%'></span></span>
    <span class='vv'>${val.toFixed(2)}</span></div>`;
}
function card(h){
  const sigs = Object.entries(h.signals).map(([k,v])=>sigRow(k,v)).join('');
  return `<div class='card'><img loading='lazy' src='/img/${h.image_id}'>
    <div class='body'><div class='badge'><span class='rk'>#${h.rank} · id ${h.image_id}</span>
      <span class='sc'>${h.score.toFixed(3)}</span></div>
      <div class='cap'>${(h.caption||'').replace(/</g,'&lt;')}</div>${sigs}</div></div>`;
}
function parsedHtml(p){
  let c='';
  for(const [axis,vals] of Object.entries(p.attributes||{}))
    for(const v of vals) c+=`<span class='chip'><span class='k'>${axis}:</span> ${v}</span>`;
  for(const b of (p.bindings||[]))
    c+=`<span class='chip bind'>🔗 ${b.color} → ${b.garment}</span>`;
  if(!c) c="<span class='hint'>no structured attributes detected — pure semantic search</span>";
  return c;
}
</script>
</body></html>"""

SEARCH_BODY = """
<div class='searchbar'>
  <input id='q' type='text' placeholder='Describe a scene… e.g. a red tie and a white shirt in a formal setting'
     onkeydown='if(event.key==="Enter")go()'>
  <select id='k'><option>12</option><option>24</option><option>36</option></select>
  <button onclick='go()'>Search</button>
</div>
<div class='hint'>Try: “a person in a bright yellow raincoat” · “casual weekend outfit for a city walk” ·
  “someone wearing a blue shirt sitting on a park bench”</div>
<div id='parsed'></div>
<div id='results' class='grid'></div>
<script>
async function go(){
  const q=document.getElementById('q').value.trim(); if(!q) return;
  const k=document.getElementById('k').value;
  document.getElementById('parsed').innerHTML="<div class='spin'>Searching…</div>";
  document.getElementById('results').innerHTML="";
  const r=await fetch(`/api/search?q=${encodeURIComponent(q)}&k=${k}`);
  const d=await r.json();
  document.getElementById('parsed').innerHTML=
    `<div class='parsed'><h4>How the system understood your query · ${d.latency_ms} ms</h4>${parsedHtml(d.parsed)}</div>`;
  document.getElementById('results').innerHTML=d.results.map(card).join('');
}
window.onload=()=>{document.getElementById('q').focus();};
</script>
"""

EVAL_BODY = """
<div class='explain'>
  <h3>How retrieval & scoring works</h3>
  <p style='color:#cdd9e5;font-size:14px;margin:0 0 8px'>Every result's final score is a weighted,
  normalized blend of four independent signals. Each targets a different failure mode of plain CLIP:</p>
  <ul>
    <li><b style='color:var(--acc)'>global</b> — FashionCLIP similarity between your text and the whole image (semantic + scene).</li>
    <li><b style='color:var(--g)'>caption</b> — similarity to the image's auto-generated caption (catches <i>style/vibe</i> with no garment words, e.g. “casual weekend”).</li>
    <li><b style='color:var(--y)'>attribute</b> — match of parsed attributes (color / garment / environment / style) to the image's zero-shot tags.</li>
    <li><b style='color:var(--p)'>region</b> — <b>compositional binding</b>: each “(color, garment)” must be satisfied by one detected garment region (defeats “red tie + white shirt” vs the swap).</li>
  </ul>
  <p style='color:var(--mut);font-size:13px;margin:8px 0 0'>Each signal is min-max normalized across the
  candidate pool, then combined with weights <code>0.40 / 0.20 / 0.20 / 0.20</code>. Signals that don't
  apply to a query (e.g. no color-garment binding) are dropped and their weight redistributed. The bars
  under each image below show the normalized contribution of each signal — this is the same data the CLI prints.</p>
</div>
<div id='eval'><div class='spin'>Running the 5 official evaluation queries live…</div></div>
<script>
async function runEval(){
  const r=await fetch('/api/evaluate'); const d=await r.json();
  document.getElementById('eval').innerHTML = d.map(row=>`
    <div class='eqhead'>${row.query.replace(/</g,'&lt;')}</div>
    <div class='eqsub'>probes: ${row.probes} · ${row.latency_ms} ms</div>
    <div class='parsed'>${parsedHtml(row.parsed)}</div>
    <div class='grid'>${row.results.map(card).join('')}</div>`).join('');
}
runEval();
</script>
"""

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "5000"))
    print(f"\n  Fashion Scene Search  ->  http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
