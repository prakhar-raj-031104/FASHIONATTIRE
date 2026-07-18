"""Evaluation harness for the five official assignment queries.

Even though automated evaluation is not strictly required, running a fixed query suite:
  * makes regressions visible when weights/models change,
  * produces a visual contact sheet for qualitative review,
  * records latency (the scalability question),
  * lets us compute precision@k *if* a relevance-label file is supplied.

Each official query is tagged with which capability it stresses, so the report can point
at the exact mechanism that handles it.
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..retrieval import Retriever
from ..utils.config import Config
from ..utils.logger import get_logger

# The five queries the system is judged on, annotated with the capability they probe.
EVAL_QUERIES: List[Dict[str, str]] = [
    {"query": "A person in a bright yellow raincoat.",
     "probes": "attribute (color + garment)"},
    {"query": "Professional business attire inside a modern office.",
     "probes": "contextual (style + scene)"},
    {"query": "Someone wearing a blue shirt sitting on a park bench.",
     "probes": "complex semantic (color + garment + scene)"},
    {"query": "Casual weekend outfit for a city walk.",
     "probes": "style inference (no explicit garments)"},
    {"query": "A red tie and a white shirt in a formal setting.",
     "probes": "compositional (color<->garment binding)"},
]


def _precision_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    if not relevant_ids:
        return float("nan")
    topk = retrieved_ids[:k]
    hits = sum(1 for i in topk if i in set(relevant_ids))
    return hits / max(len(topk), 1)


def _write_html(results: List[Dict], out_path: Path) -> None:
    """Contact sheet: each query row shows its top-k images with scores for eyeballing."""
    parts = ["<!doctype html><meta charset='utf-8'>",
             "<title>Fashion Retrieval — Eval</title>",
             "<style>body{font-family:system-ui;margin:24px;background:#0d1117;color:#e6edf3}"
             ".q{margin:8px 0 4px;font-size:18px;font-weight:700}"
             ".probe{color:#8b949e;font-size:13px;margin-bottom:8px}"
             ".row{display:flex;gap:10px;overflow-x:auto;padding-bottom:16px;"
             "border-bottom:1px solid #21262d;margin-bottom:16px}"
             ".card{flex:0 0 auto;width:170px}"
             ".card img{width:170px;height:220px;object-fit:cover;border-radius:8px}"
             ".meta{font-size:12px;color:#8b949e;margin-top:4px}</style>"]
    for r in results:
        parts.append(f"<div class='q'>{html.escape(r['query'])}</div>")
        parts.append(f"<div class='probe'>probes: {html.escape(r['probes'])} · "
                     f"{r['latency_ms']:.0f} ms</div>")
        parts.append("<div class='row'>")
        for hit in r["results"]:
            src = "file://" + html.escape(hit["image_path"])
            parts.append(
                f"<div class='card'><img src='{src}' loading='lazy'>"
                f"<div class='meta'>#{hit['rank']} · {hit['score']:.3f}<br>"
                f"{html.escape((hit.get('caption') or '')[:60])}</div></div>"
            )
        parts.append("</div>")
    out_path.write_text("".join(parts), encoding="utf-8")


def run_evaluation(cfg: Config, k: int = 10,
                   labels_path: Optional[str] = None) -> Dict:
    """Run the five official queries, writing eval.json + a visual eval.html contact sheet."""
    log = get_logger("evaluation", cfg.path("outputs", "logs"))
    retriever = Retriever(cfg)

    labels: Dict[str, List[int]] = {}
    if labels_path and Path(labels_path).exists():
        labels = json.loads(Path(labels_path).read_text())

    results: List[Dict] = []
    try:
        for item in EVAL_QUERIES:
            q = item["query"]
            t0 = time.perf_counter()
            hits = retriever.retrieve(q, k=k, explain=True)
            dt = (time.perf_counter() - t0) * 1000.0

            retrieved_ids = [h.image_id for h in hits]
            row = {
                "query": q,
                "probes": item["probes"],
                "latency_ms": dt,
                "results": [
                    {"rank": r + 1, "image_id": h.image_id, "image_path": h.image_path,
                     "score": h.score, "caption": h.caption, "signals": h.signals}
                    for r, h in enumerate(hits)
                ],
            }
            if q in labels:
                row["precision_at_k"] = _precision_at_k(retrieved_ids, labels[q], k)
            results.append(row)
            log.info("[%.0f ms] %s -> %s", dt, q,
                     [h.image_id for h in hits[:5]])
    finally:
        retriever.close()

    out_dir = cfg.path("outputs", "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(results, indent=2))
    _write_html(results, out_dir / "eval.html")
    log.info("Wrote %s and eval.html", out_dir / "eval.json")

    precisions = [r["precision_at_k"] for r in results if "precision_at_k" in r]
    summary = {"num_queries": len(results),
               "mean_latency_ms": sum(r["latency_ms"] for r in results) / len(results),
               "mean_precision_at_k": (sum(precisions) / len(precisions)
                                       if precisions else None)}
    return {"summary": summary, "results": results}
