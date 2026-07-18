#!/usr/bin/env python3
"""Multimodal Fashion & Context Retrieval — unified CLI.

    python main.py index                      # Part A: build the index from data/raw
    python main.py query "a red tie and a white shirt in a formal setting" -k 10
    python main.py evaluate                    # run the 5 official queries -> outputs/results

All behaviour is driven by configs/*.yaml; this file only wires argument parsing to the
pipeline classes.
"""
from __future__ import annotations

import argparse
import sys

from src.utils.config import load_config
from src.utils.logger import get_logger


def cmd_index(args) -> int:
    from src.indexing import Indexer

    cfg = load_config()
    log = get_logger("cli")
    log.info("Device resolved to: %s", cfg.device)
    Indexer(cfg, chunk_size=args.chunk_size).run(image_dir=args.images)
    return 0


def cmd_query(args) -> int:
    from src.retrieval import Retriever

    cfg = load_config()
    retriever = Retriever(cfg)
    try:
        hits = retriever.retrieve(args.query, k=args.k, explain=True)
        if not hits:
            print("No results (is the index built? run `python main.py index`).")
            return 1
        print(f"\nTop-{len(hits)} for: {args.query!r}\n" + "-" * 60)
        for r, h in enumerate(hits, 1):
            sig = " ".join(f"{k}={v:.2f}" for k, v in h.signals.items())
            print(f"{r:2d}. [{h.score:.3f}] {h.image_path}")
            print(f"     caption: {h.caption[:80]}")
            print(f"     signals: {sig}")
    finally:
        retriever.close()
    return 0


def cmd_evaluate(args) -> int:
    from src.evaluation import run_evaluation

    cfg = load_config()
    report = run_evaluation(cfg, k=args.k, labels_path=args.labels)
    s = report["summary"]
    print("\n=== Evaluation summary ===")
    print(f"queries        : {s['num_queries']}")
    print(f"mean latency   : {s['mean_latency_ms']:.1f} ms")
    if s["mean_precision_at_k"] is not None:
        print(f"mean P@{args.k}       : {s['mean_precision_at_k']:.3f}")
    print(f"contact sheet  : {cfg.path('outputs', 'results') / 'eval.html'}")
    return 0


def cmd_compositional(args) -> int:
    """Objective, label-free check that we beat vanilla CLIP on colour binding."""
    import json

    from src.evaluation.compositional import run_swap_test
    from src.retrieval import Retriever

    cfg = load_config()
    retriever = Retriever(cfg)
    try:
        out = run_swap_test(retriever, k=args.k)
    finally:
        retriever.close()

    print(f"\nColour-swap test @{args.k}  (lower = better compositional separation)")
    print("-" * 78)
    print(f"{'colour-swapped pair':<50}{'CLIP':>9}{'hybrid':>10}")
    print("-" * 78)
    for p in out["pairs"]:
        print(f"{p['query_a'][:48]:<50}{p['clip_overlap']:>9.2f}{p['hybrid_overlap']:>10.2f}")
    s = out["summary"]
    print("-" * 78)
    print(f"{'MEAN':<50}{s['vanilla_clip_mean_overlap']:>9.3f}"
          f"{s['hybrid_mean_overlap']:>10.3f}")
    print(f"\nrelative reduction vs vanilla CLIP: {s['relative_reduction_pct']:.1f}%")

    out_path = cfg.path("outputs", "results") / "compositional.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fashion & context retrieval CLI")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("index", help="build the search index (Part A)")
    pi.add_argument("--images", default=None, help="image dir (default: configs paths.yaml)")
    pi.add_argument("--chunk-size", type=int, default=16)
    pi.set_defaults(func=cmd_index)

    pq = sub.add_parser("query", help="retrieve top-k for a natural-language query (Part B)")
    pq.add_argument("query", help="natural language query string")
    pq.add_argument("-k", type=int, default=10)
    pq.set_defaults(func=cmd_query)

    pe = sub.add_parser("evaluate", help="run the 5 official evaluation queries")
    pe.add_argument("-k", type=int, default=10)
    pe.add_argument("--labels", default=None,
                    help="optional JSON {query: [relevant_image_ids]} for precision@k")
    pe.set_defaults(func=cmd_evaluate)

    pc = sub.add_parser("compositional",
                        help="colour-swap test: measure compositionality vs vanilla CLIP")
    pc.add_argument("-k", type=int, default=10)
    pc.set_defaults(func=cmd_compositional)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
