#!/usr/bin/env python3
"""Optional dataset builder -> populates data/raw/.

The retriever is dataset-agnostic (it just needs a folder of images), so this is a
convenience, not a dependency. It assembles a HYBRID corpus that matches the evaluation's
three axes better than any single source:

  * Fashionpedia  -> fine-grained garments, colors, accessories (great for queries 1 & 5)
  * COCO 'person'  -> real scenes: offices, streets, parks, homes (great for queries 2,3,4)

Usage (needs `pip install datasets`):
    python scripts/prepare_dataset.py --fashionpedia 700 --coco-person 300

If you already downloaded images another way, just drop them into data/raw/ and skip this.
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"


def _save_stream(ds_iter, out_dir: Path, prefix: str, limit: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for ex in ds_iter:
        if n >= limit:
            break
        img = ex.get("image")
        if img is None:
            continue
        try:
            img.convert("RGB").save(out_dir / f"{prefix}_{n:05d}.jpg", quality=90)
            n += 1
        except Exception:
            continue
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fashionpedia", type=int, default=0,
                    help="number of Fashionpedia images to fetch")
    ap.add_argument("--coco-person", type=int, default=0,
                    help="number of COCO images containing people to fetch")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install first: pip install datasets")

    total = 0
    if args.fashionpedia > 0:
        print(f"Streaming {args.fashionpedia} Fashionpedia images ...")
        ds = load_dataset("detection-datasets/fashionpedia", split="train", streaming=True)
        total += _save_stream(ds, RAW / "fashionpedia", "fp", args.fashionpedia)
    if args.coco_person > 0:
        print(f"Streaming {args.coco_person} COCO images ...")
        ds = load_dataset("rafaelpadilla/coco2017", split="val", streaming=True)
        total += _save_stream(ds, RAW / "coco", "coco", args.coco_person)

    print(f"Done. Saved {total} images under {RAW}")


if __name__ == "__main__":
    main()
