"""Compositionality ("swap test") — an objective, label-free measure of color binding.

The assignment's hint says CLIP "struggles with compositionality (e.g. distinguishing
'red shirt with blue pants' from 'blue shirt with red pants')". This module MEASURES that
rather than asserting it.

Method
------
For a query Q and its colour-swapped twin Q' (identical words, swapped colours):
  * A bag-of-words retriever cannot tell them apart, so top-k(Q) and top-k(Q') are nearly
    the SAME set  ->  overlap ~ 1.0.
  * A compositional retriever binds each colour to its garment, so the two queries pull
    DIFFERENT images  ->  overlap much lower.

We report overlap@k for two systems over the same index:
  * vanilla CLIP  : top-k straight from the global FashionCLIP index (the baseline the
                    assignment says to beat)
  * hybrid        : the full 4-signal system (adds region binding)

Lower overlap = more compositionally sensitive. The delta between the two columns is the
measured contribution of the region-binding mechanism.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# Colour-swapped query pairs. Each pair is identical except the two colours are exchanged,
# so any difference in results is attributable purely to colour<->garment binding.
SWAP_PAIRS: List[Tuple[str, str]] = [
    # NOTE ON SAMPLE SIZE: each pair contributes k slots, so 5 pairs at k=10 gives only 50
    # data points -> one image changing moves the mean by 0.02. We therefore use 12 pairs
    # (120 slots) so the reported mean is not dominated by single-image noise.
    ("a red tie and a white shirt in a formal setting",
     "a white tie and a red shirt in a formal setting"),
    ("a person wearing a blue shirt and red trousers",
     "a person wearing a red shirt and blue trousers"),
    ("a black jacket and white trousers",
     "a white jacket and black trousers"),
    ("a green top and a yellow skirt",
     "a yellow top and a green skirt"),
    ("a woman in a pink blouse and a black skirt",
     "a woman in a black blouse and a pink skirt"),
    ("a white blouse and blue jeans",
     "a blue blouse and white jeans"),
    ("a black dress and a white jacket",
     "a white dress and a black jacket"),
    ("a yellow top and a blue skirt",
     "a blue top and a yellow skirt"),
    ("a red skirt and a white blouse",
     "a white skirt and a red blouse"),
    ("a brown coat and black trousers",
     "a black coat and brown trousers"),
    ("a green jacket and a white shirt",
     "a white jacket and a green shirt"),
    ("a purple top and black trousers",
     "a black top and purple trousers"),
]


def _overlap_at_k(a: List[int], b: List[int], k: int) -> float:
    """|A ∩ B| / k  over the top-k id lists (1.0 = identical results)."""
    sa, sb = set(a[:k]), set(b[:k])
    return len(sa & sb) / float(k) if k else 0.0


def _clip_only_topk(retriever, query: str, k: int) -> List[int]:
    """Vanilla-CLIP baseline: nearest neighbours of the query text in the global index."""
    q = retriever.clip.encode_text([query])[0]
    _, ids = retriever.store.global_index.search(q, k)
    return [int(i) for i in ids[0] if i >= 0]


def run_swap_test(retriever, k: int = 10) -> Dict:
    """Run every swap pair through both systems and return measured overlaps."""
    rows = []
    for qa, qb in SWAP_PAIRS:
        clip_a = _clip_only_topk(retriever, qa, k)
        clip_b = _clip_only_topk(retriever, qb, k)
        hyb_a = [h.image_id for h in retriever.retrieve(qa, k=k)]
        hyb_b = [h.image_id for h in retriever.retrieve(qb, k=k)]
        rows.append({
            "query_a": qa,
            "query_b": qb,
            "clip_overlap": _overlap_at_k(clip_a, clip_b, k),
            "hybrid_overlap": _overlap_at_k(hyb_a, hyb_b, k),
        })

    clip_mean = float(np.mean([r["clip_overlap"] for r in rows]))
    hyb_mean = float(np.mean([r["hybrid_overlap"] for r in rows]))
    return {
        "k": k,
        "pairs": rows,
        "summary": {
            "vanilla_clip_mean_overlap": clip_mean,
            "hybrid_mean_overlap": hyb_mean,
            "absolute_reduction": clip_mean - hyb_mean,
            "relative_reduction_pct": (
                100.0 * (clip_mean - hyb_mean) / clip_mean if clip_mean > 0 else 0.0
            ),
        },
    }
