"""Hybrid scoring — where the ML contribution lives (Part B ranking logic).

Each candidate image is scored by four complementary signals, each targeting a specific
failure mode of vanilla whole-image CLIP:

  global_clip     cos(query_text, whole_image)          semantic + scene baseline
  caption_sim     cos(query, BLIP caption)              style/vibe inference (no garment
                                                        words needed)
  attribute_match query attrs vs CLIP-tagged image      explicit multi-attribute grounding
  region_binding  (color,garment) pairs vs regions      COMPOSITIONALITY / color binding

Fusion:
  1. Compute each signal for every candidate.
  2. Min-max normalize each signal ACROSS the candidate pool -> [0,1]. This is essential:
     CLIP cosines live in a narrow band (~0.15-0.35) while attribute/region scores span
     [0,1]; without normalization CLIP would be silently down-weighted. Normalizing puts
     every signal on equal footing before applying the configured weights.
  3. Only signals that are *active* for this query contribute (e.g. region_binding is off
     when the query has no color-garment binding); the remaining weights are renormalized
     so they always sum to 1. This prevents an inactive signal from dragging scores down.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..utils.schema import ImageRecord, QuerySpec


@dataclass
class ScoredImage:
    image_id: int
    image_path: str
    score: float
    caption: str = ""
    signals: Dict[str, float] = field(default_factory=dict)   # normalized per-signal
    raw_signals: Dict[str, float] = field(default_factory=dict)  # pre-normalization


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class HybridScorer:
    def __init__(self, retrieval_cfg: dict):
        self.weights = dict(retrieval_cfg["weights"])
        self.attr_cfg = retrieval_cfg.get("attribute_match", {})
        self.region_cfg = retrieval_cfg.get("region_binding", {})

    # --- individual signals -------------------------------------------- #
    def _attribute_match(self, q: QuerySpec, attrs: Dict[str, Dict[str, float]]) -> float:
        pairs: List[float] = []
        for axis, values in q.attributes.items():
            axis_conf = attrs.get(axis, {})
            for v in values:
                pairs.append(float(axis_conf.get(v, 0.0)))
        if not pairs:
            return 0.0
        if self.attr_cfg.get("mode", "soft") == "hard":
            thr = self.attr_cfg.get("present_threshold", 0.25)
            return float(np.mean([1.0 if p >= thr else 0.0 for p in pairs]))
        return float(np.mean(pairs))

    def _region_binding(self, q: QuerySpec, rec: ImageRecord) -> float:
        bound = [b for b in q.bindings if b.is_bound()]
        if not bound:
            return 0.0
        cw = self.region_cfg.get("color_weight", 0.5)
        gw = self.region_cfg.get("garment_weight", 0.5)
        per_binding: List[float] = []
        for b in bound:
            best = 0.0
            for r in rec.regions:
                g_conf = float(r.type_scores.get(b.garment_type, 0.0))
                if r.garment_type == b.garment_type:
                    g_conf = max(g_conf, 0.5)  # exact typed match floor
                c_conf = float(r.colors.get(b.color, 0.0))
                best = max(best, gw * g_conf + cw * c_conf)
            per_binding.append(best)
        # geometric-ish emphasis: an image must satisfy ALL bindings, so average is a
        # reasonable aggregate; a min() would be harsher. Mean keeps partial credit.
        return float(np.mean(per_binding))

    # --- fusion --------------------------------------------------------- #
    def score(
        self,
        query: QuerySpec,
        q_clip: np.ndarray,
        q_sent: Optional[np.ndarray],
        candidates: List[ImageRecord],
        global_vecs: np.ndarray,
        caption_vecs: Optional[np.ndarray],
    ) -> List[ScoredImage]:
        n = len(candidates)
        if n == 0:
            return []

        raw = {
            "global_clip": global_vecs @ q_clip,
            "caption_sim": (caption_vecs @ q_sent
                            if (caption_vecs is not None and q_sent is not None
                                and caption_vecs.shape[1] > 0)
                            else None),
            "attribute_match": np.array(
                [self._attribute_match(query, c.attributes) for c in candidates]),
            "region_binding": np.array(
                [self._region_binding(query, c) for c in candidates]),
        }

        # which signals are active for THIS query?
        active: Dict[str, bool] = {
            "global_clip": True,
            "caption_sim": raw["caption_sim"] is not None,
            "attribute_match": bool(query.attributes),
            "region_binding": query.has_bindings(),
        }

        # normalize active signals and renormalize weights over them
        norm: Dict[str, np.ndarray] = {}
        active_weight_sum = 0.0
        for k, is_on in active.items():
            if is_on and raw[k] is not None:
                norm[k] = _minmax(np.asarray(raw[k], dtype=np.float64))
                active_weight_sum += self.weights.get(k, 0.0)
        if active_weight_sum <= 0:
            active_weight_sum = 1.0

        fused = np.zeros(n, dtype=np.float64)
        for k, vec in norm.items():
            w = self.weights.get(k, 0.0) / active_weight_sum
            fused += w * vec

        results: List[ScoredImage] = []
        for i, c in enumerate(candidates):
            results.append(ScoredImage(
                image_id=c.image_id,
                image_path=c.image_path,
                score=float(fused[i]),
                caption=c.caption,
                signals={k: float(norm[k][i]) for k in norm},
                raw_signals={
                    k: (float(raw[k][i]) if raw[k] is not None else 0.0)
                    for k in raw
                },
            ))
        results.sort(key=lambda s: s.score, reverse=True)
        return results
