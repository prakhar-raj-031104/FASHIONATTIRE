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
    def _attribute_scores(self, q: QuerySpec,
                          candidates: List[ImageRecord]) -> np.ndarray:
        """Attribute-match score per candidate, with PER-ATTRIBUTE normalization.

        Why not just average the raw confidences? They come from a softmax *within each
        axis*, so their natural scales differ (13 colors -> max ~0.3; 10 environments ->
        max ~0.4). Averaging raw values lets the higher-scaled axis silently dominate a
        multi-attribute query ("business attire INSIDE AN OFFICE"). Instead we min-max
        normalize each requested (axis, value) across the candidate pool first, so every
        requested attribute contributes on equal footing, then average.
        """
        # Attributes already consumed by a (colour, garment) binding are EXCLUDED here.
        # Reason: this signal is order-agnostic — "red tie + white shirt" and "white tie +
        # red shirt" both reduce to the colour set {red, white}, so counting those colours
        # again would pull colour-swapped queries back TOGETHER and dilute the
        # order-sensitive region_binding signal. Measured on the swap test: leaving them in
        # made overlap@10 worse on exactly the compositional query. Unbound attributes
        # (environment, style, unbound garments) still contribute normally.
        bound_colors = {b.color for b in q.bindings if b.is_bound()}
        bound_garments = {b.garment_type for b in q.bindings if b.is_bound()}

        pairs = []
        for axis, values in q.attributes.items():
            for v in values:
                if axis == "colors" and v in bound_colors:
                    continue
                if axis in ("upper_garment", "lower_garment", "accessories") and v in bound_garments:
                    continue
                pairs.append((axis, v))

        n = len(candidates)
        if not pairs or n == 0:
            # No unbound attributes left to match (e.g. "a bright yellow raincoat", where
            # both colour and garment are consumed by the binding). Signal is INACTIVE —
            # the caller drops it and redistributes its weight instead of letting a
            # constant-zero column silently absorb 20% of the budget.
            return np.zeros(n)

        hard = self.attr_cfg.get("mode", "soft") == "hard"
        thr = self.attr_cfg.get("present_threshold", 0.25)

        cols = []
        for axis, v in pairs:
            col = np.array([
                float(c.attributes.get(axis, {}).get(v, 0.0)) for c in candidates
            ])
            if hard:
                cols.append((col >= thr).astype(np.float64))
            else:
                cols.append(_minmax(col))
        return np.mean(np.stack(cols, axis=0), axis=0)

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

        # A query like "a red tie AND a white shirt" is a CONJUNCTION: satisfying only one
        # binding should not score like satisfying both. Plain mean gives too much credit
        # for a half-match; plain min is brittle (one missed segmentation zeroes the
        # image). Default 'hybrid' = 0.5*mean + 0.5*min keeps partial credit while
        # rewarding images that satisfy every binding.
        mean_s = float(np.mean(per_binding))
        min_s = float(np.min(per_binding))
        agg = self.region_cfg.get("aggregation", "hybrid")
        if agg == "min":
            return min_s
        if agg == "mean":
            return mean_s
        return 0.5 * mean_s + 0.5 * min_s

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
            "attribute_match": self._attribute_scores(query, candidates),
            "region_binding": np.array(
                [self._region_binding(query, c) for c in candidates]),
        }

        # which signals are active for THIS query? A signal that is constant across the
        # whole pool carries no ranking information, so it is dropped and its weight is
        # redistributed to the signals that actually discriminate.
        attr_raw = np.asarray(raw["attribute_match"], dtype=np.float64)
        active: Dict[str, bool] = {
            "global_clip": True,
            "caption_sim": raw["caption_sim"] is not None,
            "attribute_match": bool(query.attributes) and float(np.ptp(attr_raw)) > 1e-9,
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
