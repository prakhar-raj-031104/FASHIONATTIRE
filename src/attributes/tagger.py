"""Zero-shot attribute tagger built on FashionCLIP.

Turns FashionCLIP into a grounded, deterministic attribute classifier. For each attribute
axis (environment/style/garment/color) we build a *text bank*: every vocabulary value is
embedded through several prompt templates and averaged (prompt ensembling reduces
sensitivity to any single phrasing). An image (or region crop) embedding is then scored
against the bank and softmax-normalized into per-value confidences.

This gives us structured metadata that is:
  * grounded in pixels (not a caption an LLM might hallucinate),
  * reproducible (no sampling),
  * aligned 1:1 with the evaluation axes.

Text banks are precomputed ONCE at construction and reused for the whole corpus.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from . import vocab


def _softmax(x: np.ndarray, temp: float = 0.01) -> np.ndarray:
    z = x / temp
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class ZeroShotTagger:
    def __init__(self, clip, temperature: float = 0.01):
        """``clip`` is a FashionCLIP wrapper. ``temperature`` mirrors CLIP's logit scale
        (~0.01 -> multiply cosine sims by 100 before softmax)."""
        self.clip = clip
        self.temp = temperature

        # Image-level axis banks: axis -> (values, embedding matrix V x D).
        self.image_banks: Dict[str, Tuple[List[str], np.ndarray]] = {}
        for axis, values in vocab.IMAGE_AXES.items():
            templates = vocab.IMAGE_PROMPT_TEMPLATES[axis]
            self.image_banks[axis] = (values, self._build_bank(values, templates))

        # Region color bank (shared across all region types).
        self.color_bank = self._build_bank(vocab.COLORS, vocab.REGION_COLOR_TEMPLATES)

        # Region type banks keyed by SegFormer label (fine-grained typing of each crop).
        self.region_type_banks: Dict[str, Tuple[List[str], np.ndarray]] = {}
        for seg_label, cands in vocab.REGION_TYPE_CANDIDATES.items():
            self.region_type_banks[seg_label] = (cands, self._build_bank(
                cands, vocab.REGION_TYPE_TEMPLATES))

    # ------------------------------------------------------------------ #
    def _build_bank(self, values: List[str], templates: List[str]) -> np.ndarray:
        """Prompt-ensembled text embedding per value: mean over templates, renormalized."""
        vecs = []
        for v in values:
            prompts = [t.format(v) for t in templates]
            emb = self.clip.encode_text(prompts)          # (T, D), L2-normed
            mean = emb.mean(axis=0)
            n = np.linalg.norm(mean)
            vecs.append(mean / n if n > 0 else mean)
        return np.stack(vecs, axis=0).astype(np.float32)  # (V, D)

    @staticmethod
    def _score(emb: np.ndarray, bank: np.ndarray, values: List[str],
               temp: float) -> Dict[str, float]:
        sims = bank @ emb                                  # (V,)
        probs = _softmax(sims, temp)
        return {val: float(p) for val, p in zip(values, probs)}

    # ------------------------------------------------------------------ #
    def tag_image(self, image_embedding: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Structured image-level attributes: axis -> {value: confidence}."""
        out: Dict[str, Dict[str, float]] = {}
        for axis, (values, bank) in self.image_banks.items():
            out[axis] = self._score(image_embedding, bank, values, self.temp)
        return out

    def type_region(self, seg_label: str, crop_embedding: np.ndarray) -> Dict[str, float]:
        """Fine-grained garment type distribution for one region crop."""
        bank = self.region_type_banks.get(seg_label)
        if bank is None:
            return {}
        values, mat = bank
        return self._score(crop_embedding, mat, values, self.temp)

    def color_region(self, crop_embedding: np.ndarray) -> Dict[str, float]:
        """Color distribution for one region crop."""
        return self._score(crop_embedding, self.color_bank, vocab.COLORS, self.temp)
