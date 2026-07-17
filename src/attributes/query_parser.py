"""Natural-language query -> structured QuerySpec.

Deliberately rule-based (keyword + synonym + adjacency), not an LLM:
  * deterministic and instant — no GPU, no sampling, trivially unit-tested;
  * the schema is small and fixed, so a curated synonym map covers it robustly;
  * the LLM parser is documented as a drop-in upgrade in the writeup (pluggable here).

The critical output is ``bindings``: (color, garment) pairs recovered from adjacency,
e.g. "a red tie and a white shirt" -> [(red, tie), (white, shirt)]. These drive the
region-binding scorer that defeats CLIP's bag-of-words failure.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..utils.schema import GarmentBinding, QuerySpec
from . import vocab

# --- synonym / alias maps (query surface form -> canonical vocab value) ---------------
COLOR_ALIASES: Dict[str, str] = {
    "grey": "gray", "navy blue": "navy", "off-white": "white", "cream": "beige",
    "tan": "beige", "maroon": "red", "crimson": "red", "scarlet": "red",
    "turquoise": "blue", "sky blue": "blue", "olive": "green",
}

GARMENT_ALIASES: Dict[str, str] = {
    "tee": "t-shirt", "t shirt": "t-shirt", "tshirt": "t-shirt",
    "button-down": "shirt", "button down": "shirt", "button-up": "shirt",
    "dress shirt": "shirt", "suit": "blazer", "sport coat": "blazer",
    "pants": "trousers", "slacks": "trousers", "chinos": "trousers",
    "denim": "jeans", "hoody": "hoodie", "pullover": "sweater",
    "jumper": "sweater", "windbreaker": "jacket", "mac": "raincoat",
    "rain jacket": "raincoat", "necktie": "tie", "cap": "cap", "beanie": "hat",
    "handbag": "bag", "purse": "bag", "backpack": "bag", "shades": "sunglasses",
}

ENVIRONMENT_KEYWORDS: Dict[str, str] = {
    "office": "office", "workplace": "office", "boardroom": "office",
    "street": "urban street", "urban": "urban street", "city": "urban street",
    "sidewalk": "urban street", "downtown": "urban street",
    "park": "park", "garden": "park", "bench": "park",
    "home": "home interior", "living room": "home interior", "bedroom": "home interior",
    "indoor": "indoor studio", "studio": "indoor studio",
    "restaurant": "restaurant", "cafe": "restaurant", "mall": "shopping mall",
    "gym": "gym", "beach": "beach", "outdoor": "nature outdoors",
    "nature": "nature outdoors",
}

STYLE_KEYWORDS: Dict[str, str] = {
    "formal": "formal", "professional": "business", "business": "business",
    "office wear": "business", "corporate": "business", "casual": "casual",
    "weekend": "casual", "relaxed": "casual", "everyday": "casual",
    "streetwear": "streetwear", "sporty": "sporty", "athletic": "sporty",
    "gym wear": "sporty", "elegant": "elegant", "chic": "elegant",
    "winter": "winter", "summer": "summer", "loungewear": "loungewear",
}

# Color modifiers that may sit between a color and its garment ("bright yellow raincoat").
COLOR_MODIFIERS = {"bright", "dark", "light", "pale", "deep", "vivid", "pastel"}

# Which image axis a garment type belongs to (for per-axis attribute matching).
_UPPER = set(vocab.UPPER_GARMENT)
_LOWER = set(vocab.LOWER_GARMENT)
_ACC = set(vocab.ACCESSORIES)


def _garment_axis(gtype: str) -> str:
    if gtype in _LOWER:
        return "lower_garment"
    if gtype in _ACC:
        return "accessories"
    return "upper_garment"  # default bucket for tops/outerwear/dress


class QueryParser:
    def __init__(self):
        # Longest-first so multi-word phrases match before their sub-words.
        self._garment_terms = sorted(
            set(vocab.GARMENT_TYPES) | set(GARMENT_ALIASES),
            key=len, reverse=True,
        )
        self._color_terms = sorted(
            set(vocab.COLORS) | set(COLOR_ALIASES), key=len, reverse=True
        )

    # ------------------------------------------------------------------ #
    def _canon_color(self, tok: str) -> Optional[str]:
        tok = tok.lower()
        if tok in vocab.COLORS:
            return tok
        return COLOR_ALIASES.get(tok)

    def _canon_garment(self, phrase: str) -> Optional[str]:
        phrase = phrase.lower()
        if phrase in vocab.GARMENT_TYPES:
            return phrase
        return GARMENT_ALIASES.get(phrase)

    def _find_spans(self, text: str, terms: List[str]) -> List[tuple]:
        """Return (start_word_idx, canonical, surface) for each term occurrence."""
        words = text.split()
        spans = []
        joined = " ".join(words)
        for term in terms:
            for m in re.finditer(r"\b" + re.escape(term) + r"\b", joined):
                start_char = m.start()
                word_idx = joined[:start_char].count(" ")
                spans.append((word_idx, term))
        return spans

    def parse(self, query: str) -> QuerySpec:
        text = query.lower().strip()
        words = text.split()

        attributes: Dict[str, List[str]] = {}

        def add(axis: str, value: str):
            attributes.setdefault(axis, [])
            if value not in attributes[axis]:
                attributes[axis].append(value)

        # --- environment & style via keyword scan --------------------------
        for kw, canon in ENVIRONMENT_KEYWORDS.items():
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                add("environment", canon)
        for kw, canon in STYLE_KEYWORDS.items():
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                add("style", canon)

        # --- locate garment and color mentions -----------------------------
        garment_hits = self._find_spans(text, self._garment_terms)
        color_hits = self._find_spans(text, self._color_terms)

        # De-duplicate overlapping garment matches (keep the longest at each start).
        garment_by_idx: Dict[int, str] = {}
        for idx, term in sorted(garment_hits, key=lambda x: -len(x[1])):
            canon = self._canon_garment(term)
            if canon and idx not in garment_by_idx:
                garment_by_idx[idx] = canon
        color_by_idx: Dict[int, str] = {}
        for idx, term in color_hits:
            canon = self._canon_color(term)
            if canon:
                color_by_idx[idx] = canon

        # Record standalone attribute presence.
        for gtype in garment_by_idx.values():
            add(_garment_axis(gtype), gtype)
        for c in color_by_idx.values():
            add("colors", c)

        # --- bind each color to the nearest following garment --------------
        bindings: List[GarmentBinding] = []
        garment_idxs = sorted(garment_by_idx.keys())
        used_garments = set()
        for cidx in sorted(color_by_idx.keys()):
            color = color_by_idx[cidx]
            # nearest garment at or after the color, within a small window (skip modifiers)
            best = None
            for gidx in garment_idxs:
                if gidx < cidx:
                    continue
                gap_words = words[cidx + 1:gidx]
                if len(gap_words) <= 3 and all(
                    w in COLOR_MODIFIERS or w in ("a", "an", "the", "colored", "coloured")
                    or self._canon_color(w) for w in gap_words
                ):
                    best = gidx
                    break
            if best is not None and best not in used_garments:
                bindings.append(GarmentBinding(color=color,
                                               garment_type=garment_by_idx[best]))
                used_garments.add(best)
            else:
                # Unbound color — still a soft color constraint (already added above).
                bindings.append(GarmentBinding(color=color, garment_type=None))

        # Garments with no color get a type-only binding (helps region typing).
        for gidx, gtype in garment_by_idx.items():
            if gidx not in used_garments and not any(
                b.garment_type == gtype for b in bindings
            ):
                bindings.append(GarmentBinding(color=None, garment_type=gtype))

        return QuerySpec(raw=query, attributes=attributes, bindings=bindings)
