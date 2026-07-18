"""Controlled fashion vocabulary + prompt templates.

Why a fixed vocabulary instead of a free-form LLM?
  * It is grounded in the image (CLIP scores pixels against these prompts) rather than
    grounded in a caption an LLM might hallucinate.
  * It is deterministic and reproducible — no sampling, no model drift.
  * It maps 1:1 onto the assignment's evaluation axes (color + garment + environment
    + style), so every eval query decomposes cleanly onto it.

The vocabulary is intentionally editable — extending to new cities/weather (see the
writeup's "future work") is just adding entries here + prompt templates below.
"""
from __future__ import annotations

from typing import Dict, List

# Attribute axes. Each axis -> list of vocabulary values that CLIP will score
# an image (or region crop) against via zero-shot classification.
ENVIRONMENT: List[str] = [
    "office", "urban street", "park", "home interior", "restaurant",
    "shopping mall", "gym", "beach", "indoor studio", "nature outdoors",
]

STYLE: List[str] = [
    "formal", "business", "casual", "streetwear", "sporty",
    "elegant", "winter", "summer", "loungewear",
]

UPPER_GARMENT: List[str] = [
    "t-shirt", "shirt", "blouse", "hoodie", "sweater", "blazer",
    "jacket", "coat", "raincoat", "tank top", "dress",
]

LOWER_GARMENT: List[str] = [
    "jeans", "trousers", "shorts", "skirt", "leggings",
]

OUTERWEAR: List[str] = [
    "coat", "jacket", "blazer", "raincoat", "cardigan", "parka",
]

ACCESSORIES: List[str] = [
    "tie", "scarf", "hat", "cap", "bag", "sunglasses", "watch", "belt",
]

COLORS: List[str] = [
    "red", "orange", "yellow", "green", "blue", "navy", "purple",
    "pink", "brown", "black", "white", "gray", "beige",
]

GARMENT_TYPES: List[str] = sorted(set(
    UPPER_GARMENT + LOWER_GARMENT + OUTERWEAR + ACCESSORIES
))

# Axes exposed to the image-level tagger. (Regions are typed/colored separately.)
IMAGE_AXES: Dict[str, List[str]] = {
    "environment": ENVIRONMENT,
    "style": STYLE,
    "upper_garment": UPPER_GARMENT,
    "lower_garment": LOWER_GARMENT,
    "accessories": ACCESSORIES,
    "colors": COLORS,
}

# SegFormer (mattmdjaga/segformer_b2_clothes, ATR 18-class) label id -> our
# normalized garment vocabulary. Non-garment classes (skin/hair/face/legs/arms/
# background) map to None and are ignored during region extraction.
SEGFORMER_ID2LABEL: Dict[int, str] = {
    0: "background", 1: "hat", 2: "hair", 3: "sunglasses", 4: "upper-clothes",
    5: "skirt", 6: "pants", 7: "dress", 8: "belt", 9: "left-shoe", 10: "right-shoe",
    11: "face", 12: "left-leg", 13: "right-leg", 14: "left-arm", 15: "right-arm",
    16: "bag", 17: "scarf",
}

# Which SegFormer labels are actual garments/accessories worth a region, and the
# default garment_type we assign (refined later by CLIP zero-shot on the crop).
SEGFORMER_GARMENT_LABELS: Dict[str, str] = {
    "hat": "hat",
    "sunglasses": "sunglasses",
    "upper-clothes": "shirt",     # refined to shirt/t-shirt/hoodie/... by CLIP on crop
    "skirt": "skirt",
    "pants": "trousers",
    "dress": "dress",
    "belt": "belt",
    "left-shoe": "shoe",
    "right-shoe": "shoe",
    "bag": "bag",
    "scarf": "scarf",
}

# For a segmented "upper-clothes" / "pants" region, which vocab subset should CLIP pick
# the fine-grained type from. Keeps zero-shot typing tight and accurate.
REGION_TYPE_CANDIDATES: Dict[str, List[str]] = {
    "upper-clothes": ["t-shirt", "shirt", "blouse", "hoodie", "sweater",
                      "blazer", "jacket", "coat", "raincoat", "tank top"],
    "pants": ["jeans", "trousers", "shorts", "leggings"],
    "skirt": ["skirt"],
    "dress": ["dress"],
    "hat": ["hat", "cap"],
    "scarf": ["scarf", "tie"],   # SegFormer often labels ties as scarf
    "bag": ["bag"],
    "sunglasses": ["sunglasses"],
    "belt": ["belt", "tie"],     # thin vertical belt/tie confusion — let CLIP decide
    "left-shoe": ["shoe"],
    "right-shoe": ["shoe"],
}

# Prompt templates for CLIP zero-shot classification (prompt ensembling: average
# the text embeddings over several templates to reduce prompt sensitivity).
IMAGE_PROMPT_TEMPLATES: Dict[str, List[str]] = {
    "environment": [
        "a photo of a person in {}",
        "a photo taken in {}",
        "someone standing in {}",
    ],
    "style": [
        "a photo of someone wearing {} clothing",
        "a {} outfit",
        "{} fashion style",
    ],
    "upper_garment": [
        "a photo of a person wearing a {}",
        "someone in a {}",
        "a {}",
    ],
    "lower_garment": [
        "a photo of a person wearing {}",
        "someone wearing {}",
        "a pair of {}",
    ],
    "accessories": [
        "a photo of a person wearing a {}",
        "someone with a {}",
        "a {} accessory",
    ],
    "colors": [
        "a photo of {} clothing",
        "a person wearing {} garments",
        "an outfit that is {} colored",
    ],
}

REGION_TYPE_TEMPLATES: List[str] = [
    "a photo of a {}",
    "a {} garment",
    "close-up of a {}",
]
REGION_COLOR_TEMPLATES: List[str] = [
    "a {} colored garment",
    "a {} piece of clothing",
    "this garment is {}",
]


def color_prompts() -> List[str]:
    return COLORS
