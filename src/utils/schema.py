"""Typed data contracts passed between pipeline stages.

Design principle: every stage receives a record and returns an enriched record, so the
pipeline is composable and each stage is unit-testable in isolation. Nothing downstream
touches raw HuggingFace tensors — only these dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np


@dataclass
class RegionRecord:
    """One detected garment region (crop) within an image.

    Region-level records are the mechanism that gives us COMPOSITIONALITY: each garment
    is embedded and tagged independently, so a query can require that *this* color binds
    to *this* garment type.
    """

    region_id: int                       # unique across the whole corpus
    image_id: int
    garment_label: str                   # segmenter class, e.g. "upper-clothes", "pants"
    garment_type: str                    # normalized vocab type, e.g. "shirt", "trousers"
    bbox: List[int]                      # [x0, y0, x1, y1] in original image pixels
    area_frac: float                     # region area / image area
    colors: Dict[str, float] = field(default_factory=dict)   # color -> confidence
    type_scores: Dict[str, float] = field(default_factory=dict)  # garment type -> conf
    embedding: Optional[np.ndarray] = None  # FashionCLIP embedding of the crop (L2-normed)

    def top_color(self) -> Optional[str]:
        return max(self.colors, key=self.colors.get) if self.colors else None


@dataclass
class ImageRecord:
    """Everything we know about one database image after indexing."""

    image_id: int
    image_path: str
    caption: str = ""
    # Structured, image-level attributes from the CLIP zero-shot tagger. Each maps a
    # vocabulary value -> confidence in [0, 1], e.g. attributes["environment"]["office"].
    attributes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    clip_embedding: Optional[np.ndarray] = None       # whole-image FashionCLIP (L2-normed)
    caption_embedding: Optional[np.ndarray] = None    # sentence embedding of caption
    regions: List[RegionRecord] = field(default_factory=list)

    def top_attribute(self, axis: str) -> Optional[str]:
        vals = self.attributes.get(axis)
        return max(vals, key=vals.get) if vals else None

    def to_metadata(self) -> Dict:
        """JSON-serialisable metadata (no numpy arrays) for the SQLite store."""
        d = {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "caption": self.caption,
            "attributes": self.attributes,
            "regions": [
                {
                    "region_id": r.region_id,
                    "garment_label": r.garment_label,
                    "garment_type": r.garment_type,
                    "bbox": r.bbox,
                    "area_frac": r.area_frac,
                    "colors": r.colors,
                    "type_scores": r.type_scores,
                }
                for r in self.regions
            ],
        }
        return d


@dataclass
class GarmentBinding:
    """A (color, garment) constraint parsed out of a query.

    e.g. "red tie and white shirt" -> [GarmentBinding("red", "tie"),
    GarmentBinding("white", "shirt")]. This is the structured object the region-binding
    scorer consumes.
    """

    color: Optional[str] = None
    garment_type: Optional[str] = None

    def is_bound(self) -> bool:
        return self.color is not None and self.garment_type is not None


@dataclass
class QuerySpec:
    """Structured decomposition of a natural-language query.

    ``raw`` is always kept so dense signals (global CLIP, caption sim) can use the full
    sentence, while the structured fields drive attribute-match and region-binding.
    """

    raw: str
    attributes: Dict[str, List[str]] = field(default_factory=dict)  # axis -> [values]
    bindings: List[GarmentBinding] = field(default_factory=list)

    def has_bindings(self) -> bool:
        return any(b.is_bound() for b in self.bindings)

    def as_dict(self) -> Dict:
        d = asdict(self)
        return d
