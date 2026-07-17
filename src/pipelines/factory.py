"""Model factories — the single place that turns config into loaded models.

Both the indexer and the retriever build models through these functions, so there is
exactly one mapping from ``configs/models.yaml`` to a live object. Change a checkpoint in
YAML and every consumer picks it up. Each factory is lazy (only builds what a stage needs
— the retriever never loads BLIP or SegFormer).
"""
from __future__ import annotations

from ..models import (
    Captioner,
    CrossEncoderReranker,
    FashionCLIP,
    GarmentSegmenter,
    SentenceEncoder,
)
from ..attributes import ZeroShotTagger
from ..utils.config import Config


def build_clip(cfg: Config) -> FashionCLIP:
    m = cfg.models["fashion_clip"]
    return FashionCLIP(
        model_name=m["model"], device=cfg.device, dtype=cfg.models.get("dtype", "float32"),
        batch_size=m.get("batch_size", 64),
    )


def build_captioner(cfg: Config):
    m = cfg.models["captioner"]
    if not m.get("enabled", True):
        return None
    return Captioner(
        model_name=m["model"], device=cfg.device, dtype=cfg.models.get("dtype", "float32"),
        max_new_tokens=m.get("max_new_tokens", 40), batch_size=m.get("batch_size", 16),
    )


def build_sentence_encoder(cfg: Config) -> SentenceEncoder:
    m = cfg.models["sentence_encoder"]
    return SentenceEncoder(
        model_name=m["model"], device=cfg.device, batch_size=m.get("batch_size", 128)
    )


def build_segmenter(cfg: Config):
    m = cfg.models["segmenter"]
    if not m.get("enabled", True):
        return None
    return GarmentSegmenter(
        model_name=m["model"], device=cfg.device, dtype=cfg.models.get("dtype", "float32"),
        min_region_area_frac=m.get("min_region_area_frac", 0.005),
        max_regions_per_image=m.get("max_regions_per_image", 6),
    )


def build_reranker(cfg: Config):
    m = cfg.models["reranker"]
    if not m.get("enabled", False):
        return None
    return CrossEncoderReranker(model_name=m["model"], device=cfg.device)


def build_tagger(clip: FashionCLIP) -> ZeroShotTagger:
    return ZeroShotTagger(clip)
