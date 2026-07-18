"""Model factories — the single mapping from configuration to loaded models.

Both the indexer and the retriever construct models through these functions, so there is
exactly one place where ``configs/models.yaml`` becomes a live object. Each factory is
lazy, so a stage only pays for what it uses (the retriever never loads BLIP or SegFormer).
"""
from .factory import (
    build_clip,
    build_captioner,
    build_sentence_encoder,
    build_segmenter,
    build_reranker,
    build_tagger,
)

__all__ = [
    "build_clip",
    "build_captioner",
    "build_sentence_encoder",
    "build_segmenter",
    "build_reranker",
    "build_tagger",
]
