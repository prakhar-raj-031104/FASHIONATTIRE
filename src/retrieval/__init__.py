"""PART B — the Retriever: natural-language query -> top-k images.

  scorer.py     HybridScorer — the four-signal ranking function (global CLIP, caption
                similarity, attribute match, compositional region binding) and their fusion
  retriever.py  Retriever — two-stage search: ANN recall from FAISS, then feature-rich
                rerank of the candidate pool
"""
from .scorer import HybridScorer, ScoredImage
from .retriever import Retriever

__all__ = ["HybridScorer", "ScoredImage", "Retriever"]
