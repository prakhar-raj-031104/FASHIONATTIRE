"""Retriever — end-to-end query -> top-k images (Part B).

Two-stage retrieval (the standard scalable pattern):
  Stage 1 (recall): pull a candidate POOL from FAISS via ANN — union of the top-N from the
                    global CLIP index and the caption index. Cheap, sublinear at scale.
  Stage 2 (precision): re-score that small pool with the full HybridScorer (attribute +
                    region-binding signals that are too expensive to index directly).
                    Optionally apply a cross-encoder over captions as a final tie-breaker.

This is exactly how web-scale retrieval works (ANN recall -> feature-rich rerank), so the
logic is unchanged whether the corpus is 1k or 1M images — only the FAISS index type
changes (flat -> IVF/HNSW).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..attributes import QueryParser
from ..database import MetadataDB, VectorStore
from ..pipelines import build_clip, build_reranker, build_sentence_encoder
from ..utils.config import Config
from ..utils.logger import get_logger
from .scorer import HybridScorer, ScoredImage


class Retriever:
    """End-to-end query -> top-k search over a previously built index."""

    def __init__(self, cfg: Config):
        """Load the query-side models and open the FAISS indexes + metadata DB.

        Only the lightweight query-side models are built (CLIP text tower, sentence
        encoder, optional reranker) — BLIP and SegFormer are index-time only.
        """
        self.cfg = cfg
        self.log = get_logger("retriever", cfg.path("outputs", "logs"))

        # Query side is lightweight: only CLIP (text) + sentence encoder (+ optional rerank).
        self.clip = build_clip(cfg)
        self.sentence = build_sentence_encoder(cfg)
        self.reranker = build_reranker(cfg)
        self.parser = QueryParser()
        self.scorer = HybridScorer(cfg.retrieval)

        self.db = MetadataDB(cfg.path("index", "metadata_db"))
        self.store = VectorStore.load(
            cfg.path("index", "faiss_global"),
            cfg.path("index", "faiss_caption"),
            cfg.path("index", "faiss_region"),
        )
        if self.store.global_index is None:
            raise RuntimeError("No global FAISS index found — run indexing first.")

        self.pool = cfg.retrieval["search"].get("candidate_pool", 200)
        self.default_k = cfg.retrieval["search"].get("top_k", 10)

    def _candidate_ids(self, q_clip: np.ndarray, q_sent: np.ndarray) -> List[int]:
        ids: set[int] = set()
        _, gids = self.store.global_index.search(q_clip, self.pool)
        ids.update(int(i) for i in gids[0] if i >= 0)
        if self.store.caption_index is not None:
            _, cids = self.store.caption_index.search(q_sent, self.pool)
            ids.update(int(i) for i in cids[0] if i >= 0)
        return sorted(ids)

    def retrieve(self, query: str, k: Optional[int] = None,
                 explain: bool = False) -> List[ScoredImage]:
        """Return the top-k images for a natural-language query.

        Args:
            query: e.g. "a red tie and a white shirt in a formal setting".
            k: number of results (defaults to configs/retrieval.yaml search.top_k).
            explain: log how the query was parsed (attributes + bindings).

        Returns:
            Ranked ScoredImage list, each carrying its per-signal score breakdown.
        """
        k = k or self.default_k
        spec = self.parser.parse(query)
        if explain:
            self.log.info("Parsed query: attributes=%s bindings=%s",
                          spec.attributes,
                          [(b.color, b.garment_type) for b in spec.bindings])

        q_clip = self.clip.encode_text([query])[0]
        q_sent = self.sentence.encode([query])[0]

        cand_ids = self._candidate_ids(q_clip, q_sent)
        if not cand_ids:
            return []

        recs_map = self.db.get_images(cand_ids)
        candidates = [recs_map[i] for i in cand_ids if i in recs_map]

        ids_arr = np.array([c.image_id for c in candidates], dtype=np.int64)
        global_vecs = self.store.global_index.reconstruct(ids_arr)
        caption_vecs = (self.store.caption_index.reconstruct(ids_arr)
                        if self.store.caption_index is not None else None)

        scored = self.scorer.score(spec, q_clip, q_sent, candidates,
                                   global_vecs, caption_vecs)

        if self.reranker is not None:
            scored = self._apply_reranker(query, scored)

        return scored[:k]

    def _apply_reranker(self, query: str, scored: List[ScoredImage]) -> List[ScoredImage]:
        """Blend a cross-encoder caption relevance into the top-N ordering."""
        top_n = self.cfg.models["reranker"].get("rerank_top_n", 30)
        head = scored[:top_n]
        captions = [s.caption or "" for s in head]
        ce = self.reranker.score(query, captions)
        ce_norm = (ce - ce.min()) / (np.ptp(ce) + 1e-8) if len(ce) else ce
        for s, extra in zip(head, ce_norm):
            s.signals["cross_encoder"] = float(extra)
            s.score = 0.7 * s.score + 0.3 * float(extra)
        head.sort(key=lambda s: s.score, reverse=True)
        return head + scored[top_n:]

    def close(self) -> None:
        """Close the metadata DB connection."""
        self.db.close()
