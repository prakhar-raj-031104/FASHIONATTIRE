"""FAISS vector store — the approximate-nearest-neighbour side of the index.

We keep THREE independent indexes, all cosine (inner product on L2-normalized vectors):
  * global.faiss   : one whole-image FashionCLIP vector per image   (id = image_id)
  * caption.faiss  : one caption sentence vector per image           (id = image_id)
  * region.faiss   : one FashionCLIP vector per garment region       (id = region_id)

Each is an IndexIDMap over IndexFlatIP so we add/search with our own stable ids and never
rely on insertion order. Flat is exact and ideal at this scale; the writeup explains the
one-line swap to IVF/HNSW for 1M+ (see VectorStore.build docstring).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class FaissIndex:
    """A single named FAISS index with explicit int64 ids."""

    def __init__(self, dim: int, index=None):
        self.dim = dim
        self._faiss = __import__("faiss")
        if index is None:
            index = self._faiss.IndexIDMap2(self._faiss.IndexFlatIP(dim))
        self.index = index

    # --- build / update ------------------------------------------------- #
    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        ids = np.ascontiguousarray(ids.astype(np.int64))
        self.index.add_with_ids(vectors, ids)

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scores, ids), each (Q, k). Missing slots have id == -1."""
        if queries.ndim == 1:
            queries = queries[None, :]
        queries = np.ascontiguousarray(queries.astype(np.float32))
        k = min(k, max(self.index.ntotal, 1))
        scores, ids = self.index.search(queries, k)
        return scores, ids

    def reconstruct(self, ids: np.ndarray) -> np.ndarray:
        """Fetch stored vectors by id (IndexIDMap2 supports exact reconstruction).

        Used by the scorer so any signal can be computed exactly for a candidate even if
        that candidate was surfaced by a *different* index. Falls back to zeros for ids
        not present (id == -1 padding from search).
        """
        out = np.zeros((len(ids), self.dim), dtype=np.float32)
        for i, _id in enumerate(ids):
            if _id is None or int(_id) < 0:
                continue
            try:
                out[i] = self.index.reconstruct(int(_id))
            except Exception:
                pass
        return out

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    # --- persistence ---------------------------------------------------- #
    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path | str) -> "FaissIndex":
        import faiss

        index = faiss.read_index(str(path))
        return cls(dim=index.d, index=index)


class VectorStore:
    """Convenience holder for the three indexes with build/save/load helpers."""

    def __init__(self, global_index: Optional[FaissIndex] = None,
                 caption_index: Optional[FaissIndex] = None,
                 region_index: Optional[FaissIndex] = None):
        self.global_index = global_index
        self.caption_index = caption_index
        self.region_index = region_index

    @staticmethod
    def build(dim: int, index_type: str = "flat", nlist: int = 100) -> FaissIndex:
        """Factory for an empty index.

        index_type:
          "flat" -> exact IndexFlatIP (default; best for <~200k vectors).
          "ivf"  -> IVF-Flat (needs training); the drop-in for 1M+ scale. HNSW is another
                    option; both are one construction-line changes, nothing else moves
                    because ids and the search() contract are identical.
        """
        import faiss

        if index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            base = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index = faiss.IndexIDMap2(base)
        else:
            index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        return FaissIndex(dim=dim, index=index)

    def save(self, global_path, caption_path, region_path) -> None:
        if self.global_index:
            self.global_index.save(global_path)
        if self.caption_index:
            self.caption_index.save(caption_path)
        if self.region_index:
            self.region_index.save(region_path)

    @classmethod
    def load(cls, global_path, caption_path, region_path) -> "VectorStore":
        def _maybe(p):
            return FaissIndex.load(p) if Path(p).exists() else None

        return cls(
            global_index=_maybe(global_path),
            caption_index=_maybe(caption_path),
            region_index=_maybe(region_path),
        )
