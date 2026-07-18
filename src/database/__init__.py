"""Storage layer: vectors in FAISS, metadata in SQLite.

The split is deliberate and is what keeps the 1M-image story clean:
  vector_store.py  FAISS answers "which vectors are nearest" (ANN recall)
  metadata_db.py   SQLite answers "what is image 42, and which regions belong to it",
                   and enables attribute pre-filtering before/after the ANN search
"""
from .metadata_db import MetadataDB
from .vector_store import FaissIndex, VectorStore

__all__ = ["MetadataDB", "FaissIndex", "VectorStore"]
