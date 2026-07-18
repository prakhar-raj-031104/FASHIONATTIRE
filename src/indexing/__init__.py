"""PART A — the Indexer: raw images -> searchable representations.

  dataset_loader.py  dataset-agnostic discovery (just needs a folder of images)
  indexer.py         orchestrates embedding, captioning, attribute tagging and garment
                     region decomposition, writing FAISS indexes + the SQLite metadata DB
"""
from .dataset_loader import discover_images, load_image
from .indexer import Indexer

__all__ = ["discover_images", "load_image", "Indexer"]
