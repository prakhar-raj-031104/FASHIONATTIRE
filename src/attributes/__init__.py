"""Structured fashion attributes: vocabulary, image tagging, and query parsing.

This package turns free-form pixels and free-form text into the SAME structured
vocabulary, which is what makes attribute matching and colour-garment binding possible:

  vocab.py         controlled vocabulary (environment / style / garment / colour) and the
                   prompt templates used for zero-shot classification
  tagger.py        image (or region crop) -> {axis: {value: confidence}} via FashionCLIP
  query_parser.py  natural-language query -> QuerySpec (attributes + (colour, garment) bindings)
"""
from . import vocab
from .tagger import ZeroShotTagger
from .query_parser import QueryParser

__all__ = ["vocab", "ZeroShotTagger", "QueryParser"]
