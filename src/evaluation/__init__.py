"""Evaluation harnesses.

  evaluate.py       runs the five official assignment queries, writes a JSON report and a
                    visual contact sheet, and computes precision@k when labels are supplied
  compositional.py  the colour-swap test: an objective, LABEL-FREE measurement of
                    colour-garment binding versus a vanilla CLIP baseline
"""
from .evaluate import EVAL_QUERIES, run_evaluation

__all__ = ["EVAL_QUERIES", "run_evaluation"]
