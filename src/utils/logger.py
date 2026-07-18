"""Centralised logging. Use ``get_logger(__name__)`` instead of ``print``.

Logs go to stdout (rich-formatted if ``rich`` is installed) and, when a log dir is
given, to ``outputs/logs/<name>.log`` so an indexing run leaves an auditable trail.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

_CONFIGURED: set[str] = set()


def get_logger(name: str, log_dir: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger; also writes to outputs/logs/<name>.log when log_dir is given."""
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger
    logger.setLevel(level)
    logger.propagate = False

    # Console handler — prefer rich for readable, coloured output.
    try:
        from rich.logging import RichHandler

        console = RichHandler(rich_tracebacks=True, show_path=False)
        console.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    except Exception:  # pragma: no cover - rich optional
        console = logging.StreamHandler()
        console.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
    logger.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name.split('.')[-1]}.log")
        fh.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(fh)

    _CONFIGURED.add(name)
    return logger
