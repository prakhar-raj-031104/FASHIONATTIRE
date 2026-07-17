"""Dataset loading — deliberately dataset-agnostic.

The retriever must not care whether images came from Fashionpedia, DeepFashion, COCO or a
folder of phone photos. So the loader just walks a directory tree for image files and
assigns each a stable integer id (sorted by relative path, so ids are reproducible across
runs). Any annotations a dataset ships are optional enrichment, not a dependency.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def discover_images(root: Path | str) -> List[Tuple[int, str]]:
    """Return [(image_id, absolute_path)] sorted deterministically by relative path.

    Uses ``os.walk(followlinks=True)`` so a symlinked dataset directory (the common case
    when you keep the raw images outside the repo) is traversed — ``Path.rglob`` does not
    follow directory symlinks.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    paths: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for fn in filenames:
            if Path(fn).suffix.lower() in IMAGE_EXTS:
                paths.append(Path(dirpath) / fn)
    paths = sorted(paths, key=lambda p: str(p))
    return [(i, str(p.resolve())) for i, p in enumerate(paths)]


def load_image(path: str):
    """Load an image as RGB PIL (handles grayscale / RGBA / palette)."""
    from PIL import Image

    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img
