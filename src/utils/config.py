"""Configuration loading and device resolution.

All tunable knobs live in ``configs/*.yaml``. Business logic never hardcodes a model
name, path, or weight — it reads them from here. This keeps ML logic decoupled from
concrete model choices (swap FashionCLIP -> generic CLIP by editing one YAML line).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

# Repo root = two levels up from this file (src/utils/config.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


@dataclass
class Config:
    """Merged, resolved view over the three YAML config files.

    Attributes are plain dicts mirroring the YAML structure. Paths are resolved to
    absolute ``Path`` objects so downstream code never guesses the working directory.
    """

    models: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, Any] = field(default_factory=dict)
    retrieval: Dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"
    root: Path = REPO_ROOT

    def path(self, *keys: str) -> Path:
        """Resolve a nested path key (e.g. ``cfg.path("index", "faiss_global")``)."""
        node: Any = self.paths
        for k in keys:
            node = node[k]
        return (self.root / node).resolve()

    def ensure_dirs(self) -> None:
        """Create the output/index/cache directories if they do not exist."""
        for key in (
            ("data", "cache"),
            ("index", "dir"),
            ("outputs", "dir"),
            ("outputs", "logs"),
            ("outputs", "results"),
        ):
            self.path(*key).mkdir(parents=True, exist_ok=True)


def resolve_device(requested: str = "auto") -> str:
    """Resolve ``auto`` to ``cuda`` when a GPU is present, otherwise ``cpu``.

    Kept import-light: torch is imported lazily so config can be read without torch.
    """
    if requested and requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_config(config_dir: Path | str | None = None) -> Config:
    """Load and merge all YAML configs, resolving the compute device.

    On CPU we force ``dtype: float32`` regardless of the YAML (fp16 matmul on CPU is
    slow/unsupported for these models).
    """
    cdir = Path(config_dir) if config_dir else CONFIG_DIR

    def _read(name: str) -> Dict[str, Any]:
        p = cdir / name
        if not p.exists():
            raise FileNotFoundError(f"Missing config file: {p}")
        with open(p, "r") as fh:
            return yaml.safe_load(fh) or {}

    models = _read("models.yaml")
    paths = _read("paths.yaml")
    retrieval = _read("retrieval.yaml")

    device = resolve_device(models.get("device", "auto"))
    if device == "cpu":
        models["dtype"] = "float32"

    # Allow HF cache redirection into the repo's data/cache for reproducibility.
    cache_dir = (REPO_ROOT / paths["data"]["cache"]).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir / "hf"))

    return Config(models=models, paths=paths, retrieval=retrieval, device=device)
