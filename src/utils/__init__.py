"""Cross-cutting utilities: configuration, logging, and the typed data contracts.

  config.py  YAML config loading + CPU/GPU device resolution (no constant is hardcoded
             in business logic)
  logger.py  centralised logging (use get_logger, never print)
  schema.py  ImageRecord / RegionRecord / QuerySpec — the objects every pipeline stage
             passes, which keeps stages composable and independently testable
"""
from .config import load_config, resolve_device, Config
from .logger import get_logger
from .schema import ImageRecord, RegionRecord, QuerySpec, GarmentBinding

__all__ = [
    "load_config",
    "resolve_device",
    "Config",
    "get_logger",
    "ImageRecord",
    "RegionRecord",
    "QuerySpec",
    "GarmentBinding",
]
