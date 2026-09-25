"""The fake store world: products, shoppers, prices and simulated behaviour."""

from .config import PRESETS, Rules, WorldConfig, get_config
from .generate import build_world, generate_world

__all__ = ["PRESETS", "Rules", "WorldConfig", "build_world", "generate_world", "get_config"]
