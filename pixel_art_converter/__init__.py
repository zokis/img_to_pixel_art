"""Grid detection and one-colour-per-cell pixel-art conversion."""

from __future__ import annotations

from .core import Detection, DetectionError, detect
from .dither import map_to_palette
from .rebuild import two_stage_pack

__all__ = [
    "detect",
    "two_stage_pack",
    "map_to_palette",
    "Detection",
    "DetectionError",
]

__version__ = "1.1.0"
