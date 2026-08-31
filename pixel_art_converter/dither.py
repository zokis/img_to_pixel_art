"""Palette mapping with optional dithering.

``map_to_palette(rgb, palette, method)`` snaps every pixel of an ``(H, W, 3)``
image to the nearest colour of *palette*.  ``method`` is one of:

* ``"none"``   -- plain nearest-colour quantisation;
* ``"ordered"`` -- a 4x4 Bayer threshold matrix perturbs the pixel before the
  nearest-colour lookup (stable, no error propagation);
* ``"floyd"``  -- Floyd-Steinberg error diffusion.

The image handled here is already the small pixel-art result, so the
Floyd-Steinberg Python loop stays cheap.
"""

from __future__ import annotations

import numpy as np

METHODS = ("none", "ordered", "floyd")

# 4x4 Bayer matrix, normalised to roughly [-0.5, 0.5)
_BAYER4 = (np.array(
    [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float64
) + 0.5) / 16.0 - 0.5


def _nearest(pixels: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """Nearest-palette index for every row of *pixels*, in memory-bounded blocks."""
    out = np.empty(len(pixels), dtype=np.int64)
    p2 = (pal ** 2).sum(1)
    for s in range(0, len(pixels), 1_000_000):
        blk = pixels[s : s + 1_000_000]
        d = -2.0 * (blk @ pal.T) + p2[None, :]
        out[s : s + len(blk)] = d.argmin(1)
    return out


def _palette_spacing(pal: np.ndarray) -> float:
    """Mean nearest-neighbour distance between palette entries (dither amplitude)."""
    if len(pal) < 2:
        return 0.0
    d = np.sqrt(((pal[:, None, :] - pal[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(np.mean(d.min(1)))


def _ordered(img: np.ndarray, pal: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    amp = _palette_spacing(pal)
    tile = np.tile(_BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w]
    perturbed = img + tile[..., None] * amp
    return _nearest(perturbed.reshape(-1, 3), pal)


def _floyd_steinberg(img: np.ndarray, pal: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    work = img.astype(np.float64).copy()
    p2 = (pal ** 2).sum(1)
    idx = np.empty((h, w), dtype=np.int64)
    for y in range(h):
        for x in range(w):
            old = work[y, x]
            j = int(np.argmin(-2.0 * (pal @ old) + p2))
            idx[y, x] = j
            err = old - pal[j]
            if x + 1 < w:
                work[y, x + 1] += err * (7.0 / 16.0)
            if y + 1 < h:
                if x > 0:
                    work[y + 1, x - 1] += err * (3.0 / 16.0)
                work[y + 1, x] += err * (5.0 / 16.0)
                if x + 1 < w:
                    work[y + 1, x + 1] += err * (1.0 / 16.0)
    return idx.reshape(-1)


def map_to_palette(rgb: np.ndarray, palette: np.ndarray, method: str = "none") -> np.ndarray:
    """Map *rgb* ``(H, W, 3)`` to *palette* ``(K, 3)``; return ``(H, W, 3)`` uint8."""
    img = np.asarray(rgb, dtype=np.float64)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("map_to_palette expects an (H, W, 3) image")
    pal = np.asarray(palette, dtype=np.float64).reshape(-1, 3)
    if len(pal) == 0:
        raise ValueError("palette is empty")
    if method not in METHODS:
        raise ValueError(f"unknown dither method {method!r}; use one of {METHODS}")

    if method == "floyd":
        idx = _floyd_steinberg(img, pal)
    elif method == "ordered":
        idx = _ordered(img, pal)
    else:
        idx = _nearest(img.reshape(-1, 3), pal)

    out = pal[idx].reshape(img.shape)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
