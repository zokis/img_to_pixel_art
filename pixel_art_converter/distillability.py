"""Distillability search for the heavy arbiter.

"Distillability" measures how well the image collapses onto a grid: partition
by the candidate ``(cols, rows)``, replace every cell with its mean, and see
how small the residual is and how few effective colours remain.  A high score
means the grid is a good explanation of the image.
"""

from __future__ import annotations

import numpy as np

from .utils import luminance


def _block_reduce_mean(a: np.ndarray, cols: int, rows: int) -> np.ndarray:
    h, w = a.shape[:2]
    ix = np.floor(np.arange(w) * cols / w).astype(int)
    iy = np.floor(np.arange(h) * rows / h).astype(int)
    ix = np.clip(ix, 0, cols - 1)
    iy = np.clip(iy, 0, rows - 1)
    cell = iy[:, None] * cols + ix[None, :]
    flat = cell.reshape(-1)
    vals = a.reshape(h * w, -1).astype(np.float64)
    ncell = cols * rows
    sums = np.zeros((ncell, vals.shape[1]))
    np.add.at(sums, flat, vals)
    cnt = np.bincount(flat, minlength=ncell).astype(np.float64)
    cnt[cnt == 0] = 1.0
    means = sums / cnt[:, None]
    return means[cell]  # broadcast back to (h, w, C)


def distillability(rgba: np.ndarray, cols: int, rows: int, lum: np.ndarray | None = None) -> float:
    """Return a normalised score in ``[0, 1]`` (0 when the grid is degenerate)."""
    h, w = rgba.shape[:2]
    if cols < 1 or rows < 1 or cols >= w or rows >= h:
        return 0.0
    lum = luminance(rgba) if lum is None else lum
    approx = _block_reduce_mean(lum[..., None], cols, rows)[..., 0]
    resid = np.abs(lum - approx)
    rms = np.sqrt(np.mean(resid**2))
    spread = np.std(lum) + 1e-6
    fidelity = 1.0 - min(1.0, rms / spread)

    # simplicity: fraction of the cell budget actually used by distinct colours
    rgb_small = _block_reduce_mean(rgba[..., :3], cols, rows)
    q = (rgb_small.reshape(-1, 3) / 16).astype(int)
    distinct = len({tuple(v) for v in q})
    simplicity = 1.0 - min(1.0, distinct / float(cols * rows))

    return float(np.clip(0.75 * fidelity + 0.25 * simplicity, 0.0, 1.0))


def axis_distillability(rgba: np.ndarray, axis: int, count: int, other_count: int,
                        lum: np.ndarray | None = None) -> float:
    if axis == 0:
        return distillability(rgba, count, max(1, other_count), lum=lum)
    return distillability(rgba, max(1, other_count), count, lum=lum)
