"""Fallback grid fitting used when the primary estimators fail."""

from __future__ import annotations

import numpy as np

from .utils import comb_score, edge_profile, luminance


def fit_grid(rgba: np.ndarray, max_output: int = 128):
    """Return ``(step_x, step_y, cols, rows)`` for non-periodic images."""
    lum = luminance(rgba)
    height, width = lum.shape

    def axis(prof, length):
        best = (0.0, -1e9)
        for step in np.arange(2.0, min(64.0, length / 2.0), 0.25):
            score, _ = comb_score(prof, step)
            if score > best[1]:
                best = (float(step), score)
        if best[1] < 0.05:
            longest = max(height, width)
            target = min(max_output, 128)
            return length / max(1, round(length * target / longest))
        return best[0]

    step_x = axis(edge_profile(lum, 0), width)
    step_y = axis(edge_profile(lum, 1), height)
    cols = int(max(1, min(max_output, round(width / step_x))))
    rows = int(max(1, min(max_output, round(height / step_y))))
    return width / cols, height / rows, cols, rows
