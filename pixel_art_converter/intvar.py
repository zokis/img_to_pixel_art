"""Internal-variance phase contrast for the heavy arbiter.

For a candidate step the axis is partitioned into cells at the best and the
worst phase; the evidence is how much lower the mean within-cell variance is at
the best phase than at the worst.
"""

from __future__ import annotations

import numpy as np

from .utils import luminance


def _cell_var_1d(sig_means: np.ndarray, step: float, phase: float) -> float:
    n = len(sig_means)
    edges = np.arange(phase, n + step, step)
    edges = np.clip(np.round(edges).astype(int), 0, n)
    edges = np.unique(edges)
    if len(edges) < 3:
        return float(np.var(sig_means))
    vs = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a >= 1:
            vs.append(np.var(sig_means[a:b]))
    return float(np.mean(vs)) if vs else float(np.var(sig_means))


def evidence(rgba: np.ndarray, axis: int, step: float, lum: np.ndarray | None = None) -> float:
    """1-D phase-contrast evidence, normalised to ``[0, 1]``."""
    if step < 2:
        return 0.0
    lum = luminance(rgba) if lum is None else lum
    line = lum.mean(axis=0) if axis == 0 else lum.mean(axis=1)
    n = len(line)
    if step >= n / 2:
        return 0.0
    phases = np.linspace(0, step, 8, endpoint=False)
    vals = [_cell_var_1d(line, step, p) for p in phases]
    best, worst = min(vals), max(vals)
    total = np.var(line) + 1e-9
    return float(np.clip((worst - best) / total, 0.0, 1.0))
