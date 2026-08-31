"""Self-similarity / shift-dissimilarity grid estimator.

For candidate shifts ``d`` it measures how dissimilar the image is from a copy
of itself translated by ``d`` pixels along one axis, using raw gradient,
smoothed gradient and Laplacian maps.  A true grid period produces deep local
minima of the dissimilarity at multiples of the cell size; a comb statistic
over those minima yields the vote, phase and dispersion.  Runs on a pixel
budget.
"""

from __future__ import annotations

import numpy as np

from .utils import (
    Estimate,
    drift_aware_count,
    gaussian_blur,
    luminance,
    subsample_for_budget,
)

NORMAL_MIN, NORMAL_MAX = 2.0, 24.0
PIXEL_BUDGET = 220_000


def _maps(lum: np.ndarray):
    gx = np.abs(np.gradient(lum, axis=1))
    gy = np.abs(np.gradient(lum, axis=0))
    raw = gx + gy
    sm = gaussian_blur(raw, 1.0)
    lap = np.abs(gaussian_blur(lum, 1.2) * 2.0 - np.roll(lum, 1, 0) - np.roll(lum, -1, 0))
    return [raw, sm, lap]


def _dissim_curve(m: np.ndarray, axis: int, dmax: int) -> np.ndarray:
    out = np.zeros(dmax + 1)
    for d in range(1, dmax + 1):
        if axis == 1:  # shift along columns
            a = m[:, d:]
            b = m[:, :-d]
        else:
            a = m[d:, :]
            b = m[:-d, :]
        if a.size == 0:
            out[d] = out[d - 1]
        else:
            out[d] = np.mean(np.abs(a - b))
    return out


def _period_from_curve(curve: np.ndarray, smin: float, smax: float):
    dmax = len(curve) - 1
    smax = min(smax, dmax)
    if smax <= smin:
        return 0.0, 0.0, 0.0
    inv = curve.max() - curve  # minima -> maxima
    grid = np.arange(smin, smax + 1e-9, 0.05)
    base = np.mean(inv[1:]) + 1e-9
    best = (0.0, -1e9)
    for s in grid:
        k = np.arange(1, int(dmax / s) + 1)
        pos = (k * s).astype(int)
        pos = pos[pos <= dmax]
        if pos.size == 0:
            continue
        anti = np.clip((pos - int(round(s / 2))), 1, dmax)
        score = (inv[pos].mean() - inv[anti].mean()) / base
        if score > best[1]:
            best = (float(s), float(score))
    s = best[0]
    if s <= 0:
        return 0.0, 0.0, 0.0
    return s, max(0.0, min(1.0, best[1])), 0.0


def estimate(rgba: np.ndarray, heavy: bool = False) -> Estimate:
    small = subsample_for_budget(rgba, PIXEL_BUDGET)
    fx = rgba.shape[1] / small.shape[1]
    fy = rgba.shape[0] / small.shape[0]
    lum = luminance(small)
    h, w = lum.shape
    H, W = rgba.shape[:2]

    dmax = int(min(NORMAL_MAX / min(fx, fy) + 4, min(h, w) // 2))
    dmax = max(dmax, int(NORMAL_MIN) + 2)

    votes_x, votes_y, sx_list, sy_list = [], [], [], []
    for m in _maps(lum):
        cx = _dissim_curve(m, axis=1, dmax=dmax)
        cy = _dissim_curve(m, axis=0, dmax=dmax)
        s, sc, _ = _period_from_curve(cx, NORMAL_MIN / fx, NORMAL_MAX / fx + 2)
        if s > 0:
            sx_list.append(s * fx)
            votes_x.append(sc)
        s, sc, _ = _period_from_curve(cy, NORMAL_MIN / fy, NORMAL_MAX / fy + 2)
        if s > 0:
            sy_list.append(s * fy)
            votes_y.append(sc)

    sx = float(np.median(sx_list)) if sx_list else W / 2.0
    sy = float(np.median(sy_list)) if sy_list else H / 2.0
    scx = float(np.mean(votes_x)) if votes_x else 0.0
    scy = float(np.mean(votes_y)) if votes_y else 0.0
    if sx_list:
        scx *= 1.0 - min(0.5, np.std(sx_list) / (np.mean(sx_list) + 1e-9))
    if sy_list:
        scy *= 1.0 - min(0.5, np.std(sy_list) / (np.mean(sy_list) + 1e-9))

    cols = drift_aware_count(W, sx)
    rows = drift_aware_count(H, sy)
    return Estimate(
        name="selfsim",
        step_x=W / cols,
        step_y=H / rows,
        cols=cols,
        rows=rows,
        score_x=max(0.0, scx),
        score_y=max(0.0, scy),
        candidates_x=sx_list,
        candidates_y=sy_list,
    )
