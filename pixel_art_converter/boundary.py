"""Boundary-comb grid estimator (formerly the "runlength" estimator).

Locates coherent boundaries (local maxima of the edge profile above a noise
floor), takes the distances between successive boundaries and multiple lags,
and picks the period that best explains those distances across a few phase
tilings.  The returned score is the fraction of boundary spacing explained.
"""

from __future__ import annotations

import numpy as np

from .utils import Estimate, drift_aware_count, edge_profile, luminance

NORMAL_MIN, NORMAL_MAX = 2.0, 24.0

# --- pinned constants (kept explicit so tuning is reviewable) --------------- #
BOUNDARY_THRESH_STD = 0.5      # peak must exceed mean + this * std
GAP_MULTIPLE_TOL = 0.18       # |ratio - round(ratio)| below this counts as "on grid"
GAP_MATCH_BONUS = 0.5         # bonus weight when the step itself is a common gap


def _boundaries(prof: np.ndarray) -> np.ndarray:
    """Indices of coherent local maxima of *prof* above a noise floor."""
    if not np.any(prof):
        return np.array([], dtype=int)
    thr = prof.mean() + BOUNDARY_THRESH_STD * prof.std()
    left = prof[1:-1] >= prof[:-2]
    right = prof[1:-1] > prof[2:]
    high = prof[1:-1] >= thr
    idx = np.nonzero(left & right & high)[0] + 1
    return idx.astype(int)


def _period_from_boundaries(bnd: np.ndarray, length: int, smin: float, smax: float):
    if bnd.size < 3:
        return 0.0, 0.0
    gaps = np.diff(bnd)
    gaps = gaps[(gaps >= 1) & (gaps <= smax * 3)]
    if gaps.size == 0:
        return 0.0, 0.0

    smax = min(smax, length / 2.0)
    grid = np.arange(smin, smax + 1e-9, 0.05)
    best_s, best_score = 0.0, 0.0
    for s in grid:
        # how well every gap sits on an integer multiple of s
        ratio = gaps / s
        err = np.abs(ratio - np.round(ratio))
        near = err < GAP_MULTIPLE_TOL
        if not near.any():
            continue
        score = near.mean() * (1.0 - err[near].mean() / GAP_MULTIPLE_TOL)
        # bonus when s itself is a common gap
        score *= 1.0 + GAP_MATCH_BONUS * np.mean(np.abs(gaps - s) < 0.5)
        if score > best_score:
            best_s, best_score = float(s), float(score)

    if best_s <= 0:
        return 0.0, 0.0
    return best_s, min(1.0, best_score)


def estimate(rgba: np.ndarray, heavy: bool = False, lum: np.ndarray | None = None) -> Estimate:
    lum = luminance(rgba) if lum is None else lum
    h, w = lum.shape

    px = edge_profile(lum, axis=0)
    py = edge_profile(lum, axis=1)

    sx, scx = _period_from_boundaries(_boundaries(px), w, NORMAL_MIN, NORMAL_MAX)
    sy, scy = _period_from_boundaries(_boundaries(py), h, NORMAL_MIN, NORMAL_MAX)

    if sx <= 0:
        sx, scx = w / 2.0, 0.0
    if sy <= 0:
        sy, scy = h / 2.0, 0.0

    cols = drift_aware_count(w, sx)
    rows = drift_aware_count(h, sy)
    return Estimate(
        name="boundary",
        step_x=w / cols,
        step_y=h / rows,
        cols=cols,
        rows=rows,
        score_x=float(scx),
        score_y=float(scy),
        candidates_x=[sx],
        candidates_y=[sy],
    )
