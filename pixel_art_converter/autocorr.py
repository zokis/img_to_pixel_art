"""Autocorrelation / cepstrum grid estimator.

Builds edge-energy profiles from ``|diff|`` and ``|second diff|`` of the
alpha-composited luminance, then combines three votes per axis:

* the first strong local maximum of the FFT autocorrelation,
* a cepstral period estimate,
* a comb-minus-anticomb refinement on a 0.02 px grid, evaluated at ``k*s``.
"""

from __future__ import annotations

import numpy as np

from .utils import (
    Estimate,
    acf_candidates,
    cepstrum_period,
    comb_score,
    drift_aware_count,
    edge_profile,
    luminance,
    refine_step,
)

NORMAL_MIN, NORMAL_MAX = 2.0, 24.0
HEAVY_MAX = 64.0

# --- pinned constants (kept explicit so tuning is reviewable) --------------- #
SMALLEST_QUALIFIED_FRAC = 0.88   # a candidate this close to the best score qualifies
HARMONIC_KEEP_FRAC = 0.7        # adopt step/2 or step/3 when it scores at least this
                                # fraction of the current best


def _axis_step(prof: np.ndarray, smin: float, smax: float):
    """Return ``(step, phase, score, candidates)`` for a single profile."""
    n = len(prof)
    smax = min(smax, n / 2.0)
    if smax <= smin:
        return 0.0, 0.0, 0.0, []

    cands = acf_candidates(prof, smin, smax, top=6)
    cep = cepstrum_period(prof, smin, smax)
    if cep >= smin:
        cands.append(cep)

    # also seed with sub-multiples so a 2x/3x ACF peak still yields the base
    seeds = set()
    for c in cands:
        for div in (1, 2, 3):
            v = c / div
            if smin <= v <= smax:
                seeds.add(round(v, 2))
    if not seeds:
        seeds = {round(v, 2) for v in np.arange(smin, smax, 0.5)}

    results = []
    for s in sorted(seeds):
        step, phase, score = refine_step(prof, s, span=0.6, res=0.01)
        # also score the untouched seed: refine's grid can just miss an
        # integer period and unfairly penalise the finer (more teeth) candidate
        ssc, sph = comb_score(prof, s)
        if ssc > score:
            step, phase, score = s, sph, ssc
        if step >= smin:
            results.append((float(step), float(phase), float(score)))
    if not results:
        return 0.0, 0.0, 0.0, sorted(seeds)

    results.sort(key=lambda r: r[0])
    best = max(results, key=lambda r: r[2])
    best_score = best[2]

    # smallest-qualified: among near-best candidates, take the finest step
    qualified = [r for r in results
                 if r[2] >= SMALLEST_QUALIFIED_FRAC * best_score and best_score > 0]
    step, phase, score = min(qualified, key=lambda r: r[0]) if qualified else best

    # explicit 2x / 3x harmonic reconciliation: a real grid at s explains
    # everything the 2s / 3s peak explains, so prefer the divisor when it holds
    changed = True
    while changed:
        changed = False
        for div in (2, 3):
            target = step / div
            if target < smin:
                continue
            for r in results:
                if abs(r[0] - target) <= max(0.15, 0.03 * target) and r[2] >= HARMONIC_KEEP_FRAC * score:
                    step, phase, score = r
                    changed = True
                    break
    return float(step), float(phase), float(score), sorted(seeds)


def estimate(rgba: np.ndarray, heavy: bool = False, lum: np.ndarray | None = None) -> Estimate:
    lum = luminance(rgba) if lum is None else lum
    h, w = lum.shape
    smax = HEAVY_MAX if heavy else NORMAL_MAX

    px = edge_profile(lum, axis=0)
    py = edge_profile(lum, axis=1)

    sx, phx, scx, cx = _axis_step(px, NORMAL_MIN, smax)
    sy, phy, scy, cy = _axis_step(py, NORMAL_MIN, smax)

    if sx <= 0:
        sx = w / 2.0
    if sy <= 0:
        sy = h / 2.0

    cols = drift_aware_count(w, sx)
    rows = drift_aware_count(h, sy)
    return Estimate(
        name="autocorr",
        step_x=w / cols,
        step_y=h / rows,
        cols=cols,
        rows=rows,
        phase_x=phx,
        phase_y=phy,
        score_x=max(0.0, scx),
        score_y=max(0.0, scy),
        candidates_x=cx,
        candidates_y=cy,
    )
