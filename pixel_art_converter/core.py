"""Detection orchestration: cheap consensus and heavy arbitration.

Public entry point is :func:`detect`.  The three cheap estimators
(:mod:`autocorr`, :mod:`boundary`, :mod:`selfsim`) run first; if they agree
we return early.  Otherwise the heavy arbiter builds fused / variance /
distillability evidence per axis and picks a step.  The arbiter zeroes the
public phases; sub-modules still compute phases internally for reconstruction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import autocorr, evidence, distillability, boundary, selfsim, intvar
from .utils import (
    Estimate,
    comb_score,
    drift_aware_count,
    edge_profile,
    luminance,
    subsample_for_budget,
    to_rgba,
)
from .fit_grid import fit_grid

BOUNDARY_FAST_SCORE = 0.30
SQUARE_LOG_TOL = 0.35
QUALIFIED_FRAC = 0.75
DETECT_BUDGET = 1_600_000
LOWMEM_BUDGET = 500_000

# --- heavy-arbiter fusion weights (pinned so tuning stays reviewable) ------- #
ARB_W_INTVAR = 0.20          # weight on the phase-contrast variance evidence
ARB_W_AGREE = 0.25           # weight per cheap estimator agreeing (beyond the first)
ARB_W_DISTILL = 0.60         # weight on the distillability evidence
ARB_DETAIL_PENALTY = 0.6     # score multiplier for steps coarser than detail_limit
ARB_HARMONIC_FRAC = 0.85     # adopt step/2 or step/3 at this fraction of the best score
ARB_SQUARE_LOG_DIVERGE = 0.6  # |log(cell aspect / image aspect)| above this = "diverged"


@dataclass
class Detection:
    step_x: float
    step_y: float
    cols: int
    rows: int
    phase_x: float
    phase_y: float
    consensus: str


class DetectionError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
def _agree(a: int, b: int) -> bool:
    return abs(a - b) <= max(1, round(0.01 * max(a, b)))


def _run_cheap(rgba, lum=None):
    """Run the three cheap estimators over the normal 2-24 px range.

    Each is isolated: a sub-detector raising must not sink the others.
    *autocorr* and *boundary* share the luminance of *rgba*, computed once here;
    *selfsim* subsamples internally and derives its own.
    """
    lum = luminance(rgba) if lum is None else lum
    out = []
    for mod in (autocorr, boundary, selfsim):
        try:
            if mod is selfsim:
                out.append(mod.estimate(rgba, heavy=False))
            else:
                out.append(mod.estimate(rgba, heavy=False, lum=lum))
        except Exception:
            out.append(None)
    return out


def _median_counts(ests):
    cols = int(round(float(np.median([e.cols for e in ests]))))
    rows = int(round(float(np.median([e.rows for e in ests]))))
    return max(1, cols), max(1, rows)


def _finish(hw, cols, rows, consensus) -> Detection:
    """Build a :class:`Detection`; *hw* is the ``(height, width)`` of the
    original image so ``step_x`` / ``step_y`` stay in source-pixel units even
    when detection ran on a subsampled copy."""
    h, w = hw
    cols, rows = max(1, int(cols)), max(1, int(rows))
    return Detection(
        step_x=w / cols,
        step_y=h / rows,
        cols=cols,
        rows=rows,
        phase_x=0.0,
        phase_y=0.0,
        consensus=consensus,
    )


# --------------------------------------------------------------------------- #
# heavy arbitration
# --------------------------------------------------------------------------- #
def _axis_candidates(ests, autoc: Estimate, axis: int):
    vals = []
    for e in ests:
        if e is None:
            continue
        vals.append(e.step_x if axis == 0 else e.step_y)
    if autoc is not None:
        vals.extend(autoc.candidates_x if axis == 0 else autoc.candidates_y)
        vals.append(autoc.step_x if axis == 0 else autoc.step_y)
    uniq = []
    for v in sorted(v for v in vals if v and v >= 2):
        if not uniq or abs(v - uniq[-1]) > 0.4:
            uniq.append(float(v))
    return uniq


def _arbitrate_axis(rgba, ests, autoc, axis, other_count, detail_limit, lum=None):
    h, w = rgba.shape[:2]
    length = w if axis == 0 else h
    cands = _axis_candidates(ests, autoc, axis)
    if not cands:
        cands = [length / max(1, other_count)]

    steps_seen = [
        (e.step_x if axis == 0 else e.step_y)
        for e in ests
        if e is not None
    ]

    scored = []
    for c in cands:
        count = drift_aware_count(length, c)
        fused = evidence.evidence(rgba, axis, c, lum=lum)
        sv = intvar.evidence(rgba, axis, c, lum=lum)
        agree = sum(1 for s in steps_seen if s and abs(s - c) <= max(0.5, 0.05 * c))
        dist = distillability.axis_distillability(rgba, axis, count, other_count, lum=lum)
        score = fused + ARB_W_INTVAR * sv + ARB_W_AGREE * max(0, agree - 1)
        if dist > 0:
            score += ARB_W_DISTILL * dist
        if detail_limit and c > detail_limit:
            score *= ARB_DETAIL_PENALTY
        scored.append((c, count, score))

    best = max(scored, key=lambda t: t[2])
    best_score = best[2]

    # smallest-qualified: among near-best candidates prefer the finest step
    qualified = [t for t in scored if t[2] >= QUALIFIED_FRAC * best_score and best_score > 0]
    if qualified:
        best = min(qualified, key=lambda t: t[0])

    # 2x / 3x harmonic reconciliation
    step = best[0]
    for div in (2, 3):
        finer = step / div
        if finer < 2:
            continue
        fs = next((t for t in scored if abs(t[0] - finer) <= max(0.5, 0.06 * finer)), None)
        if fs and fs[2] >= ARB_HARMONIC_FRAC * best_score:
            best = fs
            step = fs[0]

    return best[0], best[1], best_score


def _heavy_arbitrate(rgba, ests, autoc, orig_hw) -> Detection:
    h, w = rgba.shape[:2]
    wh_ratio = w / h

    lum = luminance(rgba)
    # cap steps that are much coarser than half the shorter side
    detail_limit = 0.5 * min(h, w)

    # first pass, axes independent, seeded with the cheap consensus counts
    live_counts = [e.rows for e in ests if e is not None]
    seed_rows = int(np.median(live_counts)) if live_counts else max(1, round(h / 8))

    sx, cols, scx = _arbitrate_axis(rgba, ests, autoc, 0, seed_rows, detail_limit, lum=lum)
    sy, rows, scy = _arbitrate_axis(rgba, ests, autoc, 1, cols, detail_limit, lum=lum)

    # near-square prior: nudge toward cell aspect matching the image aspect
    if rows > 0 and cols > 0:
        aspect_log = math.log((cols / rows) / wh_ratio)
        if abs(aspect_log) > ARB_SQUARE_LOG_DIVERGE:
            # axes diverge a lot -> try the finer step on both
            fine = min(w / cols, h / rows)
            cc = drift_aware_count(w, fine)
            rr = drift_aware_count(h, fine)
            if distillability.distillability(rgba, cc, rr, lum=lum) >= distillability.distillability(
                rgba, cols, rows, lum=lum
            ):
                cols, rows = cc, rr

    # final comb refine + drift-aware recount
    px = edge_profile(lum, axis=0)
    py = edge_profile(lum, axis=1)
    cols = _refine_count(px, w, cols)
    rows = _refine_count(py, h, rows)

    # harmonise near-equal axes through the harmonic mean
    if abs(cols - rows) <= 1 and cols != rows:
        hm = int(round(2 * cols * rows / (cols + rows)))
        cols = rows = max(1, hm)

    return _finish(orig_hw, cols, rows, "arbitrated")


def _refine_count(prof, length, count):
    count = max(1, int(count))
    best = (count, -1e9)
    for cand in {count - 1, count, count + 1}:
        if cand < 1:
            continue
        sc, _ = comb_score(prof, length / cand)
        if sc > best[1]:
            best = (cand, sc)
    return best[0]


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def detect(rgba, mode: str = "full", low_memory: bool = False) -> Detection:
    """Detect the native grid of *rgba* independently on X and Y.

    *rgba* may be anything :func:`pixel_art_converter.utils.to_rgba` accepts.
    ``mode`` is ``"full"`` (cheap consensus then heavy arbitration) or
    ``"fast"`` (cheap only).  Returns a :class:`Detection` whose ``step_x`` /
    ``step_y`` are always in source-pixel units.
    """
    rgba = to_rgba(rgba)
    budget = LOWMEM_BUDGET if low_memory else DETECT_BUDGET
    work = subsample_for_budget(rgba, budget)
    H, W = rgba.shape[:2]
    hw = (H, W)

    ests = _run_cheap(work)
    live = [e for e in ests if e is not None]
    autoc = ests[0]
    bnd = ests[1]

    if not live:
        try:
            sx, sy, cols, rows = fit_grid(rgba)
            return _finish(hw, cols, rows, "fallback:fit_grid")
        except Exception as exc:  # nothing worked
            raise DetectionError("all sub-detectors failed") from exc

    # ---- immediate fast path: autocorr + boundary strongly agree ----
    if (
        autoc is not None
        and bnd is not None
        and _agree(autoc.cols, bnd.cols)
        and _agree(autoc.rows, bnd.rows)
        and bnd.score_x >= BOUNDARY_FAST_SCORE
        and bnd.score_y >= BOUNDARY_FAST_SCORE
    ):
        return _finish(hw, autoc.cols, autoc.rows, "fast:ac+bd(S)")

    # ---- fast mode: no heavy arbitration ----
    if mode == "fast":
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                if _agree(live[i].cols, live[j].cols) and _agree(live[i].rows, live[j].rows):
                    return _finish(
                        hw,
                        live[i].cols,
                        live[i].rows,
                        f"fastmode:{live[i].name}+{live[j].name}",
                    )
        base = autoc if autoc is not None else live[0]
        return _finish(hw, base.cols, base.rows, "fastmode:lowconf")

    # ---- full: quick supermajority ----
    if len(live) == 3 and all(_agree(live[0].cols, e.cols) for e in live[1:]) and all(
        _agree(live[0].rows, e.rows) for e in live[1:]
    ):
        cols, rows = _median_counts(live)
        if abs(math.log((cols / rows) / (W / H))) < SQUARE_LOG_TOL:
            names = "+".join(e.name for e in live)
            return _finish(hw, cols, rows, f"fast:{names}")

    # ---- heavy arbitration ----
    try:
        return _heavy_arbitrate(work, ests, autoc, hw)
    except Exception:
        base = autoc if autoc is not None else live[0]
        return _finish(hw, base.cols, base.rows, "arbitrated")
