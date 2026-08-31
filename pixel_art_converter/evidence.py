"""Multichannel edge-evidence fusion for the heavy arbiter.

Combines comb responses computed on several independent 1-D profiles per axis:

* per-channel edge energy (R, G, B and, when present, A),
* a "radius" profile (distance-from-centre weighted edge energy),
* a band profile (edge energy of a mid-band slice), and
* a spectral profile (magnitude of the profile's own FFT peaks).

``evidence(rgba, axis, step)`` returns a scalar in roughly ``[0, 1]``.
"""

from __future__ import annotations

import numpy as np

from .utils import comb_score, luminance


def _channel_profiles(rgba: np.ndarray, axis: int):
    profs = []
    for c in range(rgba.shape[-1]):
        ch = rgba[..., c].astype(np.float64)
        profs.append(_edge_1d(ch, axis))
    return profs


def _edge_1d(plane: np.ndarray, axis: int) -> np.ndarray:
    if axis == 0:
        d = np.abs(np.diff(plane, axis=1))
        p = np.zeros(plane.shape[1])
        p[1:] = d.mean(axis=0)
    else:
        d = np.abs(np.diff(plane, axis=0))
        p = np.zeros(plane.shape[0])
        p[1:] = d.mean(axis=1)
    return p


def _radius_profile(lum: np.ndarray, axis: int) -> np.ndarray:
    h, w = lum.shape
    yy = np.linspace(-1, 1, h)[:, None]
    xx = np.linspace(-1, 1, w)[None, :]
    wgt = 1.0 / (0.25 + xx**2 + yy**2)
    return _edge_1d(lum * wgt, axis)


def _band_profile(lum: np.ndarray, axis: int) -> np.ndarray:
    h, w = lum.shape
    if axis == 0:
        band = lum[h // 3 : 2 * h // 3, :]
    else:
        band = lum[:, w // 3 : 2 * w // 3]
    return _edge_1d(band, axis)


def _spectral_profile(prof: np.ndarray) -> np.ndarray:
    x = prof - prof.mean()
    if not np.any(x):
        return prof
    mag = np.abs(np.fft.rfft(x))
    full = np.zeros_like(prof)
    full[: len(mag)] = mag
    return full


def evidence(rgba: np.ndarray, axis: int, step: float, lum: np.ndarray | None = None) -> float:
    if step < 2:
        return 0.0
    lum = luminance(rgba) if lum is None else lum
    profs = _channel_profiles(rgba, axis)
    profs.append(_radius_profile(lum, axis))
    profs.append(_band_profile(lum, axis))

    scores = []
    weights = []
    for i, p in enumerate(profs):
        if not np.any(p):
            continue
        s, _ = comb_score(p, step)
        scores.append(max(0.0, s))
        weights.append(0.6 if i >= len(profs) - 2 else 1.0)

    sp = _spectral_profile(_edge_1d(lum, axis))
    n = len(sp)
    if n > 4 and step > 0:
        f = n / step
        lo, hi = int(max(1, f - 1)), int(min(n - 1, f + 2))
        if hi > lo:
            peak = sp[lo:hi].max() / (sp[1:].mean() + 1e-9)
            scores.append(min(1.0, 0.15 * peak))
            weights.append(0.8)

    if not scores:
        return 0.0
    w = np.asarray(weights)
    return float(np.clip(np.average(scores, weights=w), 0.0, 1.5))
