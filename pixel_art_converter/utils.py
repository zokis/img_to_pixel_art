"""Shared helpers: image coercion, luminance, profiles, comb scoring.

These are intentionally dependency-light (numpy + Pillow only).  OpenCV and
SciPy are used opportunistically elsewhere with graceful fallbacks.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List

import numpy as np
from PIL import Image

try:  # optional, only used for a faster gaussian blur
    import cv2  # type: ignore

    _HAVE_CV2 = True
except Exception:  # pragma: no cover - environment dependent
    cv2 = None
    _HAVE_CV2 = False


GRAY_BG = 127.5


class InputError(ValueError):
    """Raised for malformed or out-of-bounds inputs on the public API."""


@dataclass
class Estimate:
    """Result of one cheap sub-detector for both axes."""

    name: str
    step_x: float
    step_y: float
    cols: int
    rows: int
    phase_x: float = 0.0
    phase_y: float = 0.0
    score_x: float = 0.0
    score_y: float = 0.0
    candidates_x: List[float] = field(default_factory=list)
    candidates_y: List[float] = field(default_factory=list)

    @property
    def score(self) -> float:
        return 0.5 * (self.score_x + self.score_y)


# --------------------------------------------------------------------------- #
# image coercion
# --------------------------------------------------------------------------- #
def to_rgba(image) -> np.ndarray:
    """Coerce *image* to a contiguous ``uint8`` array with shape ``(H, W, 4)``.

    Accepts:
      * ``bytes``/``bytearray`` holding an encoded image (PNG/JPEG/...);
      * a :class:`PIL.Image.Image`;
      * a numpy array shaped ``(H, W)``, ``(H, W, 3)`` or ``(H, W, 4)``.

    RGB inputs receive an opaque alpha channel (255).
    """
    if isinstance(image, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(image))) as im:
            arr = np.array(im.convert("RGBA"))
        return np.ascontiguousarray(arr, dtype=np.uint8)

    if isinstance(image, Image.Image):
        return np.ascontiguousarray(np.array(image.convert("RGBA")), dtype=np.uint8)

    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if not np.issubdtype(arr.dtype, np.number):
            raise InputError("array is not numeric / uint8 convertible")
        finite = np.isfinite(arr)
        if not finite.all():
            raise InputError("array holds non-finite values")
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0 and arr.min() >= 0.0:
            arr = arr * 255.0
        if arr.min() < 0 or arr.max() > 255:
            raise InputError("array values outside [0, 255]; not uint8 convertible")
        arr = np.round(arr).astype(np.uint8)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        pass
    else:
        raise InputError(f"unsupported array shape {arr.shape!r}")

    return np.ascontiguousarray(arr, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# luminance / blur
# --------------------------------------------------------------------------- #
def luminance(rgba: np.ndarray) -> np.ndarray:
    """Rec.601 luma with alpha composited over mid gray (127.5)."""
    rgb = rgba[..., :3].astype(np.float64)
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if rgba.shape[-1] == 4:
        a = rgba[..., 3].astype(np.float64) / 255.0
        lum = a * lum + (1.0 - a) * GRAY_BG
    return lum


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img.astype(np.float64)
    if _HAVE_CV2:
        k = int(max(3, round(sigma * 6) | 1))
        return cv2.GaussianBlur(img.astype(np.float32), (k, k), sigma).astype(np.float64)
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(img.astype(np.float64), sigma)
    except Exception:  # pragma: no cover
        # separable box approximation
        r = max(1, int(round(sigma)))
        ker = np.ones(2 * r + 1) / (2 * r + 1)
        out = img.astype(np.float64)
        for ax in (0, 1):
            out = np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), ax, out)
        return out


# --------------------------------------------------------------------------- #
# axis profiles
# --------------------------------------------------------------------------- #
def edge_profile(lum: np.ndarray, axis: int) -> np.ndarray:
    """1-D edge-energy profile along *axis* (0 => vertical grid / x, 1 => y).

    Combines ``|first diff|`` and ``|second diff|`` averaged across the other
    axis, which is robust to internal texture and anti-aliasing.
    """
    if axis == 0:  # variation along columns -> profile length W
        d1 = np.abs(np.diff(lum, axis=1))
        p1 = np.zeros(lum.shape[1])
        p1[1:] = d1.mean(axis=0)
        d2 = np.abs(np.diff(lum, n=2, axis=1))
        p2 = np.zeros(lum.shape[1])
        p2[1:-1] = d2.mean(axis=0)
    else:  # profile length H
        d1 = np.abs(np.diff(lum, axis=0))
        p1 = np.zeros(lum.shape[0])
        p1[1:] = d1.mean(axis=1)
        d2 = np.abs(np.diff(lum, n=2, axis=0))
        p2 = np.zeros(lum.shape[0])
        p2[1:-1] = d2.mean(axis=1)
    prof = p1 + 0.5 * p2
    return prof


def autocorr(sig: np.ndarray) -> np.ndarray:
    """Biased autocorrelation of a zero-mean copy of *sig* via FFT (lags >= 0)."""
    x = np.asarray(sig, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 4 or not np.any(x):
        return np.zeros(n)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, m)
    ac = np.fft.irfft(f * np.conj(f), m)[:n]
    if ac[0] != 0:
        ac = ac / ac[0]
    return ac


def comb_score(prof: np.ndarray, step: float, phase: float | None = None):
    """Comb-minus-anticomb response of *prof* for a given *step*.

    Returns ``(score, best_phase)``.  ``score`` is normalised to roughly
    ``[0, 1]`` by the profile's mean absolute value.  All candidate phases are
    scored in one vectorised pass.
    """
    n = len(prof)
    if step < 2 or step > n:
        return 0.0, 0.0
    prof = np.asarray(prof, dtype=np.float64)
    base = float(np.mean(np.abs(prof))) + 1e-9
    if phase is None:
        phases = np.arange(0.0, step, max(0.05, step / 24.0))
    else:
        phases = np.asarray([phase], dtype=np.float64)
    kmax = int(n / step)
    k = np.arange(kmax + 1)
    xp = np.arange(n)

    teeth_pos = phases[:, None] + k[None, :] * step
    anti_pos = phases[:, None] + step / 2.0 + k[None, :] * step
    teeth = np.interp(teeth_pos.ravel(), xp, prof).reshape(teeth_pos.shape)
    anti = np.interp(anti_pos.ravel(), xp, prof).reshape(anti_pos.shape)

    tmask = (teeth_pos >= 0) & (teeth_pos <= n - 1)
    amask = (anti_pos >= 0) & (anti_pos <= n - 1)
    tcnt = tmask.sum(axis=1)
    acnt = amask.sum(axis=1)
    tmean = np.where(tcnt > 0, np.where(tmask, teeth, 0.0).sum(axis=1) / np.maximum(tcnt, 1), 0.0)
    amean = np.where(acnt > 0, np.where(amask, anti, 0.0).sum(axis=1) / np.maximum(acnt, 1), 0.0)

    scores = (tmean - amean) / base
    j = int(np.argmax(scores))
    return float(scores[j]), float(phases[j] % step)


def refine_step(prof: np.ndarray, step0: float, span: float = 1.0, res: float = 0.02):
    """Local comb search around *step0*.  Returns ``(step, phase, score)``."""
    lo = max(2.0, step0 - span)
    hi = min(len(prof) / 2.0, step0 + span)
    if hi <= lo:
        return step0, 0.0, 0.0
    grid = np.arange(lo, hi + 1e-9, res)
    best = (step0, 0.0, -1e9)
    for s in grid:
        sc, ph = comb_score(prof, s)
        if sc > best[2]:
            best = (float(s), ph, sc)
    return best


def cepstrum_period(prof: np.ndarray, smin: float, smax: float) -> float:
    x = np.asarray(prof, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 8 or not np.any(x):
        return 0.0
    m = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.abs(np.fft.rfft(x, m)) ** 2
    logspec = np.log(spec + 1e-12)
    cep = np.fft.irfft(logspec)[: m // 2]
    lo, hi = int(max(2, smin)), int(min(len(cep) - 1, smax))
    if hi <= lo:
        return 0.0
    q = lo + int(np.argmax(cep[lo:hi]))
    return float(q)


def acf_candidates(prof: np.ndarray, smin: float, smax: float, top: int = 5):
    """Local maxima of the ACF within ``[smin, smax]`` as period candidates."""
    ac = autocorr(prof)
    hi = int(min(len(ac) - 2, smax))
    lo = int(max(2, smin))
    if hi <= lo + 1:
        return []
    seg = ac[lo : hi + 1]
    peaks = []
    for i in range(1, len(seg) - 1):
        if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0:
            peaks.append((seg[i], lo + i))
    peaks.sort(reverse=True)
    return [float(p[1]) for p in peaks[:top]]


def drift_aware_count(length: int, step: float) -> int:
    """Number of cells spanning *length* pixels at cell size ~*step*."""
    if step <= 0:
        return 1
    return int(max(1, round(length / step)))


def subsample_for_budget(rgba: np.ndarray, max_pixels: int) -> np.ndarray:
    """Shrink *rgba* below *max_pixels* with an area-averaging box filter.

    Plain decimation (``rgba[::f, ::f]``) can alias the very grid we are trying
    to detect when the stride lands near the cell size; averaging each ``f*f``
    block keeps the periodic signal intact.
    """
    h, w = rgba.shape[:2]
    if h * w <= max_pixels:
        return rgba
    factor = int(np.ceil(np.sqrt(h * w / max_pixels)))
    if factor <= 1:
        return rgba
    hc, wc = (h // factor) * factor, (w // factor) * factor
    if hc < factor or wc < factor or h * w > 24_000_000:
        # degenerate aspect, or an image large enough that the float copy would
        # dwarf the budget: fall back to plain decimation.
        return np.ascontiguousarray(rgba[::factor, ::factor])
    c = rgba.shape[2]
    blocks = rgba[:hc, :wc].astype(np.float32).reshape(hc // factor, factor, wc // factor, factor, c)
    reduced = blocks.mean(axis=(1, 3))
    return np.ascontiguousarray(np.round(reduced).astype(np.uint8))
