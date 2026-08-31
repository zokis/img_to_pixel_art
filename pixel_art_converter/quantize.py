"""RGB k-means and label assignment for the two-stage packer.

OpenCV k-means is preferred (fast, and deterministic with fixed attempts and a
fixed sample).  A SciPy / NumPy Lloyd fallback keeps the module usable without
OpenCV.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2  # type: ignore

    _HAVE_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None
    _HAVE_CV2 = False

SAMPLE_CAP = 60_000
RNG_SEED = 0


def adaptive_k(rgba: np.ndarray) -> int:
    """Pick K from the number of populated 4-bit RGB bins among opaque pixels.

    Bins holding at least 0.3% of the opaque pixels count; K is clamped to
    ``[16, 48]``.
    """
    rgb = rgba[..., :3].reshape(-1, 3)
    if rgba.shape[-1] == 4:
        opaque = rgba[..., 3].reshape(-1) > 127
        if opaque.any():
            rgb = rgb[opaque]
    if rgb.size == 0:
        return 16
    bins = (rgb.astype(np.uint16) >> 4)
    codes = (bins[:, 0] << 8) | (bins[:, 1] << 4) | bins[:, 2]
    counts = np.bincount(codes, minlength=1)
    thresh = max(1.0, 0.003 * len(rgb))
    populated = int(np.count_nonzero(counts >= thresh))
    return int(min(48, max(16, populated)))


def _kmeans_cv2(sample: np.ndarray, k: int):
    # cv2's kmeans++ seeding draws from the global cv::theRNG(); pin it so the
    # same sample yields the same centres on every run.
    cv2.setRNGSeed(RNG_SEED)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, centers = cv2.kmeans(
        sample.astype(np.float32), k, None, crit, 3, cv2.KMEANS_PP_CENTERS
    )
    return centers.astype(np.float64)


def _kmeans_numpy(sample: np.ndarray, k: int):
    rng = np.random.default_rng(RNG_SEED)
    pts = sample.astype(np.float64)
    idx = rng.choice(len(pts), size=min(k, len(pts)), replace=False)
    centers = pts[idx].copy()
    for _ in range(25):
        d = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        new = centers.copy()
        for j in range(len(centers)):
            m = lab == j
            if m.any():
                new[j] = pts[m].mean(0)
        if np.allclose(new, centers, atol=1e-3):
            centers = new
            break
        centers = new
    return centers


def _assign(pixels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Nearest-centre label for every pixel, computed in memory-bounded blocks."""
    n = len(pixels)
    out = np.empty(n, dtype=np.int32)
    step = 1_000_000
    c = centers.astype(np.float64)
    c2 = (c ** 2).sum(1)
    for s in range(0, n, step):
        blk = pixels[s : s + step].astype(np.float64)
        d = -2.0 * (blk @ c.T) + c2[None, :]
        out[s : s + step] = d.argmin(1)
    return out


def kmeans_rgb(pixels_rgb: np.ndarray, k: int, sample_cap: int = SAMPLE_CAP):
    """Cluster ``(N, 3)`` RGB pixels into *k* colours.

    Returns ``(labels, centers)`` where ``labels`` has one entry per input
    pixel and ``centers`` is ``(k', 3)`` float64.
    """
    pixels_rgb = np.ascontiguousarray(pixels_rgb).reshape(-1, 3)
    n = len(pixels_rgb)
    k = int(max(1, min(k, n)))
    if k == 1:
        return np.zeros(n, dtype=np.int32), pixels_rgb.mean(0, keepdims=True)

    if n > sample_cap:
        rng = np.random.default_rng(RNG_SEED)
        sample = pixels_rgb[rng.choice(n, size=sample_cap, replace=False)]
    else:
        sample = pixels_rgb

    if _HAVE_CV2:
        centers = _kmeans_cv2(sample, k)
    else:  # pragma: no cover - exercised only without OpenCV
        centers = _kmeans_numpy(sample, k)

    labels = _assign(pixels_rgb, centers)
    return labels, centers
