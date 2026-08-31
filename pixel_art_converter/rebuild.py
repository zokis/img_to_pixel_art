"""Canonical two-stage reconstruction for the active conversion path."""

from __future__ import annotations

import numpy as np

from .quantize import adaptive_k, kmeans_rgb

WINNER_MIN_WEIGHT = 1e-3


def two_stage_pack(rgba: np.ndarray, cols: int, rows: int, k_colors: int = 0,
                   binary_alpha: bool = True) -> np.ndarray:
    """Rebuild an image with one representative colour per detected cell.

    *k_colors* caps the global k-means palette (``0`` = pick adaptively).  With
    *binary_alpha* (the default) each cell's alpha is snapped to 0/255 by an
    opaque-majority vote; set it ``False`` to keep a tent-weighted mean alpha.
    """
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("two_stage_pack expects an (H, W, 4) uint8 array")
    h, w = rgba.shape[:2]
    cols, rows = int(max(1, cols)), int(max(1, rows))

    rgb = rgba[..., :3].astype(np.float64).reshape(-1, 3)
    alpha = rgba[..., 3].reshape(-1)
    opaque = alpha > 127
    k = int(k_colors) if k_colors and k_colors > 0 else adaptive_k(rgba)
    src_for_kmeans = rgb[opaque] if opaque.any() else rgb
    _, centers = kmeans_rgb(src_for_kmeans, k)

    c2 = (centers**2).sum(1)
    labels = np.empty(len(rgb), dtype=np.int64)
    for start in range(0, len(rgb), 1_000_000):
        block = rgb[start : start + 1_000_000]
        distances = -2.0 * (block @ centers.T) + c2[None, :]
        labels[start : start + len(block)] = distances.argmin(1)
    clusters = len(centers)

    xs = np.arange(w)
    ys = np.arange(h)
    fx = xs * cols / w
    fy = ys * rows / h
    ix = np.clip(np.floor(fx).astype(np.int64), 0, cols - 1)
    iy = np.clip(np.floor(fy).astype(np.int64), 0, rows - 1)
    wx = 1.0 - 2.0 * np.abs((fx - ix) - 0.5)
    wy = 1.0 - 2.0 * np.abs((fy - iy) - 0.5)
    cell = (iy[:, None] * cols + ix[None, :]).reshape(-1)
    weight = (wy[:, None] * wx[None, :]).reshape(-1) + 1e-4
    ncell = rows * cols

    votes = np.zeros(ncell * clusters, dtype=np.float64)
    np.add.at(votes, cell * clusters + labels, weight)
    winner = votes.reshape(ncell, clusters).argmax(1)
    selected = weight * (labels == winner[cell])
    sums = [np.bincount(cell, weights=selected * rgb[:, channel], minlength=ncell)
            for channel in range(3)]
    all_weight = np.bincount(cell, weights=weight, minlength=ncell)
    all_sums = [np.bincount(cell, weights=weight * rgb[:, channel], minlength=ncell)
                for channel in range(3)]
    selected_weight = np.bincount(cell, weights=selected, minlength=ncell)
    denom = np.where(selected_weight > WINNER_MIN_WEIGHT, selected_weight, 1.0)
    channels = [np.where(selected_weight > WINNER_MIN_WEIGHT, value / denom,
                         fallback / np.where(all_weight > 0, all_weight, 1.0))
                for value, fallback in zip(sums, all_sums)]
    out_rgb = np.clip(np.round(np.stack(channels, axis=1)), 0, 255).astype(np.uint8)
    out_rgb = out_rgb.reshape(rows, cols, 3)

    if not (alpha < 255).any():
        return np.ascontiguousarray(out_rgb)
    if binary_alpha:
        opaque_count = np.bincount(cell, weights=opaque.astype(np.float64), minlength=ncell)
        total_count = np.maximum(np.bincount(cell, minlength=ncell), 1)
        out_alpha = np.where(opaque_count / total_count > 0.5, 255, 0).astype(np.uint8)
    else:
        alpha_sum = np.bincount(cell, weights=weight * alpha.astype(np.float64), minlength=ncell)
        out_alpha = np.clip(
            np.round(alpha_sum / np.where(all_weight > 0, all_weight, 1.0)), 0, 255
        ).astype(np.uint8)
    return np.ascontiguousarray(np.concatenate([out_rgb, out_alpha.reshape(rows, cols, 1)], axis=2))
