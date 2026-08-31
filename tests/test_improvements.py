"""Regression tests for the v1.1 changes: area-averaging subsample, source-unit
steps, k-means palette cap, palette-image reduction, dithering, alpha modes, and
a synthetic-grid detection sweep that pins the estimator behaviour."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from img_to_pixel_art import _apply_palette, _limit_palette, _parse_palette
from pixel_art_converter import detect, map_to_palette, two_stage_pack
from pixel_art_converter.dither import METHODS
from pixel_art_converter.utils import comb_score, subsample_for_budget

ROOT = Path(__file__).resolve().parents[1]


def synthetic_grid(cols, rows, cell, *, seed=1, blur=0.0, n_colors=0, alpha=None):
    rng = np.random.default_rng(seed)
    if n_colors:
        palette = rng.integers(0, 256, size=(n_colors, 3), dtype=np.uint8)
        native = palette[rng.integers(0, n_colors, size=(rows, cols))]
    else:
        native = rng.integers(0, 256, size=(rows, cols, 3), dtype=np.uint8)
    img = np.repeat(np.repeat(native, cell, axis=0), cell, axis=1).astype(np.float64)
    if blur:
        from pixel_art_converter.utils import gaussian_blur

        img = gaussian_blur(img, blur)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if alpha is not None:
        img = np.dstack([img, np.full(img.shape[:2], alpha, np.uint8)])
    return img


# --------------------------------------------------------------------------- #
# synthetic-grid detection sweep
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cols,rows,cell", [
    (16, 16, 10), (20, 16, 12), (32, 32, 6), (24, 24, 8),
    (40, 30, 5), (12, 12, 20), (16, 24, 9), (30, 30, 7),
])
@pytest.mark.parametrize("blur", [0.0, 0.8])
@pytest.mark.parametrize("n_colors", [0, 8])
def test_detect_recovers_synthetic_grid(cols, rows, cell, blur, n_colors):
    img = synthetic_grid(cols, rows, cell, blur=blur, n_colors=n_colors)
    det = detect(img)
    assert (det.cols, det.rows) == (cols, rows)


def test_detect_fast_mode_recovers_grid():
    img = synthetic_grid(28, 22, 9)
    det = detect(img, mode="fast")
    assert (det.cols, det.rows) == (28, 22)


# --------------------------------------------------------------------------- #
# subsampling + source-unit steps
# --------------------------------------------------------------------------- #
def test_subsample_is_area_averaged_not_decimated():
    h = w = 400
    img = np.zeros((h, w, 4), np.uint8)
    img[..., 3] = 255
    img[::2, :, 0] = 255  # every other row bright red; decimation would drop it
    small = subsample_for_budget(img, 400 * 400 // 9)
    assert small.shape[0] < h and small.shape[1] < w
    # a mean over 2+ rows keeps a mid-range red everywhere, not pure 0 or 255
    assert 40 < small[..., 0].mean() < 215
    assert small[..., 0].std() < 90


def test_detection_step_is_in_source_pixels_when_subsampled():
    # 1800x1800 (> 1.6M px budget) with 45px cells -> 40x40 grid
    img = synthetic_grid(40, 40, 45, n_colors=10)
    assert img.shape[0] * img.shape[1] > 1_600_000
    det = detect(img)
    assert (det.cols, det.rows) == (40, 40)
    assert abs(det.step_x - 45.0) < 1.0
    assert abs(det.step_y - 45.0) < 1.0


# --------------------------------------------------------------------------- #
# palette cap via k-means
# --------------------------------------------------------------------------- #
def test_limit_palette_hard_caps_distinct_colours():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(60, 60, 3), dtype=np.uint8)  # ~3600 unique
    out = _limit_palette(img, 12)
    assert len(np.unique(out.reshape(-1, 3), axis=0)) <= 12


def test_limit_palette_is_deterministic():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(50, 50, 3), dtype=np.uint8)
    assert np.array_equal(_limit_palette(img, 8), _limit_palette(img, 8))


def test_parse_palette_reduces_large_palette_image(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)  # way over 256 unique
    path = tmp_path / "pal.png"
    Image.fromarray(data).save(path)
    palette = _parse_palette(str(path))
    assert 1 <= len(palette) <= 256
    assert palette.dtype == np.uint8


# --------------------------------------------------------------------------- #
# dithering
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", METHODS)
def test_map_to_palette_only_emits_palette_colours(method):
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    palette = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 128, 255]], np.uint8)
    out = map_to_palette(img, palette, method)
    emitted = {tuple(c) for c in np.unique(out.reshape(-1, 3), axis=0)}
    assert emitted <= {tuple(c) for c in palette}


def test_dither_changes_output_on_a_gradient():
    ramp = np.linspace(0, 255, 64, dtype=np.uint8)
    img = np.repeat(ramp[None, :, None], 16, axis=0).repeat(3, axis=2)
    palette = np.array([[0, 0, 0], [255, 255, 255]], np.uint8)
    plain = map_to_palette(img, palette, "none")
    floyd = map_to_palette(img, palette, "floyd")
    ordered = map_to_palette(img, palette, "ordered")
    assert not np.array_equal(plain, floyd)
    assert not np.array_equal(plain, ordered)


def test_apply_palette_preserves_alpha_with_dither():
    palette = _parse_palette("#000000,ffffff")
    img = np.array([[[240, 240, 240, 0], [10, 10, 10, 255]]], dtype=np.uint8)
    out = _apply_palette(img, palette, dither="ordered")
    assert out.shape == (1, 2, 4)
    assert out[0, 0, 3] == 0 and out[0, 1, 3] == 255


# --------------------------------------------------------------------------- #
# alpha modes
# --------------------------------------------------------------------------- #
def test_two_stage_pack_continuous_alpha():
    img = synthetic_grid(16, 16, 10)
    ramp = np.linspace(0, 255, img.shape[1], dtype=np.uint8)
    rgba = np.dstack([img, np.tile(ramp, (img.shape[0], 1))])
    binary = two_stage_pack(rgba, 16, 16, binary_alpha=True)
    cont = two_stage_pack(rgba, 16, 16, binary_alpha=False)
    assert set(np.unique(binary[..., 3])) <= {0, 255}
    assert len(np.unique(cont[..., 3])) > 2


def test_two_stage_pack_k_colors_limits_kmeans_palette():
    img = synthetic_grid(20, 20, 8, n_colors=30)
    rgba = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    out = two_stage_pack(rgba, 20, 20, k_colors=6)
    # cell means can drift, but the winner colours come from <=6 clusters
    assert len(np.unique(out.reshape(-1, 3), axis=0)) <= 20 * 20


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _run_cli(*args):
    return subprocess.run(
        [sys.executable, str(ROOT / "img_to_pixel_art.py"), *args],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_cli_n_colors_output_is_capped(tmp_path):
    src = tmp_path / "src.png"
    Image.fromarray(synthetic_grid(20, 16, 12)).save(src)
    out = tmp_path / "out.png"
    res = _run_cli(str(src), "--block_size", "12", "--n_colors", "10",
                   "--dither", "floyd", "--output", str(out))
    assert res.returncode == 0, res.stderr
    colours = np.unique(np.array(Image.open(out)).reshape(-1, 3), axis=0)
    assert len(colours) <= 10


def test_cli_keep_alpha_and_output_path(tmp_path):
    src = tmp_path / "src.png"
    img = synthetic_grid(16, 16, 12, alpha=255)
    ramp = np.linspace(0, 255, img.shape[1], dtype=np.uint8)
    img[..., 3] = np.tile(ramp, (img.shape[0], 1))
    Image.fromarray(img, mode="RGBA").save(src)
    out = tmp_path / "kept.png"
    res = _run_cli(str(src), "--block_size", "12", "--keep-alpha", "--output", str(out))
    assert res.returncode == 0, res.stderr
    assert out.exists()
    alpha = np.array(Image.open(out).convert("RGBA"))[..., 3]
    assert len(np.unique(alpha)) > 2


# --------------------------------------------------------------------------- #
# vectorised comb_score parity
# --------------------------------------------------------------------------- #
def test_comb_score_matches_reference_loop():
    rng = np.random.default_rng(0)

    def reference(prof, step):
        n = len(prof)
        base = float(np.mean(np.abs(prof))) + 1e-9
        phases = np.arange(0.0, step, max(0.05, step / 24.0))
        kmax = int(n / step)
        k = np.arange(kmax + 1)
        best = (-1e9, 0.0)
        for p in phases:
            def teeth(off):
                pos = p + off + k * step
                pos = pos[(pos >= 0) & (pos <= n - 1)]
                return np.interp(pos, np.arange(n), prof) if pos.size else np.array([0.0])
            s = (teeth(0).mean() - teeth(step / 2).mean()) / base
            if s > best[0]:
                best = (s, float(p % step))
        return best

    for _ in range(60):
        n = int(rng.integers(30, 400))
        prof = rng.random(n) * rng.integers(1, 8)
        step = float(rng.uniform(2.0, n / 2.5))
        a = reference(prof, step)
        b = comb_score(prof, step)
        assert abs(a[0] - b[0]) < 1e-9
        assert abs(a[1] - b[1]) < 1e-9
