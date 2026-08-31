"""Tests for the single-image conversion flow."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from img_to_pixel_art import _apply_palette, _limit_palette, _parse_palette
from pixel_art_converter import detect, two_stage_pack

ROOT = Path(__file__).resolve().parents[1]


def grid(cols=16, rows=16, step=10, alpha=False):
    rng = np.random.default_rng(42)
    native = rng.integers(0, 256, size=(rows, cols, 3), dtype=np.uint8)
    image = np.repeat(np.repeat(native, step, axis=0), step, axis=1)
    if alpha:
        channel = np.full(image.shape[:2] + (1,), 255, dtype=np.uint8)
        channel[:step * 3] = 0
        image = np.concatenate([image, channel], axis=2)
    return image


def test_detection_and_reconstruction_rgb():
    image = grid()
    result = detect(image)
    assert (result.cols, result.rows) == (16, 16)
    rgba = np.dstack([image, np.full(image.shape[:2], 255, np.uint8)])
    out = two_stage_pack(rgba, 16, 16)
    assert out.shape == (16, 16, 3) and out.dtype == np.uint8


def test_rgba_alpha_is_preserved_as_binary_channel():
    image = grid(alpha=True)
    out = two_stage_pack(image, 16, 16)
    assert out.shape == (16, 16, 4)
    assert set(np.unique(out[..., 3])) <= {0, 255}
    assert (out[:3, :, 3] == 0).all()


def test_fast_detection_and_manual_block_size():
    image = grid(cols=20, rows=16)
    result = detect(image, mode="fast")
    assert result.cols >= 1 and result.rows >= 1


def test_palette_limit_does_not_count_alpha():
    image = grid(alpha=True)
    out = _limit_palette(image, 4)
    assert len(np.unique(out[..., :3].reshape(-1, 3), axis=0)) <= 4
    assert set(np.unique(out[..., 3])) <= {0, 255}


def test_palette_limit_preserves_colour_extremes():
    image = np.array([[[240, 20, 20], [20, 220, 20], [20, 20, 230], [230, 220, 20]]], dtype=np.uint8)
    out = _limit_palette(image, 4)
    colors = {tuple(color) for color in np.unique(out.reshape(-1, 3), axis=0)}
    assert any(color[0] > 150 and color[1] < 100 for color in colors)
    assert any(color[1] > 150 and color[0] < 100 for color in colors)
    assert any(color[2] > 150 and color[0] < 100 for color in colors)


def test_explicit_palette_maps_rgb_and_preserves_alpha():
    palette = _parse_palette("#000000,ffffff")
    image = np.array([[[240, 240, 240, 0], [20, 20, 20, 255]]], dtype=np.uint8)
    out = _apply_palette(image, palette)
    assert out.tolist() == [[[255, 255, 255, 0], [0, 0, 0, 255]]]


def test_palette_can_be_loaded_from_image(tmp_path):
    palette_path = tmp_path / "palette.png"
    data = np.array([[[255, 0, 0], [0, 255, 0], [255, 0, 0]]], dtype=np.uint8)
    Image.fromarray(data).save(palette_path)
    assert _parse_palette(str(palette_path)).tolist() == [[0, 255, 0], [255, 0, 0]]


def test_cli_writes_png_and_accepts_options(tmp_path):
    source = tmp_path / "source.png"
    Image.fromarray(grid(cols=20, rows=16)).save(source)
    result = subprocess.run(
        [sys.executable, str(ROOT / "img_to_pixel_art.py"), str(source),
         "--block_size", "20", "--n_colors", "8", "--mode", "fast"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    output = tmp_path / "source_10x8_pixeled.png"
    assert output.exists() and Image.open(output).format == "PNG"


@pytest.mark.parametrize("option", [("--block_size", "0"), ("--n_colors", "257")])
def test_cli_rejects_invalid_options(tmp_path, option):
    source = tmp_path / "source.png"
    Image.fromarray(grid()).save(source)
    result = subprocess.run(
        [sys.executable, str(ROOT / "img_to_pixel_art.py"), str(source), *option],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
