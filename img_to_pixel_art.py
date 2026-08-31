"""img_to_pixel_art -- single-file conversion CLI.

    python img_to_pixel_art.py INPUT [--block_size FLOAT] [--n_colors INT]
                                     [--palette HEX,...] [--mode full|fast]
                                     [--dither none|ordered|floyd]
                                     [--keep-alpha] [--output PATH]

Reads INPUT as RGBA, detects the native grid (or derives it from --block_size),
repacks it with ``two_stage_pack``, optionally reduces the RGB palette, and
saves ``STEM_COLSxROWS_pixeled.png`` next to the input (or to --output).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from pixel_art_converter.core import detect
from pixel_art_converter.dither import METHODS as DITHER_METHODS, map_to_palette
from pixel_art_converter.quantize import kmeans_rgb
from pixel_art_converter.rebuild import two_stage_pack


def _load_rgba(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.ascontiguousarray(np.array(im.convert("RGBA")), dtype=np.uint8)


def _limit_palette(arr: np.ndarray, n_colors: int, dither: str = "none") -> np.ndarray:
    """Reduce to at most *n_colors* distinct RGB colours via RGB k-means; alpha
    keeps its own binary channel and never consumes a palette slot."""
    has_alpha = arr.ndim == 3 and arr.shape[2] == 4
    n = max(1, min(256, int(n_colors)))
    rgb = arr[..., :3]
    flat = np.ascontiguousarray(rgb).reshape(-1, 3)
    uniq = np.unique(flat, axis=0)
    if len(uniq) <= n and dither == "none":
        rgb_q = np.ascontiguousarray(rgb, dtype=np.uint8)
    else:
        _, centers = kmeans_rgb(flat.astype(np.float64), n)
        rgb_q = map_to_palette(rgb, centers, dither)
    if not has_alpha:
        return rgb_q
    return np.ascontiguousarray(np.dstack([rgb_q, arr[..., 3]]))


def _parse_palette(value: str) -> np.ndarray:
    """Parse RGB hex colours or extract the palette from an image.

    An image with more than 256 unique RGB colours is reduced to 256 with
    k-means rather than rejected."""
    palette_path = Path(value)
    if palette_path.is_file():
        with Image.open(palette_path) as image:
            rgb = np.array(image.convert("RGB"), dtype=np.uint8).reshape(-1, 3)
        colors = np.unique(rgb, axis=0)
        if len(colors) < 1:
            raise ValueError("palette image contains no colours")
        if len(colors) > 256:
            _, centers = kmeans_rgb(rgb.astype(np.float64), 256)
            colors = np.unique(np.clip(np.round(centers), 0, 255).astype(np.uint8), axis=0)
        return np.ascontiguousarray(colors)

    tokens = [token.strip().lstrip("#") for token in value.split(",") if token.strip()]
    if not 1 <= len(tokens) <= 256:
        raise ValueError("palette must contain between 1 and 256 colours")
    colors = []
    for token in tokens:
        if len(token) != 6:
            raise ValueError(f"invalid palette colour {token!r}; use RRGGBB hex")
        try:
            colors.append(tuple(int(token[i:i + 2], 16) for i in (0, 2, 4)))
        except ValueError as exc:
            raise ValueError(f"invalid palette colour {token!r}; use RRGGBB hex") from exc
    return np.asarray(colors, dtype=np.uint8)


def _apply_palette(arr: np.ndarray, palette: np.ndarray, dither: str = "none") -> np.ndarray:
    """Map each RGB pixel to an explicit palette, optionally with dithering."""
    pal = np.asarray(palette, dtype=np.uint8).reshape(-1, 3)
    mapped = map_to_palette(arr[..., :3], pal, dither)
    if arr.ndim == 3 and arr.shape[2] == 4:
        mapped = np.dstack([mapped, arr[..., 3]])
    return np.ascontiguousarray(mapped)


def main(image_path: str, n_colors: int | None = None,
         specified_block_size: float | None = None, mode: str = "full",
         palette: str | None = None, dither: str = "none",
         keep_alpha: bool = False, output_path: str | None = None) -> str:
    print(f"Processing image: {image_path}")
    rgba = _load_rgba(image_path)
    h, w = rgba.shape[:2]
    print(f"  input: {w}x{h} RGBA")

    if specified_block_size is not None:
        b = float(specified_block_size)
        if not np.isfinite(b) or b <= 0:
            raise ValueError("block_size must be a positive finite number")
        cols = max(1, round(w / b))
        rows = max(1, round(h / b))
        print(f"  block_size={b} -> grid {cols}x{rows}")
    else:
        print(f"  detecting grid (mode={mode}) ...")
        det = detect(rgba, mode=mode)
        cols, rows = det.cols, det.rows
        print(f"  detected grid {cols}x{rows}  step=({det.step_x:.3f}, {det.step_y:.3f})"
              f"  consensus={det.consensus}")

    if n_colors is not None and not 1 <= int(n_colors) <= 256:
        raise ValueError("n_colors must be between 1 and 256")
    if dither != "none" and n_colors is None and palette is None:
        print("  note: --dither has no effect without --n_colors or --palette")

    print("  packing (two_stage_pack) ...")
    out = two_stage_pack(
        rgba, cols, rows,
        k_colors=int(n_colors) if n_colors else 0,
        binary_alpha=not keep_alpha,
    )

    if n_colors is not None:
        print(f"  reducing palette to <= {n_colors} colours"
              f"{'' if palette is not None or dither == 'none' else f' (dither={dither})'} ...")
        out = _limit_palette(out, int(n_colors), dither="none" if palette is not None else dither)
    if palette is not None:
        parsed_palette = _parse_palette(palette)
        print(f"  mapping to explicit palette ({len(parsed_palette)} colours,"
              f" dither={dither}) ...")
        out = _apply_palette(out, parsed_palette, dither=dither)

    if output_path is not None:
        out_path = output_path
    else:
        stem = Path(image_path)
        out_path = str(stem.with_name(f"{stem.stem}_{cols}x{rows}_pixeled.png"))
    mode_str = "RGBA" if out.ndim == 3 and out.shape[2] == 4 else "RGB"
    Image.fromarray(np.ascontiguousarray(out), mode=mode_str).save(
        out_path, format="PNG", optimize=True
    )
    print(f"New image saved as: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to pixel art.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--block_size", type=float, default=None,
                        help="Manual cell size in source pixels (optional)")
    parser.add_argument("--n_colors", type=int, default=None,
                        help="Cap the number of distinct RGB colours (optional)")
    parser.add_argument("--palette", type=str, default=None,
                        help="RGB hex colours or an image path containing the palette")
    parser.add_argument("--mode", choices=("full", "fast"), default="full",
                        help="Detection mode when block_size is not given")
    parser.add_argument("--dither", choices=DITHER_METHODS, default="none",
                        help="Dithering when snapping to --n_colors / --palette")
    parser.add_argument("--keep-alpha", dest="keep_alpha", action="store_true",
                        help="Keep a continuous alpha channel instead of 0/255 per cell")
    parser.add_argument("--output", type=str, default=None,
                        help="Explicit output path (default: next to the input)")
    args = parser.parse_args()
    main(args.image_path, n_colors=args.n_colors,
         specified_block_size=args.block_size, mode=args.mode, palette=args.palette,
         dither=args.dither, keep_alpha=args.keep_alpha, output_path=args.output)
