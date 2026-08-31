"""Generate the visual examples referenced by the README."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from img_to_pixel_art import _load_rgba
from pixel_art_converter import two_stage_pack


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "vulcao.png"


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    rgba = _load_rgba(str(SOURCE))
    source = Image.fromarray(rgba, mode="RGBA")
    grid = (171, 171)
    pixel_art = Image.fromarray(two_stage_pack(rgba, *grid), mode="RGB")
    normal = source.convert("RGB").resize(grid, Image.Resampling.LANCZOS)

    OUT.mkdir(exist_ok=True)
    pixel_art.save(OUT / "vulcao_pixel_art.png", optimize=True)

    scale = 2
    tile_size = (grid[0] * scale, grid[1] * scale)
    canvas = Image.new("RGB", (tile_size[0] * 2, tile_size[1] + 44), "#16181d")
    draw = ImageDraw.Draw(canvas)
    labels = ("Downscale comum (Lanczos)", "Conversor (171 x 171)")
    for index, image in enumerate((normal, pixel_art)):
        image = image.resize(tile_size, Image.Resampling.NEAREST)
        x = index * tile_size[0]
        canvas.paste(image, (x, 44))
        draw.text((x + 8, 13), labels[index], fill="#f4f1ea", font=font(18))
    canvas.save(OUT / "vulcao_comparison.png", optimize=True)


if __name__ == "__main__":
    main()
