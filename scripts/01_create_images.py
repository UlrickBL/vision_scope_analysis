"""
Render each text prompt as a clean PNG image.
Output: data/images/{idx:03d}_{lang}.png for all 200 prompts.
"""

import pathlib
import textwrap
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 16       # Qwen3.5 vision encoder patch size
MIN_W = 1120          # divisible by patch size, fits ~70 chars at font 28
MIN_H = 224           # minimum height, also divisible by patch size
FONT_SIZE = 28
PADDING = 24
WRAP_WIDTH = 68


def load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "DejaVuSans.ttf",
        "arial.ttf",
        "Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_prompt(text: str, out_path: pathlib.Path, font: ImageFont.FreeTypeFont) -> None:
    lines = textwrap.wrap(text, width=WRAP_WIDTH) or [text]

    # measure line height with a dummy draw
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    line_h = dummy.textbbox((0, 0), "Ag", font=font)[3]
    line_gap = 6

    total_text_h = len(lines) * (line_h + line_gap) - line_gap
    canvas_h = max(MIN_H, total_text_h + 2 * PADDING)
    # snap to patch grid
    canvas_h = ((canvas_h + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE

    img = Image.new("RGB", (MIN_W, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = PADDING
    for line in lines:
        draw.text((PADDING, y), line, fill=(0, 0, 0), font=font)
        y += line_h + line_gap

    img.save(out_path, format="PNG")


def main() -> None:
    font = load_font(FONT_SIZE)

    for lang, fname in [("en", "questions_en.txt"), ("fr", "questions_fr.txt")]:
        lines = (DATA_DIR / fname).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 100, f"Expected 100 lines in {fname}, got {len(lines)}"

        for idx, text in enumerate(lines):
            out_path = IMG_DIR / f"{idx:03d}_{lang}.png"
            render_prompt(text, out_path, font)

        print(f"[{lang}] Created {len(lines)} images in {IMG_DIR}")

    total = sum(1 for _ in IMG_DIR.glob("*.png"))
    print(f"Total: {total} images")


if __name__ == "__main__":
    main()
