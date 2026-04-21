"""Generate banner.png — screenshot-mosaic hero with PodLab wordmark sidebar.

Dimensions: 1280x640 (GitHub social preview + README inline).
Layout: left 320px wordmark sidebar (near-black with plum->magenta gradient divider),
right 960x640 mosaic (4 cols x 2 rows of 240x320 thumbnails).

Source frames: 8 CC0 video frames at tools/sample_frames/*.jpg (see SAMPLE_FRAMES_LICENSE.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ERROR: Pillow required. `pip install pillow`")
    sys.exit(1)


REPO_ROOT = Path(__file__).parent.parent
SAMPLES_DIR = Path(__file__).parent / "sample_frames"
OUTPUT_PATH = REPO_ROOT / "banner.png"

WIDTH, HEIGHT = 1280, 640
SIDEBAR_WIDTH = 320
MOSAIC_COLS, MOSAIC_ROWS = 4, 2
TILE_W = (WIDTH - SIDEBAR_WIDTH) // MOSAIC_COLS  # 240
TILE_H = HEIGHT // MOSAIC_ROWS  # 320

BG = (11, 11, 20)              # #0B0B14
GRADIENT_TOP = (75, 29, 84)    # #4B1D54
GRADIENT_BOT = (224, 69, 123)  # #E0457B
TEXT = (245, 243, 238)         # #F5F3EE
SCORE_BG = (0, 0, 0, 160)      # translucent overlay for quality score


def _draw_gradient_divider(img: Image.Image, x: int, width: int = 4) -> None:
    draw = ImageDraw.Draw(img)
    for y in range(HEIGHT):
        t = y / HEIGHT
        r = int(GRADIENT_TOP[0] + t * (GRADIENT_BOT[0] - GRADIENT_TOP[0]))
        g = int(GRADIENT_TOP[1] + t * (GRADIENT_BOT[1] - GRADIENT_TOP[1]))
        b = int(GRADIENT_TOP[2] + t * (GRADIENT_BOT[2] - GRADIENT_TOP[2]))
        draw.rectangle([x, y, x + width - 1, y], fill=(r, g, b))


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/seguibl.ttf",    # Segoe UI Black
        "C:/Windows/Fonts/seguibld.ttf",   # Segoe UI Bold
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def _load_tiles() -> list[Image.Image]:
    if not SAMPLES_DIR.exists():
        print(f"ERROR: sample_frames/ missing at {SAMPLES_DIR}")
        print("Provide 8 CC0 .jpg/.png frames in tools/sample_frames/ before running.")
        sys.exit(1)
    paths = sorted(list(SAMPLES_DIR.glob("*.jpg")) + list(SAMPLES_DIR.glob("*.png")))
    needed = MOSAIC_COLS * MOSAIC_ROWS
    if len(paths) < needed:
        print(f"ERROR: need {needed} frames, found {len(paths)}")
        sys.exit(1)
    tiles = []
    for p in paths[:needed]:
        img = Image.open(p).convert("RGB")
        img = img.resize((TILE_W, TILE_H), Image.LANCZOS)
        tiles.append(img)
    return tiles


def _draw_score_chip(banner: Image.Image, x: int, y: int, score: str) -> None:
    """Draw a small translucent score chip in the bottom-right corner of a tile."""
    overlay = Image.new("RGBA", banner.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(14)
    try:
        bbox = draw.textbbox((0, 0), score, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except AttributeError:
        tw, th = draw.textsize(score, font=font)
    pad = 6
    chip_w = tw + pad * 2
    chip_h = th + pad * 2
    chip_x = x + TILE_W - chip_w - 8
    chip_y = y + TILE_H - chip_h - 8
    draw.rectangle([chip_x, chip_y, chip_x + chip_w, chip_y + chip_h], fill=SCORE_BG)
    draw.text((chip_x + pad, chip_y + pad - 2), score, font=font, fill=TEXT)
    banner.alpha_composite(overlay) if banner.mode == "RGBA" else banner.paste(
        Image.alpha_composite(banner.convert("RGBA"), overlay).convert("RGB")
    )


def main() -> int:
    banner = Image.new("RGB", (WIDTH, HEIGHT), BG)

    # Mosaic
    tiles = _load_tiles()
    scores = ["0.94", "0.87", "0.82", "0.79", "0.76", "0.72", "0.68", "0.61"]
    for idx, tile in enumerate(tiles):
        col = idx % MOSAIC_COLS
        row = idx // MOSAIC_COLS
        x = SIDEBAR_WIDTH + col * TILE_W
        y = row * TILE_H
        banner.paste(tile, (x, y))

    # Score chips after all tiles placed (so the overlay doesn't get painted over)
    banner_rgba = banner.convert("RGBA")
    draw_rgba = ImageDraw.Draw(banner_rgba)
    chip_font = _load_font(14)
    for idx in range(len(tiles)):
        col = idx % MOSAIC_COLS
        row = idx // MOSAIC_COLS
        x = SIDEBAR_WIDTH + col * TILE_W
        y = row * TILE_H
        score = scores[idx]
        try:
            bbox = draw_rgba.textbbox((0, 0), score, font=chip_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw_rgba.textsize(score, font=chip_font)
        pad = 6
        chip_w = tw + pad * 2
        chip_h = th + pad * 2
        chip_x = x + TILE_W - chip_w - 8
        chip_y = y + TILE_H - chip_h - 8
        draw_rgba.rectangle(
            [chip_x, chip_y, chip_x + chip_w, chip_y + chip_h],
            fill=(0, 0, 0, 160),
        )
        draw_rgba.text((chip_x + pad, chip_y + pad - 2), score, font=chip_font, fill=TEXT)
    banner = banner_rgba.convert("RGB")

    # Gradient divider between sidebar and mosaic
    _draw_gradient_divider(banner, SIDEBAR_WIDTH - 4, width=4)

    # Wordmark on sidebar
    draw = ImageDraw.Draw(banner)
    title_font = _load_font(48)
    subtitle_font = _load_font(18)
    tagline_font = _load_font(18)
    suite_font = _load_font(22)

    draw.text((32, 80), "PodLab", font=suite_font, fill=(224, 69, 123))

    draw.text((32, 140), "Thumbnail", font=title_font, fill=TEXT)
    draw.text((32, 200), "Extractor", font=title_font, fill=TEXT)

    draw.text((32, 300), "AI-picked thumbnails", font=tagline_font, fill=TEXT)
    draw.text((32, 324), "from any video.", font=tagline_font, fill=TEXT)

    draw.text((32, HEIGHT - 74), "github.com/REMvisual/", font=subtitle_font, fill=(180, 180, 190))
    draw.text((32, HEIGHT - 50), "podcast-thumbnail-extractor", font=subtitle_font, fill=(180, 180, 190))

    banner.save(OUTPUT_PATH, "PNG", optimize=True)
    print(f"Wrote {OUTPUT_PATH} ({banner.size[0]}x{banner.size[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
