#!/usr/bin/env python3
"""
Overlay coordinate labels onto every image in a dataset folder.

Defaults:
- input_dir:  data/images
- output_dir: augmented_data
- 4x4 grid => 16 coords
- labels: (2,row,col) with row/col consecutive (no skipping)
- dot is the anchor; text is placed right/below the dot so dot is top-left pointer
- text is SOLID BLACK (no outline)
"""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def _load_font(font_size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial.ttf",
        "arial.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_coords(
    im: Image.Image,
    n: int = 4,                 # 4x4 => 16 coords
    prefix: int = 2,
    margin_x_frac: float = 0.10,
    margin_y_frac: float = 0.10,
    font_frac: float = 0.019,   # smaller so more fits
    dot_frac: float = 0.0040,
    text_offset_frac: float = 0.010,
) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)

    mx = int(w * margin_x_frac)
    my = int(h * margin_y_frac)

    # Evenly spaced anchor points (n x n)
    if n == 1:
        xs = [mx + (w - 2 * mx) / 2]
        ys = [my + (h - 2 * my) / 2]
    else:
        step_x = (w - 2 * mx) / (n - 1)
        step_y = (h - 2 * my) / (n - 1)
        xs = [mx + i * step_x for i in range(n)]
        ys = [my + j * step_y for j in range(n)]

    m = min(w, h)
    font_size = max(8, int(m * font_frac))
    dot_r = max(2, int(m * dot_frac))
    offset = max(2, int(m * text_offset_frac))
    font = _load_font(font_size)

    for r in range(1, n + 1):
        y = ys[r - 1]
        for c in range(1, n + 1):
            x = xs[c - 1]

            # Dot at the anchor point
            draw.ellipse(
                (x - dot_r, y - dot_r, x + dot_r, y + dot_r),
                fill=(0, 0,0),
                outline=(0, 0, 0),
                width=1,
            )

            # Solid black label placed right/below the dot
            label = f"({r},{c})"
            tx = x + offset
            ty = y + offset
            draw.text((tx, ty), label, font=font, fill=(0, 0, 0))

    return im


def main():
    ap = argparse.ArgumentParser()

    # ✅ Defaults as you requested
    ap.add_argument(
        "--input_dir",
        default="data/images",
        help="Folder containing images_000.jpg ... images_499.jpg",
    )
    ap.add_argument(
        "--output_dir",
        default="augmented_data",
        help="Output folder for augmented images",
    )

    ap.add_argument(
        "--n",
        type=int,
        default=4,
        help="Number of points per side (default 4 => 16 coords).",
    )
    ap.add_argument(
        "--prefix",
        type=int,
        default=2,
        help="Leading number in labels (default 2).",
    )
    ap.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality (default 95).",
    )

    # Visual tuning
    ap.add_argument("--font_frac", type=float, default=0.035,
                    help="Font size fraction of min dim (smaller => fits more).")
    ap.add_argument("--dot_frac", type=float, default=0.0040,
                    help="Dot radius fraction of min dim.")
    ap.add_argument("--text_offset_frac", type=float, default=0.010,
                    help="Text offset fraction of min dim.")
    ap.add_argument("--margin_x_frac", type=float, default=0.10,
                    help="Horizontal margin fraction.")
    ap.add_argument("--margin_y_frac", type=float, default=0.10,
                    help="Vertical margin fraction.")

    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.exists():
        raise SystemExit(f"Input directory not found: {in_dir.resolve()}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer your naming scheme; fallback to common image extensions
    files = sorted(list(in_dir.glob("images_*.jpg")) + list(in_dir.glob("images_*.jpeg")))
    if not files:
        files = sorted([p for p in in_dir.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])

    if not files:
        raise SystemExit(f"No images found in: {in_dir.resolve()}")

    n_written = 0
    for p in files:
        try:
            im = Image.open(p)
            im2 = draw_coords(
                im,
                n=args.n,
                prefix=args.prefix,
                margin_x_frac=args.margin_x_frac,
                margin_y_frac=args.margin_y_frac,
                font_frac=args.font_frac,
                dot_frac=args.dot_frac,
                text_offset_frac=args.text_offset_frac,
            )

            out_path = out_dir / p.name
            if out_path.suffix.lower() in {".jpg", ".jpeg"}:
                im2.save(out_path, format="JPEG", quality=args.quality, optimize=True)
            else:
                im2.save(out_path)

            n_written += 1
        except Exception as e:
            print(f"[WARN] Failed on {p.name}: {e}")

    print(f"Done. Wrote {n_written} images to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
