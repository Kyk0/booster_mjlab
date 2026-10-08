"""Stitch 4 equal-length videos into a 2x2 grid with small column/row labels.

    python -m booster_mjlab.scripts.make_grid_video \
        --tl a.mp4 --tr b.mp4 --bl c.mp4 --br d.mp4 \
        --cols "BEFORE: standard walking policy" "AFTER: legs RL + arms free for teleop" \
        --rows "arms: no randomization" "arms: random motion (teleop stand-in)" \
        --out grid.mp4
"""

from __future__ import annotations

import argparse

import mediapy as media
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    p = argparse.ArgumentParser()
    for k in ("tl", "tr", "bl", "br"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--cols", nargs=2, required=True)
    p.add_argument("--rows", nargs=2, required=True)
    p.add_argument("--cell", default="960x540")
    p.add_argument("--title", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    w, h = (int(v) for v in a.cell.split("x"))
    readers = [media.VideoReader(x) for x in (a.tl, a.tr, a.bl, a.br)]
    for r in readers:
        r.__enter__()
    fps = readers[0].metadata.fps
    n = min(r.num_images for r in readers)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        small = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = small = ImageFont.load_default()

    def label(d, xy, text, f):
        x, y = xy
        box = d.textbbox((x, y), text, font=f)
        d.rectangle((box[0] - 8, box[1] - 5, box[2] + 8, box[3] + 5), fill=(255, 255, 255, 215))
        d.text((x, y), text, fill=(20, 20, 20), font=f)

    streams = [iter(r) for r in readers]
    with media.VideoWriter(a.out, shape=(2 * h, 2 * w), fps=fps, crf=18) as vw:
        for _ in range(n):
            cells = [np.asarray(Image.fromarray(next(s_)).resize((w, h), Image.LANCZOS)) for s_ in streams]
            img = Image.fromarray(np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])])).convert("RGBA")
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(overlay)
            for c in range(2):
                label(d, (c * w + 16, 14), a.cols[c], font)
            for r in range(2):
                label(d, (16, r * h + h - 44), a.rows[r], small)
            if a.title:
                tb = d.textbbox((0, 0), a.title, font=font)
                label(d, (w - (tb[2] - tb[0]) // 2, h - 22), a.title, font)
            d.line((w, 0, w, 2 * h), fill=(255, 255, 255, 255), width=3)
            d.line((0, h, 2 * w, h), fill=(255, 255, 255, 255), width=3)
            vw.add_image(np.asarray(Image.alpha_composite(img, overlay).convert("RGB")))
    for r in readers:
        r.__exit__(None, None, None)
    print(f"[grid] wrote {a.out} ({n} frames, {fps:.0f} fps, {2 * w}x{2 * h})")


if __name__ == "__main__":
    main()
