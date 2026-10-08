"""Put two equal-length videos side by side with a title on top of each.

    python -m booster_mjlab.scripts.make_sbs_video --left a.mp4 --right b.mp4 \
        --left-title "Standard walking policy" --right-title "Ours" --out sbs.mp4
"""

from __future__ import annotations

import argparse

import mediapy as media
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--left-title", required=True)
    p.add_argument("--right-title", required=True)
    p.add_argument("--left-sub", default="")
    p.add_argument("--right-sub", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--font-size", type=int, default=34)
    p.add_argument("--crop-width", type=int, default=None, help="center-crop each panel to this width")
    p.add_argument("--vertical", action="store_true", help="stack top/bottom (X-friendly) instead of side by side")
    p.add_argument("--start", type=float, default=0.0, help="seconds to skip")
    p.add_argument("--duration", type=float, default=None, help="seconds to keep")
    a = p.parse_args()
    rl, rr = media.VideoReader(a.left), media.VideoReader(a.right)
    rl.__enter__(), rr.__enter__()
    h, w0 = rl.shape
    w = a.crop_width or w0
    x0 = (w0 - w) // 2
    fps, n = rl.metadata.fps, min(rl.num_images, rr.num_images)
    try:
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", a.font_size)
        sub = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        big = sub = ImageFont.load_default()
    first = int(a.start * fps)
    last = n if a.duration is None else min(n, first + int(a.duration * fps))
    shape = (2 * h, w) if a.vertical else (h, 2 * w)
    with media.VideoWriter(a.out, shape=shape, fps=fps, crf=17) as vw:
        for i, (fl, fr) in enumerate(zip(rl, rr)):
            if i >= last:
                break
            if i < first:
                continue
            fl, fr = fl[:, x0 : x0 + w], fr[:, x0 : x0 + w]
            img = Image.fromarray(np.vstack([fl, fr]) if a.vertical else np.hstack([fl, fr]))
            d = ImageDraw.Draw(img)
            for k, (title, s) in enumerate(((a.left_title, a.left_sub), (a.right_title, a.right_sub))):
                ox, oy = (0, k * h) if a.vertical else (k * w, 0)
                cx = ox + w // 2
                tw = d.textlength(title, font=big)
                d.text((cx - tw / 2, oy + 22), title, fill=(25, 25, 25), font=big)
                if s:
                    sw = d.textlength(s, font=sub)
                    d.text((cx - sw / 2, oy + 66), s, fill=(90, 90, 90), font=sub)
            if a.vertical:
                d.line((0, h, w, h), fill=(200, 200, 200), width=3)
            else:
                d.line((w, 0, w, h), fill=(200, 200, 200), width=3)
            vw.add_image(np.asarray(img))
    n = last - first
    rl.__exit__(None, None, None), rr.__exit__(None, None, None)
    print(f"[sbs] wrote {a.out} ({n} frames, {shape[1]}x{shape[0]})")


if __name__ == "__main__":
    main()
