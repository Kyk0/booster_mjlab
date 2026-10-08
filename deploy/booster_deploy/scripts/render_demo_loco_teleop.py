"""Render a side-by-side mp4 of loco-teleop tasks in booster_deploy's MuJoCo sim.

All panels get the same velocity program and (same seed) the same arm motion.

    MUJOCO_GL=egl PYTHONPATH=<booster_assets>/src \
      python scripts/render_demo_loco_teleop.py \
      --tasks k1_loco_teleop_walk_auto \
      --labels "auto arms" --arms motion --out demo.mp4
"""

from __future__ import annotations

import argparse
import sys

import mediapy as media
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.append(".")

# (start_s, vx, vy, wz, caption)
PROGRAM = (
    (0.0, 0.0, 0.0, 0.0, "standing"),
    (5.0, 0.5, 0.0, 0.0, "walk 0.5 m/s"),
    (14.0, 1.0, 0.0, 0.0, "walk 1.0 m/s"),
    (21.0, 0.4, 0.0, 0.8, "walk + turn"),
    (27.0, 0.0, 0.0, 0.0, "stop"),
)


def command_at(t: float):
    cur = PROGRAM[0]
    for row in PROGRAM:
        if t >= row[0]:
            cur = row
    return cur[1:]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--arms", default="motion", help="upper-body mode for every panel")
    p.add_argument("--max-rate", type=float, default=15.0, help="arm target speed limit for every panel (rad/s)")
    p.add_argument("--smooth-vel", type=float, default=None, help="smooth mode peak arm speed (rad/s)")
    p.add_argument("--smooth-scale", type=float, default=None, help="smooth mode motion size (0-1)")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--out", default="loco_teleop_demo.mp4")
    args = p.parse_args()
    labels = args.labels or args.tasks

    import tasks.loco_teleop  # noqa: F401
    from booster_deploy.controllers.mujoco_controller import MujocoController
    from booster_deploy.utils.registry import get_task

    panels = []
    for task in args.tasks:
        ctrl = MujocoController(get_task(task))
        ctrl.update_vel_command = lambda: None
        ctrl.policy.cfg.upper_body_mode = args.arms
        ctrl.policy.max_upper_step = args.max_rate * ctrl.cfg.policy_dt
        if args.smooth_vel is not None:
            ctrl.policy.cfg.upper_body_smooth_max_vel = args.smooth_vel
        if args.smooth_scale is not None:
            ctrl.policy.cfg.upper_body_smooth_scale = args.smooth_scale
        ctrl.update_state()
        ctrl.start()
        ctrl.mj_model.vis.global_.offwidth = max(ctrl.mj_model.vis.global_.offwidth, args.width)
        ctrl.mj_model.vis.global_.offheight = max(ctrl.mj_model.vis.global_.offheight, args.height)
        renderer = mujoco.Renderer(ctrl.mj_model, args.height, args.width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 1
        cam.distance, cam.azimuth, cam.elevation = 2.2, 140.0, -12.0
        panels.append({"ctrl": ctrl, "r": renderer, "cam": cam, "falls": 0, "down": None,
                       "init": ctrl.mj_data.qpos.copy()})

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        small = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = small = ImageFont.load_default()

    dt = panels[0]["ctrl"].cfg.policy_dt
    every = max(1, round(1.0 / (args.fps * dt)))
    frames = []
    for i in range(int(args.seconds / dt)):
        t = i * dt
        vx, vy, wz, caption = command_at(t)
        tiles = []
        for pnl in panels:
            ctrl = pnl["ctrl"]
            if ctrl.is_running:
                cmd = ctrl.vel_command
                cmd.lin_vel_x, cmd.lin_vel_y, cmd.ang_vel_yaw = vx, vy, wz
                ctrl.update_state()
                ctrl.ctrl_step(ctrl.policy_step())
            else:  # safety stop = fell; show it for 1.5 s, then reset
                if pnl["down"] is None:
                    pnl["down"], pnl["falls"] = t, pnl["falls"] + 1
                    for _ in range(40):  # let it settle on the ground
                        mujoco.mj_step(ctrl.mj_model, ctrl.mj_data)
                elif t - pnl["down"] > 1.5:
                    mujoco.mj_resetData(ctrl.mj_model, ctrl.mj_data)
                    ctrl.mj_data.qpos[:] = pnl["init"]
                    mujoco.mj_forward(ctrl.mj_model, ctrl.mj_data)
                    ctrl.update_state()
                    ctrl.start()
                    pnl["down"] = None
            if i % every == 0:
                pnl["r"].update_scene(ctrl.mj_data, camera=pnl["cam"])
                tiles.append(pnl["r"].render())
        if i % every:
            continue
        frame = Image.fromarray(np.concatenate(tiles, axis=1))
        draw = ImageDraw.Draw(frame)
        for k, pnl in enumerate(panels):
            x0 = k * args.width + 12
            draw.text((x0, 10), labels[k], fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
            status = f"falls: {pnl['falls']}" + ("   FELL" if pnl["down"] is not None else "")
            color = (255, 80, 80) if pnl["falls"] else (180, 255, 180)
            draw.text((x0, 40), status, fill=color, font=small, stroke_width=2, stroke_fill=(0, 0, 0))
        info = f"t={t:4.1f}s  {caption}  cmd vx={vx:.1f} wz={wz:.1f}   arms: {args.arms} (same for both)"
        draw.text((12, args.height - 32), info, fill=(255, 255, 0), font=small, stroke_width=2, stroke_fill=(0, 0, 0))
        frames.append(np.asarray(frame))

    media.write_video(args.out, frames, fps=args.fps)
    print(f"[render] wrote {args.out}: {len(frames)} frames, falls " +
          ", ".join(f"{labels[k]}={p['falls']}" for k, p in enumerate(panels)))


if __name__ == "__main__":
    main()
