"""Side-by-side offscreen video of loco-teleop policies in the 50 mm heel shoe.

Each panel runs the same task (shoes on), the same velocity program and the same
random seed; only the policy differs. Front camera, no pushes, no debug drawing.

    MUJOCO_GL=egl uv run python -m booster_mjlab.scripts.render_shoe_compare \
        --policy "before (normal walk)=logs/.../model_3999.pt" \
        --policy "after (shoe warm-start)=logs/.../model_1000.pt" \
        --arms motion --out shoe_motion.mp4
"""

from __future__ import annotations

import argparse
import os

import mediapy as media
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from booster_mjlab.scripts.loco_teleop_utils import load_runner, load_task_cfgs, make_fixed_command

TASK = "Mjlab-LocoTeleop-Flat-Amp-Lower-ArmMotion-Walk-Shoe50mm-Muon-Booster-K1"
# (start_s, vx, vy, wz, caption)
PROGRAM = (
    (0.0, 0.0, 0.0, 0.0, "standing"),
    (4.0, 0.5, 0.0, 0.0, "walk 0.5 m/s"),
    (12.0, 1.0, 0.0, 0.0, "walk 1.0 m/s"),
    (19.0, 0.4, 0.0, 0.8, "walk + turn"),
    (25.0, 0.0, 0.0, 0.0, "stop"),
)


def build(args, arms: str, w: int, h: int):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    from booster_mjlab.tasks.loco_teleop.config.k1.env_cfgs import add_upper_body_events

    cfg, agent = load_task_cfgs(args.task, play=True, dataset_root=None)
    cfg.scene.num_envs = 1
    make_fixed_command(cfg)
    cfg.events.pop("push_robot", None)
    if arms == "still":
        add_upper_body_events(cfg, 0.0)  # hold the default pose
    else:  # full-amplitude arm motion from the first step (no curriculum ramp)
        for name in ("upper_body_reset", "upper_body_teleop"):
            cfg.events[name].params["amp_scale"] = 1.0
    for term in cfg.commands.values():
        term.debug_vis = False
    for sensor in cfg.scene.sensors or ():
        if hasattr(sensor, "debug_vis"):
            sensor.debug_vis = False
    cfg.viewer.width, cfg.viewer.height = w, h
    cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 2.0, -8.0, 180.0
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array"),
        clip_actions=agent.clip_actions,
    )
    return env, agent


@torch.no_grad()
def rollout(env, policy, seconds: float, seed: int, program=PROGRAM) -> tuple[list[np.ndarray], list[int]]:
    base = env.unwrapped
    torch.manual_seed(seed)
    base.seed(seed)
    obs, _ = env.reset()
    twist = base.command_manager.get_term("twist")
    robot = base.scene["robot"]
    frames, falls, n = [], [], 0
    heading = None
    for i in range(int(seconds / base.step_dt)):
        t = i * base.step_dt
        cmd = [row[1:4] for row in program if t >= row[0]][-1]
        twist.vel_command_b[:] = torch.tensor(cmd, device=base.device)
        obs, _, done, extras = env.step(policy(obs))
        n += int(done.bool().item() and not extras["time_outs"].bool().item())
        # Front view: camera azimuth follows the trunk heading (smoothed).
        w_, x_, y_, z_ = robot.data.root_link_quat_w[0].tolist()
        yaw = float(np.degrees(np.arctan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))))
        if heading is None:
            heading = yaw
        heading += 0.05 * ((yaw - heading + 180.0) % 360.0 - 180.0)
        if base._offline_renderer is not None:
            base._offline_renderer._cam.azimuth = heading + 180.0
        frames.append(base.render())
        falls.append(n)
    return frames, falls


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", action="append", required=True, help="LABEL=CHECKPOINT (repeat)")
    p.add_argument("--task", default=TASK)
    p.add_argument("--arms", choices=("motion", "still"), default="motion")
    p.add_argument("--seconds", type=float, default=28.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--panel", default="960x720")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True)
    p.add_argument("--no-text", action="store_true", help="plain frames, no labels/captions")
    p.add_argument("--light", action="store_true", help="viser-like white background / light floor")
    p.add_argument("--straight", type=float, default=None, help="straight walk at this vx (after 2 s standing) instead of PROGRAM")
    p.add_argument("--separate", action="store_true",
                   help="one video per policy: <out>_<LABEL-before-colon>.mp4 instead of side by side")
    args = p.parse_args()
    w, h = (int(v) for v in args.panel.split("x"))
    policies = [s.split("=", 1) for s in args.policy]

    panels = []
    for label, ckpt in policies:
        env, agent = build(args, args.arms, w, h)
        policy = load_runner(args.task, env, agent, ckpt, args.device).get_inference_policy(device=args.device)
        if args.light:
            from booster_mjlab.scripts.render_fullbody_arms import light_theme

            env.unwrapped.render()
            light_theme(env.unwrapped)
        program = PROGRAM if args.straight is None else ((0.0, 0.0, 0.0, 0.0, "standing"), (2.0, args.straight, 0.0, 0.0, "walk"))
        frames, falls = rollout(env, policy, args.seconds, args.seed, program)
        panels.append((label, frames, falls))
        env.close()
        print(f"[render] {label}: {falls[-1]} falls")

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
        small = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = small = ImageFont.load_default()
    fps = round(len(panels[0][1]) / args.seconds)
    arms_txt = "arms: big random motion" if args.arms == "motion" else "arms: still"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.separate:
        for label, frames, falls in panels:
            tag = label.split(":")[0].strip().lower().replace(" ", "_")
            path = f"{os.path.splitext(args.out)[0]}_{tag}.mp4"
            write_video(path, [(label, frames, falls)], w, h, fps, arms_txt, font, small, args.no_text)
            print(f"[render] wrote {path}")
        return
    write_video(args.out, panels, w, h, fps, arms_txt, font, small, args.no_text)
    print(f"[render] wrote {args.out}")


def write_video(path, panels, w, h, fps, arms_txt, font, small, no_text=False) -> None:
    with media.VideoWriter(path, shape=(h, w * len(panels)), fps=fps, crf=17) as vw:
        if no_text:
            for i in range(len(panels[0][1])):
                vw.add_image(np.concatenate([pnl[1][i] for pnl in panels], axis=1))
            return
        for i in range(len(panels[0][1])):
            img = Image.fromarray(np.concatenate([pnl[1][i] for pnl in panels], axis=1))
            d = ImageDraw.Draw(img)
            for k, (label, _, falls) in enumerate(panels):
                x = k * w + 20
                d.text((x, 16), label, fill=(20, 20, 20), font=font)
                d.text((x, 56), f"falls: {falls[i]}", fill=(200, 30, 30) if falls[i] else (30, 140, 30), font=small)
            t = i / fps
            cap = [row[4] for row in PROGRAM if t >= row[0]][-1]
            d.text((20, h - 44), f"t={t:4.1f}s  {cap}   |   50 mm heel shoe, 350 g   |   {arms_txt}",
                   fill=(20, 20, 20), font=small)
            vw.add_image(np.asarray(img))


if __name__ == "__main__":
    main()
