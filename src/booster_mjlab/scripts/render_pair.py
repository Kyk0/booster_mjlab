"""Offscreen MuJoCo videos (one file per policy, no text) of a straight walk, camera
following the robot from a slight side angle. Any task/checkpoint pairs; the policy
drives everything itself (no arm override).

    MUJOCO_GL=egl uv run python -m booster_mjlab.scripts.render_pair \
        --policy "flat=Mjlab-Velocity-Flat-Amp-DA-Shoe50mm-Muon-Booster-K1@logs/.../model_2000.pt" \
        --policy "shoe=Mjlab-Velocity-Flat-Amp-DA-Shoe50mm-Muon-Booster-K1@logs/.../model_1400.pt" \
        --light --side-angle 35 --out releases/videos/fullbody_shoes
"""

from __future__ import annotations

import argparse

import mediapy as media
import numpy as np
import torch

from booster_mjlab.scripts.loco_teleop_utils import load_runner, load_task_cfgs, make_fixed_command
from booster_mjlab.scripts.render_fullbody_arms import light_theme


def build(task: str, w: int, h: int, device: str):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    cfg, agent = load_task_cfgs(task, play=True, dataset_root=None)
    cfg.scene.num_envs = 1
    make_fixed_command(cfg)
    cfg.events.pop("push_robot", None)
    for term in cfg.commands.values():
        term.debug_vis = False
    for sensor in cfg.scene.sensors or ():
        if hasattr(sensor, "debug_vis"):
            sensor.debug_vis = False
    cfg.viewer.width, cfg.viewer.height = w, h
    cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 2.0, -8.0, 180.0
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array"), clip_actions=agent.clip_actions
    )
    return env, agent


@torch.no_grad()
def rollout(env, policy, seconds, vx, seed, side_angle, stand_s=2.0):
    base = env.unwrapped
    torch.manual_seed(seed)
    base.seed(seed)
    obs, _ = env.reset()
    twist, robot = base.command_manager.get_term("twist"), base.scene["robot"]
    frames, falls, heading = [], 0, None
    for i in range(int(seconds / base.step_dt)):
        twist.vel_command_b[:] = torch.tensor([vx if i * base.step_dt >= stand_s else 0.0, 0.0, 0.0], device=base.device)
        obs, _, done, extras = env.step(policy(obs))
        falls += int(done.bool().item() and not extras["time_outs"].bool().item())
        w_, x_, y_, z_ = robot.data.root_link_quat_w[0].tolist()
        yaw = float(np.degrees(np.arctan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))))
        heading = yaw if heading is None else heading + 0.05 * ((yaw - heading + 180.0) % 360.0 - 180.0)
        base._offline_renderer._cam.azimuth = heading + 180.0 + side_angle  # front, turned a bit to the side
        frames.append(base.render())
    return frames, falls


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", action="append", required=True, help="TAG=TASK@CHECKPOINT (repeat)")
    p.add_argument("--vx", type=float, default=0.6)
    p.add_argument("--seconds", type=float, default=22.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", default="1280x720")
    p.add_argument("--side-angle", type=float, default=35.0, help="degrees off the front view")
    p.add_argument("--light", action="store_true")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True, help="prefix; writes <out>_<TAG>.mp4")
    args = p.parse_args()
    w, h = (int(v) for v in args.size.split("x"))

    for spec in args.policy:
        tag, rest = spec.split("=", 1)
        task, ckpt = rest.split("@", 1)
        env, agent = build(task, w, h, args.device)
        policy = load_runner(task, env, agent, ckpt, args.device).get_inference_policy(device=args.device)
        if args.light:
            env.unwrapped.render()
            light_theme(env.unwrapped)
        frames, falls = rollout(env, policy, args.seconds, args.vx, args.seed, args.side_angle)
        path = f"{args.out}_{tag}.mp4"
        media.write_video(path, frames, fps=round(1.0 / env.unwrapped.step_dt), crf=17)
        print(f"[render] {tag}: {falls} falls -> {path}")
        env.close()


if __name__ == "__main__":
    main()
