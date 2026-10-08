"""Front-view videos (one file per policy, no text) of a straight walk while the arms do
random big motions: the full-body walk policy (arm outputs overridden by the arm-motion
generator) vs. the arm-motion-trained loco-teleop policy. Same seed, same arm motion.

    MUJOCO_GL=egl uv run python -m booster_mjlab.scripts.render_fullbody_arms \
        --fullbody logs/rsl_rl/k1_velocity_amp_symmetric_muon/2026-09-18_21-54-10/model_2000.pt \
        --armmotion logs/rsl_rl/k1_loco_teleop_lower_armmotion_walk_muon/<run>/model_3999.pt \
        --out releases/videos/random_arms_straight
"""

from __future__ import annotations

import argparse

import mediapy as media
import mujoco
import numpy as np
import torch

from booster_mjlab.scripts.compare_fullbody_arms import (
    ARMMOTION_TASK,
    FULLBODY_TASK,
    ArmOverride,
    _add_arm_motion,
    _set_arm_mode,
)
from booster_mjlab.scripts.loco_teleop_utils import load_runner, load_task_cfgs, make_fixed_command


def build(task: str, w: int, h: int, device: str, arms: str = "motion"):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    cfg, agent = load_task_cfgs(task, play=True, dataset_root=None)
    cfg.scene.num_envs = 1
    make_fixed_command(cfg)
    cfg.events.pop("push_robot", None)
    cfg.events.pop("upper_body_teleop", None)
    cfg.events.pop("upper_body_reset", None)
    if arms == "none" and task == ARMMOTION_TASK:  # loco-teleop: hold the default arm pose
        from booster_mjlab.tasks.loco_teleop.config.k1.env_cfgs import add_upper_body_events

        add_upper_body_events(cfg, 0.0)
    _add_arm_motion(cfg)
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


def light_theme(base) -> None:
    """Viser-like look for the offscreen render: white sky, light grey checker floor."""
    ren = base._offline_renderer
    m = ren._model  # the renderer's own model copy
    con = ren.renderer._mjr_context
    for t in range(m.ntex):
        adr, w, h, nc = (int(m.tex_adr[t]), int(m.tex_width[t]), int(m.tex_height[t]), int(m.tex_nchannel[t]))
        data = m.tex_data[adr : adr + w * h * nc].reshape(-1, nc)
        if m.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            data[:, :3] = 255
        else:  # floor checker -> light greys, same pattern
            g = data[:, :3].astype(np.float32).mean(1)
            g = 150.0 + 60.0 * (g - g.min()) / max(float(g.max() - g.min()), 1e-6)
            data[:, :3] = g[:, None].astype(np.uint8)
        mujoco.mjr_uploadTexture(m, con, t)
    m.vis.map.haze = 0.0  # no band at the horizon
    m.mat_reflectance[:] = np.minimum(m.mat_reflectance, 0.05)
    m.light_diffuse[:] *= 0.55  # the floor would otherwise saturate to white
    m.light_specular[:] *= 0.3


@torch.no_grad()
def rollout(env, policy, seconds: float, vx: float, seed: int, stand_s: float = 2.0) -> tuple[list, int]:
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
        base._offline_renderer._cam.azimuth = heading + 180.0  # always from the front
        frames.append(base.render())
    return frames, falls


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fullbody", required=True)
    p.add_argument("--armmotion", required=True)
    p.add_argument("--vx", type=float, default=0.6)
    p.add_argument("--amp", type=float, default=1.0, help="arm motion amplitude scale")
    p.add_argument("--arms", choices=("motion", "none"), default="motion",
                   help="none: full-body policy moves its own arms, loco-teleop arms hold the default pose")
    p.add_argument("--seconds", type=float, default=22.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", default="1280x720")
    p.add_argument("--light", action="store_true", help="viser-like white background / light floor")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True, help="prefix; writes <out>_fullbody.mp4 and <out>_armmotion.mp4")
    args = p.parse_args()
    w, h = (int(v) for v in args.size.split("x"))

    for tag, task, ckpt in (("fullbody", FULLBODY_TASK, args.fullbody), ("armmotion", ARMMOTION_TASK, args.armmotion)):
        env, agent = build(task, w, h, args.device, args.arms)
        policy = load_runner(task, env, agent, ckpt, args.device).get_inference_policy(device=args.device)
        motion = args.arms == "motion"
        _set_arm_mode(env.unwrapped, f"motion:{args.amp}" if motion else "none")
        if args.light:
            env.unwrapped.render()  # make sure the GL context exists
            light_theme(env.unwrapped)
        if task == FULLBODY_TASK and motion:
            policy = ArmOverride(env, policy)
            policy.active = True
        frames, falls = rollout(env, policy, args.seconds, args.vx, args.seed)
        path = f"{args.out}_{tag}.mp4"
        media.write_video(path, frames, fps=round(1.0 / env.unwrapped.step_dt), crf=17)
        print(f"[render] {tag}: {falls} falls -> {path}")
        env.close()


if __name__ == "__main__":
    main()
