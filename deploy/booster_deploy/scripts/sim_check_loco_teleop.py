"""Headless MuJoCo check of a booster_deploy task (no viewer): walks with a fixed
command and reports falls, base height and velocity tracking.

    python scripts/sim_check_loco_teleop.py --task k1_loco_teleop_walk_auto --vx 0.5
"""

from __future__ import annotations

import argparse
import pkgutil
import sys

import numpy as np

sys.path.append(".")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="k1_loco_teleop_walk_auto")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--vx", type=float, default=0.5)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--wz", type=float, default=0.0)
    args = p.parse_args()

    import tasks as tasks_pkg

    for mod in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        __import__(mod.name)
    from booster_deploy.controllers.mujoco_controller import MujocoController
    from booster_deploy.utils.registry import get_task

    ctrl = MujocoController(get_task(args.task))
    ctrl.update_vel_command = lambda: None  # no stdin in headless mode
    ctrl.vel_command.lin_vel_x = args.vx
    ctrl.vel_command.lin_vel_y = args.vy
    ctrl.vel_command.ang_vel_yaw = args.wz

    ctrl.update_state()
    ctrl.start()
    steps = int(args.seconds / ctrl.cfg.policy_dt)
    heights, vx, arm_dev, pitch, roll = [], [], [], [], []
    upper_idx = ctrl.policy.upper_idx
    upper_default = ctrl.robot.default_joint_pos[upper_idx]
    fell_at = None
    for i in range(steps):
        ctrl.update_state()
        targets = ctrl.policy_step()
        ctrl.ctrl_step(targets)
        if not ctrl.is_running:
            fell_at = i * ctrl.cfg.policy_dt
            break
        heights.append(float(ctrl.mj_data.qpos[2]))
        w, x, y, z = (float(v) for v in ctrl.mj_data.qpos[3:7])
        roll.append(np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))))
        pitch.append(np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))))
        vx.append(float(ctrl.robot.data.root_lin_vel_b[0]))
        arm_dev.append(float((ctrl.robot.data.joint_pos[upper_idx] - upper_default).abs().mean()))

    settled = vx[len(vx) // 4:] or vx
    w, x, y, z = (float(v) for v in ctrl.mj_data.qpos[3:7])
    yaw_end = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    pos_end = ctrl.mj_data.qpos[:2]
    print(
        f"[sim_check] task={args.task} cmd=({args.vx},{args.vy},{args.wz}) "
        f"ran={len(heights) * ctrl.cfg.policy_dt:.1f}s fell={'at %.1fs' % fell_at if fell_at is not None else 'no'} "
        f"height mean={np.mean(heights):.4f} min={np.min(heights):.4f} "
        f"vx mean={np.mean(settled):.3f} (cmd {args.vx}) "
        f"upper-body |q-default| mean={np.mean(arm_dev):.3f} rad"
    )
    print(
        f"[sim_check] trunk pitch mean={np.mean(pitch):+.1f} deg (+ = nose down)  "
        f"roll mean={np.mean(roll):+.1f} deg  yaw drift={yaw_end:+.1f} deg  "
        f"end xy=({pos_end[0]:+.2f}, {pos_end[1]:+.2f}) m"
    )


if __name__ == "__main__":
    main()
