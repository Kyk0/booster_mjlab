"""Headless MuJoCo gait measurement: speed, cadence, step length, foot clearance.

    python scripts/gait_check_loco_teleop.py --task k1_loco_teleop_walk
"""

from __future__ import annotations

import argparse
import sys

import mujoco
import numpy as np

sys.path.append(".")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="k1_loco_teleop_walk")
    p.add_argument("--speeds", type=float, nargs="+", default=[0.3, 0.5, 0.8, 1.0])
    p.add_argument("--seconds", type=float, default=15.0)
    args = p.parse_args()

    import tasks.loco_teleop  # noqa: F401
    from booster_deploy.controllers.mujoco_controller import MujocoController
    from booster_deploy.utils.registry import get_task

    print(f"[gait_check] task={args.task} (arms at default pose)")
    for vx in args.speeds:
        ctrl = MujocoController(get_task(args.task))
        ctrl.update_vel_command = lambda: None
        ctrl.policy.cfg.upper_body_mode = "default"
        ctrl.vel_command.lin_vel_x = vx
        ctrl.update_state()
        ctrl.start()
        feet = [
            mujoco.mj_name2id(ctrl.mj_model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ("left_foot_link", "right_foot_link")
        ]
        warmup = int(5.0 / ctrl.cfg.policy_dt)
        z, base, fell = [[], []], [], None
        for i in range(warmup + int(args.seconds / ctrl.cfg.policy_dt)):
            ctrl.update_state()
            if not ctrl.is_running:
                fell = i * ctrl.cfg.policy_dt
                break
            ctrl.ctrl_step(ctrl.policy_step())
            if i >= warmup:
                for k, f in enumerate(feet):
                    z[k].append(ctrl.mj_data.xpos[f][2])
                base.append(ctrl.mj_data.qpos[:2].copy())
        if fell is not None:
            print(f"  vx={vx:.1f}: FELL at {fell:.1f}s")
            continue
        z = np.array(z)
        h0 = z.min(axis=1, keepdims=True)
        contact = z < h0 + 0.01
        steps = sum(int((np.diff(contact[k].astype(int)) == 1).sum()) for k in range(2))
        T = len(base) * ctrl.cfg.policy_dt
        dist = float(np.linalg.norm(base[-1] - base[0]))
        print(
            f"  vx={vx:.1f}: speed {dist / T:.2f} m/s  cadence {steps / T:.2f} steps/s  "
            f"step length {dist / max(steps, 1) * 100:.1f} cm  "
            f"foot clearance {(z - h0).max(axis=1).mean() * 100:.1f} cm"
        )


if __name__ == "__main__":
    main()
