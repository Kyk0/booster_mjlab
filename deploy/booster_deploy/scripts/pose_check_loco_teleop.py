"""Headless MuJoCo test: walk at several speeds while holding extreme arm poses.

    python scripts/pose_check_loco_teleop.py --task k1_loco_teleop_full

Prints, per speed and pose, whether the robot fell and its mean trunk pitch.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.append(".")

POSES = {
    "default": {},
    "arms forward": {"Left_Shoulder_Pitch": -1.5, "Right_Shoulder_Pitch": -1.5},
    "arms up": {"Left_Shoulder_Pitch": -3.0, "Right_Shoulder_Pitch": -3.0},
    "T-pose": {"Left_Shoulder_Roll": 0.0, "Right_Shoulder_Roll": 0.0},
    "right arm up": {"Right_Shoulder_Pitch": -3.0, "Right_Elbow_Pitch": -0.5},
    "arms back": {"Left_Shoulder_Pitch": 1.0, "Right_Shoulder_Pitch": 1.0},
    "head turned": {"Head_Yaw": 0.9, "Head_Pitch": 0.6},
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="k1_loco_teleop_full")
    p.add_argument("--speeds", type=float, nargs="+", default=[0.0, 0.5, 0.8, 1.0])
    p.add_argument("--seconds", type=float, default=15.0)
    args = p.parse_args()

    import tasks.loco_teleop  # noqa: F401
    from booster_deploy.controllers.mujoco_controller import MujocoController
    from booster_deploy.utils.registry import get_task

    print(f"[pose_check] task={args.task}")
    for vx in args.speeds:
        row = []
        for name, pose in POSES.items():
            ctrl = MujocoController(get_task(args.task))
            ctrl.update_vel_command = lambda: None
            pol = ctrl.policy
            pol.cfg.upper_body_mode = "manual"
            q = pol.upper_default.clone()
            names = pol.meta["upper_body_joint_names"]
            for joint, value in pose.items():
                q[names.index(joint)] = value
            pol.upper_source.set_manual(q)
            ctrl.vel_command.lin_vel_x = vx
            ctrl.update_state()
            ctrl.start()
            fell, pitch = None, []
            for i in range(int(args.seconds / ctrl.cfg.policy_dt)):
                ctrl.update_state()
                if not ctrl.is_running:
                    fell = i * ctrl.cfg.policy_dt
                    break
                ctrl.ctrl_step(ctrl.policy_step())
                w, x, y, z = ctrl.mj_data.qpos[3:7]
                pitch.append(np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))))
            row.append(
                f"{name}: " + (f"FELL {fell:.1f}s" if fell is not None else f"ok {np.mean(pitch):+.0f}°")
            )
        print(f"vx={vx}: " + " | ".join(row), flush=True)


if __name__ == "__main__":
    main()
