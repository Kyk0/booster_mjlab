"""Headless MuJoCo test of upper-body motion while walking: falls, self-contacts.

    python scripts/smooth_check_loco_teleop.py --task k1_loco_teleop_walk_smooth
    python scripts/smooth_check_loco_teleop.py --task k1_loco_teleop_walk_auto   # fast, for contrast

Per step it counts
  physics contacts  arm/head collision geoms touching another (non-neighbour) robot
                    body in the simulation itself
  clearance misses  steps where an arm/head part (visual mesh hull) is closer than
                    1.5 cm to the body / other arm (pairs close at rest: rest - 2 cm)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.append(".")
sys.path.append(os.path.join(os.path.dirname(__file__)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="k1_loco_teleop_walk_smooth")
    p.add_argument("--speeds", type=float, nargs="+", default=[0.0, 0.5, 0.8])
    p.add_argument("--seconds", type=float, default=60.0)
    args = p.parse_args()

    import tasks.loco_teleop  # noqa: F401
    from booster_deploy.controllers.mujoco_controller import MujocoController
    from booster_deploy.utils.registry import get_task
    from build_safe_arm_motion import UPPER_BODIES, Checker

    print(f"[smooth_check] task={args.task}")
    for vx in args.speeds:
        ctrl = MujocoController(get_task(args.task))
        ctrl.update_vel_command = lambda: None
        ctrl.vel_command.lin_vel_x = vx
        pol = ctrl.policy
        meta = pol.meta
        chk = Checker(
            ctrl.mj_model and pol.controller.robot.cfg.mjcf_path.replace(
                "{BOOSTER_ASSETS_DIR}", __import__("booster_assets").BOOSTER_ASSETS_DIR.__str__()
            ),
            meta["upper_body_joint_names"],
            0.015,
        )
        chk.set_rest_pose(np.array(meta["upper_body_default_joint_pos"]), slack=0.02)

        m = ctrl.mj_model
        bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""  # noqa: E731
        upper = {b for b in range(m.nbody) if bname(b) in UPPER_BODIES}

        def neighbours(b1, b2):
            return b1 in chk.chain[b2] or b2 in chk.chain[b1]

        ctrl.update_state()
        ctrl.start()
        steps = int(args.seconds / ctrl.cfg.policy_dt)
        phys, misses, fell, speeds, upper_speed = 0, 0, None, [], []
        prev_q = None
        for i in range(steps):
            ctrl.update_state()
            if not ctrl.is_running:
                fell = i * ctrl.cfg.policy_dt
                break
            ctrl.ctrl_step(ctrl.policy_step())
            d = ctrl.mj_data
            hit = False
            for c in d.contact[: d.ncon]:
                b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                if b1 == 0 or b2 == 0 or not ((b1 in upper) or (b2 in upper)):
                    continue
                if neighbours(b1, b2) or (b1 in upper and b2 in upper and chk.side[b1] == chk.side[b2]):
                    continue
                hit = True
            phys += hit
            chk.d.qpos[:] = d.qpos
            mujoco.mj_kinematics(chk.m, chk.d)
            misses += not chk.clear()
            speeds.append(float(ctrl.robot.data.root_lin_vel_b[0]))
            q = ctrl.robot.data.joint_pos[pol.upper_idx].numpy()
            if prev_q is not None:
                upper_speed.append(np.abs(q - prev_q).max() / ctrl.cfg.policy_dt)
            prev_q = q
        n = len(speeds)
        print(
            f"  vx={vx:.1f}: ran {n * ctrl.cfg.policy_dt:.0f}s fell={'at %.1fs' % fell if fell is not None else 'no'}  "
            f"physics self-contacts {phys} steps ({100 * phys / max(n, 1):.1f}%)  "
            f"clearance misses {misses} steps ({100 * misses / max(n, 1):.1f}%)  "
            f"speed {np.mean(speeds[n // 4:]) if n else 0:.2f} m/s  "
            f"arm speed median {np.median(upper_speed):.1f} / max {np.max(upper_speed):.1f} rad/s",
            flush=True,
        )


if __name__ == "__main__":
    main()
