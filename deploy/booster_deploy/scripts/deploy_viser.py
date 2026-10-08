"""Run a k1_loco_teleop task in booster_deploy's MuJoCo sim and view/drive it in the
browser (viser) instead of the native MuJoCo window.

Physics, PD control and the policy are exactly booster_deploy's MujocoController;
only rendering and input differ. Run with the booster-mjlab venv (has viser/mjviser):

    cd <booster_deploy>
    PYTHONPATH=<booster_assets>/src \
      ~/booster-mjlab/.venv/bin/python scripts/deploy_viser.py --port 8090

then open http://localhost:8090 (ssh -L 8090:localhost:8090 ... from a laptop).
"""

from __future__ import annotations

import argparse
import sys
import time

import mujoco
import numpy as np
import torch
import viser
from mjviser import ViserMujocoScene

sys.path.append(".")

PRESETS = {
    "default": {},
    "arms forward": {"Left_Shoulder_Pitch": -1.5, "Right_Shoulder_Pitch": -1.5},
    "arms up": {"Left_Shoulder_Pitch": -3.0, "Right_Shoulder_Pitch": -3.0},
    "T-pose": {"Left_Shoulder_Roll": 0.0, "Right_Shoulder_Roll": 0.0},
    "right arm up": {"Right_Shoulder_Pitch": -3.0, "Right_Elbow_Pitch": -0.5},
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="k1_loco_teleop_walk_auto")
    p.add_argument("--arms", default=None, help="initial arm mode (motion/random/manual/default/udp)")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--vx", type=float, default=0.8)
    args = p.parse_args()

    import tasks.loco_teleop  # noqa: F401  (registers the k1_loco_teleop tasks)
    from booster_deploy.controllers.mujoco_controller import MujocoController
    from booster_deploy.utils.registry import get_task

    ctrl = MujocoController(get_task(args.task))
    ctrl.update_vel_command = lambda: None  # commands come from the GUI
    policy = ctrl.policy
    src = policy.upper_source
    init_qpos = ctrl.mj_data.qpos.copy()

    server = viser.ViserServer(port=args.port, label="k1_loco_teleop")
    scene = ViserMujocoScene(server, ctrl.mj_model, num_envs=1)

    state = {"reset": False, "falls": 0}
    with server.gui.add_folder("Walk"):
        vx = server.gui.add_slider("vx (m/s)", -1.0, 1.5, 0.05, args.vx)
        vy = server.gui.add_slider("vy (m/s)", -1.0, 1.0, 0.05, 0.0)
        wz = server.gui.add_slider("yaw (rad/s)", -1.5, 1.5, 0.05, 0.0)
        stop = server.gui.add_button("Stop")
        reset = server.gui.add_button("Reset robot")
    with server.gui.add_folder("Arms"):
        mode = server.gui.add_dropdown(
            "mode", ("smooth", "motion", "random", "manual", "default", "udp"), initial_value=policy.cfg.upper_body_mode
            if policy.cfg.upper_body_mode in ("smooth", "motion", "random", "manual", "default", "udp") else "random"
        )
        pose = server.gui.add_dropdown("pose (manual)", tuple(PRESETS), initial_value="default")
        amp = server.gui.add_slider("random amplitude (rad)", 0.0, 1.0, 0.05, policy.cfg.upper_body_amplitude)
        full = server.gui.add_slider("random full-range prob", 0.0, 1.0, 0.05, policy.cfg.upper_body_full_range_prob)
        period = server.gui.add_slider("random period (s)", 0.3, 3.0, 0.1, policy.cfg.upper_body_period_s)
        mscale = server.gui.add_slider("motion amplitude (1 = full range)", 0.0, 1.0, 0.05, policy.cfg.upper_body_motion_scale)
    status = server.gui.add_markdown("")
    scene.create_scene_gui()

    @stop.on_click
    def _(_):
        vx.value = vy.value = wz.value = 0.0

    @reset.on_click
    def _(_):
        state["reset"] = True

    def apply_pose() -> None:
        names = policy.meta["upper_body_joint_names"]
        q = policy.upper_default.clone()
        for name, v in PRESETS[pose.value].items():
            q[names.index(name)] = v
        src.set_manual(q)

    @mode.on_update
    def _(_):
        policy.cfg.upper_body_mode = mode.value
        if mode.value == "manual":
            apply_pose()
        if mode.value == "udp":
            src.start()

    @pose.on_update
    def _(_):
        mode.value = "manual"
        policy.cfg.upper_body_mode = "manual"
        apply_pose()

    @amp.on_update
    def _(_):
        policy.cfg.upper_body_amplitude = amp.value

    @full.on_update
    def _(_):
        policy.cfg.upper_body_full_range_prob = full.value

    @period.on_update
    def _(_):
        policy.cfg.upper_body_period_s = period.value

    @mscale.on_update
    def _(_):
        policy.cfg.upper_body_motion_scale = mscale.value

    def do_reset() -> None:
        mujoco.mj_resetData(ctrl.mj_model, ctrl.mj_data)
        ctrl.mj_data.qpos[:] = init_qpos
        mujoco.mj_forward(ctrl.mj_model, ctrl.mj_data)
        ctrl.update_state()
        ctrl.start()

    if args.arms:
        mode.value = args.arms
    policy.cfg.upper_body_mode = mode.value
    do_reset()
    print(f"[deploy_viser] {args.task}: open http://localhost:{args.port}")

    dt = ctrl.cfg.policy_dt
    next_t = time.perf_counter()
    fell_at = None
    step = 0
    speeds: list[float] = []
    while True:
        if state["reset"]:
            state["reset"] = False
            do_reset()
            fell_at = None
        if not ctrl.is_running:  # safety fallback tripped -> auto reset after 1 s
            if fell_at is None:
                fell_at = time.perf_counter()
                state["falls"] += 1
            elif time.perf_counter() - fell_at > 1.0:
                do_reset()
                fell_at = None
        else:
            cmd = ctrl.vel_command
            cmd.lin_vel_x, cmd.lin_vel_y, cmd.ang_vel_yaw = vx.value, vy.value, wz.value
            ctrl.update_state()
            ctrl.ctrl_step(ctrl.policy_step())
            speeds.append(float(ctrl.robot.data.root_lin_vel_b[0]))
            speeds = speeds[-50:]
        step += 1
        if step % 2 == 0:
            scene.update_from_mjdata(ctrl.mj_data)
        if step % 25 == 0:
            status.content = (
                f"**speed** {np.mean(speeds) if speeds else 0:.2f} m/s · "
                f"**height** {ctrl.mj_data.qpos[2]:.2f} m · **falls** {state['falls']}"
            )
        next_t += dt
        time.sleep(max(0.0, next_t - time.perf_counter()))
        if time.perf_counter() - next_t > 0.5:  # fell behind (e.g. slow machine)
            next_t = time.perf_counter()


if __name__ == "__main__":
    main()
