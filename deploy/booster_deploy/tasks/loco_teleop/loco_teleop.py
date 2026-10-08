"""Loco-teleop deployment: legs from the RL policy, head/arms from an external source.

Policy and observation layout match booster-mjlab ``Mjlab-LocoTeleop-*`` tasks; all
numbers (joint order, default pose, PD gains, action scale, upper-body limits) come
from ``deploy_meta.json`` written by ``booster_mjlab.scripts.export_loco_teleop_deploy``.

Observation (65): ang_vel_b(3), projected_gravity(3), joint_pos - default (22),
joint_vel (22), last_action (12), command vx/vy/wz (3). Joint order = training order.
"""

from __future__ import annotations

import json
import math
import os
import socket
import threading
import time
from dataclasses import MISSING

import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg,
    PolicyCfg,
    VelocityCommandCfg,
)
from booster_deploy.robots.booster import K1_CFG
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.isaaclab.configclass import configclass


# Training motor torque limits (Nm) in robot joint order: head, arms, legs.
K1_TRAIN_EFFORT_LIMIT = (
    [6.0, 6.0]
    + [14.0] * 8
    + [68.0, 76.0, 38.3, 112.0, 38.3, 38.3] * 2
)


def _norm_joint_name(name: str) -> str:
    # Robot names carry ordering prefixes ("AAHead_yaw", "ALeft_Shoulder_Pitch").
    return name.lower().lstrip("a")


def _load_meta(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "deploy_meta.json")) as f:
        return json.load(f)


class UpperBodySource:
    """Upper-body joint targets (training upper-body order, absolute radians).

    modes:
      default  hold the training default pose
      udp      latest JSON packet {"q": [..]} on ``port``; falls back to the default
               pose if no packet arrives for ``timeout_s``
      random   seeded random pose every ``period_s`` (sim tests): with probability
               ``full_range_prob`` uniform over the joint limits, else
               default ± amplitude
      motion   big continuous motions like in training (``upper_body_motion``):
               every 1-4 s pick shoulder circles / per-joint sweeps / random jumps
               over the whole joint range, amplitude x ``motion_scale``
      smooth   self-collision-free smooth motion: min-jerk moves between poses of
               ``safe_arm_motion.json`` along transitions checked offline
               (scripts/build_safe_arm_motion.py), peak joint speed
               ``smooth_max_vel``, short pauses in between
      manual   whatever was last passed to ``set_manual`` (in-process GUIs)
    """

    def __init__(
        self,
        cfg: LocoTeleopPolicyCfg,
        default: torch.Tensor,
        dt: float,
        lo: torch.Tensor,
        hi: torch.Tensor,
    ):
        self.cfg = cfg
        self.default = default.clone()
        self.lo, self.hi = lo, hi
        self.dt = dt
        self._target = default.clone()
        self._last_rx = 0.0
        self._lock = threading.Lock()
        self._steps = 0
        self._gen = torch.Generator().manual_seed(cfg.upper_body_seed)
        self._sock = None

    def start(self) -> None:
        if self.cfg.upper_body_mode == "udp" and self._sock is None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind(("0.0.0.0", self.cfg.upper_body_udp_port))
            threading.Thread(target=self._rx_loop, daemon=True).start()
            print(f"[loco_teleop] upper body: listening on UDP :{self.cfg.upper_body_udp_port}")
        else:
            print(f"[loco_teleop] upper body mode: {self.cfg.upper_body_mode}")

    def _rx_loop(self) -> None:
        n = len(self.default)
        while True:
            data, _ = self._sock.recvfrom(4096)
            try:
                q = json.loads(data.decode())["q"]
                if len(q) != n:
                    continue
                with self._lock:
                    self._target = torch.tensor(q, dtype=torch.float32)
                    self._last_rx = time.monotonic()
            except (ValueError, KeyError, TypeError):
                continue

    def _motion(self) -> torch.Tensor:
        g, n, dt = self._gen, len(self.default), self.dt
        rnd = lambda *s: torch.rand(*s, generator=g)  # noqa: E731
        lo, hi = self.lo, self.hi
        m = getattr(self, "_m", None)
        if m is None or m["t"] >= m["seg"]:
            mode = int(torch.multinomial(torch.tensor([0.25, 0.5, 0.25]), 1, generator=g))
            amp = self.cfg.upper_body_motion_scale * (0.3 + 0.7 * rnd(n)) * (hi - lo) / 2
            w = 2 * math.pi * (0.2 + 1.3 * rnd(n))
            phase = 2 * math.pi * rnd(n)
            if mode == 1:  # circles: shoulder roll follows pitch by a quarter period
                for a, b in self._circle_pairs:
                    w[b], phase[b] = w[a], phase[a] + math.pi / 2
            m = self._m = {
                "mode": mode, "t": 0.0, "seg": 1.0 + 3.0 * float(rnd(1)),
                "amp": amp, "w": w, "phase": phase,
                "center": lo + amp + rnd(n) * (hi - lo - 2 * amp),
                "jump": lo + rnd(n) * (hi - lo), "jump_t": 0.3 + 1.2 * float(rnd(1)),
            }
        m["t"] += dt
        if m["mode"] == 0:
            m["jump_t"] -= dt
            if m["jump_t"] <= 0:
                m["jump"] = lo + rnd(n) * (hi - lo)
                m["jump_t"] = 0.3 + 1.2 * float(rnd(1))
            return m["jump"].clone()
        return m["center"] + m["amp"] * torch.sin(m["w"] * m["t"] + m["phase"])

    def _smooth(self) -> torch.Tensor:
        lib = self._safe_lib
        s = getattr(self, "_s", None)
        if s is None:
            s = self._s = {"cur": 0, "prev": -1, "q0": lib["poses"][0], "q1": lib["poses"][0],
                           "T": 0.0, "t": 0.0, "hold": 0.5}
        s["t"] += self.dt
        if s["t"] >= s["T"] + s["hold"]:
            nbrs = lib["edges"][s["cur"]]
            choices = [j for j in nbrs if j != s["prev"]] or nbrs or [s["cur"]]
            nxt = choices[int(torch.randint(len(choices), (1,), generator=self._gen))]
            q0, q1 = lib["poses"][s["cur"]], lib["poses"][nxt]
            # Min-jerk peak speed is 1.875 * distance / T.
            dist = self.cfg.upper_body_smooth_scale * float((q1 - q0).abs().max())
            T = max(0.6, 1.875 * dist / self.cfg.upper_body_smooth_max_vel)
            hold = 0.3 + 0.7 * float(torch.rand(1, generator=self._gen))
            s.update(prev=s["cur"], cur=nxt, q0=q0, q1=q1, T=T, t=0.0, hold=hold)
        tau = min(1.0, s["t"] / s["T"]) if s["T"] > 0 else 1.0
        blend = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        q = s["q0"] + (s["q1"] - s["q0"]) * blend
        # Optionally shrink the motion towards the rest pose (smaller, calmer moves).
        return self.default + self.cfg.upper_body_smooth_scale * (q - self.default)

    def set_manual(self, q: torch.Tensor) -> None:
        with self._lock:
            self._target = q.clone()

    def get(self) -> torch.Tensor:
        mode = self.cfg.upper_body_mode
        if mode == "motion":
            return self._motion().clamp(self.lo, self.hi)
        if mode == "smooth":
            return self._smooth()
        if mode == "manual":
            with self._lock:
                return self._target.clone()
        if mode == "random":
            period = max(1, round(self.cfg.upper_body_period_s / self.dt))
            if self._steps % period == 0:
                n = len(self.default)
                u = torch.rand(n, generator=self._gen) * 2 - 1
                self._target = self.default + self.cfg.upper_body_amplitude * u
                if torch.rand(1, generator=self._gen).item() < self.cfg.upper_body_full_range_prob:
                    self._target = self.lo + torch.rand(n, generator=self._gen) * (self.hi - self.lo)
            self._steps += 1
            return self._target.clone()
        if mode == "udp":
            with self._lock:
                fresh = time.monotonic() - self._last_rx < self.cfg.upper_body_timeout_s
                return self._target.clone() if fresh else self.default.clone()
        return self.default.clone()


class LocoTeleopPolicy(Policy):
    def __init__(self, cfg: LocoTeleopPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot
        model_dir = cfg.model_dir
        if not os.path.isabs(model_dir):
            model_dir = os.path.join(self.task_path, model_dir)
        meta = _load_meta(model_dir)
        self.meta = meta

        self._model = torch.jit.load(os.path.join(model_dir, "policy.pt"), map_location="cpu")
        self._model.eval()

        robot_names = [_norm_joint_name(n) for n in self.robot.cfg.joint_names]

        def to_robot_idx(names: list[str]) -> torch.Tensor:
            return torch.tensor([robot_names.index(_norm_joint_name(n)) for n in names], dtype=torch.long)

        # Training joint order <-> robot joint order.
        self.obs_joint_idx = to_robot_idx(meta["all_joint_names"])
        self.leg_idx = to_robot_idx(meta["joint_names"])
        self.upper_idx = to_robot_idx(meta["upper_body_joint_names"])

        f32 = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
        self.obs_default = f32(meta["all_default_joint_pos"])
        self.leg_default = f32(meta["default_joint_pos"])
        self.action_scale = f32(meta["action_scale"])
        upper_limits = f32(meta["upper_body_joint_limits"]).reshape(-1, 2)  # empty for full-body policies
        self.upper_lo, self.upper_hi = upper_limits[:, 0], upper_limits[:, 1]
        self.upper_default = f32(meta["upper_body_default_joint_pos"])

        dt = controller.cfg.policy_dt
        self.upper_source = UpperBodySource(
            cfg, self.upper_default, dt, self.upper_lo, self.upper_hi
        )
        upper_names = meta["upper_body_joint_names"]
        lib_path = os.path.join(os.path.dirname(__file__), "safe_arm_motion.json")
        if upper_names and os.path.isfile(lib_path):
            with open(lib_path) as f:
                lib = json.load(f)
            assert lib["joint_names"] == upper_names, "safe_arm_motion.json joint order mismatch"
            self.upper_source._safe_lib = {
                "poses": [torch.tensor(q, dtype=torch.float32) for q in lib["poses"]],
                "edges": lib["edges"],
            }
        elif upper_names and cfg.upper_body_mode == "smooth":
            raise FileNotFoundError(f"{lib_path} missing: run scripts/build_safe_arm_motion.py")
        self.upper_source._circle_pairs = [
            (upper_names.index(f"{side}_Shoulder_Pitch"), upper_names.index(f"{side}_Shoulder_Roll"))
            for side in ("Left", "Right")
            if f"{side}_Shoulder_Pitch" in upper_names
        ]
        self.max_upper_step = cfg.upper_body_max_rate * dt

        self.last_action = torch.zeros(len(self.leg_idx))
        self.heading_target: float | None = None
        self.upper_cmd = self.upper_default.clone()
        self.start_pos: torch.Tensor | None = None
        self.steps = 0

    def reset(self) -> None:
        self.last_action.zero_()
        self.heading_target = None
        self.upper_cmd = self.upper_default.clone()
        self.start_pos = None
        self.steps = 0
        self.upper_source.start()

    def compute_observation(self) -> torch.Tensor:
        data = self.robot.data
        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        projected_gravity = lab_math.quat_apply_inverse(data.root_quat_w, gravity_w)

        if self.cfg.enable_safety_fallback and projected_gravity[2] > -0.5:
            print("\n[loco_teleop] Falling detected, stopping policy.")
            self.controller.stop()

        cmd = self.controller.vel_command
        wz = self._heading_hold(data.root_quat_w, cmd.ang_vel_yaw)
        command = torch.tensor([cmd.lin_vel_x, cmd.lin_vel_y, wz], dtype=torch.float32)
        q = data.joint_pos[self.obs_joint_idx]
        dq = data.joint_vel[self.obs_joint_idx]
        return torch.cat(
            [
                data.root_ang_vel_b,
                projected_gravity,
                q - self.obs_default,
                dq,
                self.last_action,
                command,
            ]
        )

    def _heading_hold(self, quat_wxyz: torch.Tensor, wz_cmd: float) -> float:
        """Yaw-rate command; holds the current heading while no turn is commanded.

        Same law as the training heading command: wz = k * wrap(target - yaw).
        The policy only sees yaw *rate*, so without this small errors add up and
        the robot drifts off course.
        """
        w, x, y, z = (float(v) for v in quat_wxyz)
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        if not self.cfg.heading_hold or abs(wz_cmd) > 1e-3:
            self.heading_target = yaw
            return wz_cmd
        if self.heading_target is None:
            self.heading_target = yaw
        err = math.atan2(math.sin(self.heading_target - yaw), math.cos(self.heading_target - yaw))
        lim = self.cfg.heading_hold_max_wz
        return max(-lim, min(lim, self.cfg.heading_hold_stiffness * err))

    def inference(self) -> torch.Tensor:
        if self.start_pos is None:
            self.start_pos = self.robot.data.joint_pos.clone()

        obs = self.compute_observation()
        with torch.no_grad():
            action = self._model(obs.unsqueeze(0)).squeeze(0).clamp(-100.0, 100.0)
        self.last_action = action.clone()

        targets = self.robot.default_joint_pos.clone()
        targets[self.leg_idx] = self.leg_default + action * self.action_scale

        # Upper body: external target, clipped to limits and rate limited.
        want = self.upper_source.get().clamp(self.upper_lo, self.upper_hi)
        step = (want - self.upper_cmd).clamp(-self.max_upper_step, self.max_upper_step)
        self.upper_cmd = self.upper_cmd + step
        targets[self.upper_idx] = self.upper_cmd

        # Blend from the pose we started in (e.g. the prepare pose on the robot).
        self.steps += 1
        ramp_steps = max(1, round(self.cfg.ramp_s / self.controller.cfg.policy_dt))
        if self.steps < ramp_steps:
            alpha = self.steps / ramp_steps
            targets = self.start_pos + alpha * (targets - self.start_pos)
        return targets


@configclass
class LocoTeleopPolicyCfg(PolicyCfg):
    constructor = LocoTeleopPolicy
    checkpoint_path: str = "unused"  # model + metadata live in model_dir
    model_dir: str = MISSING  # type: ignore
    ramp_s: float = 1.5
    heading_hold: bool = True  # hold heading when the yaw command is 0
    heading_hold_stiffness: float = 0.5  # same as training heading_control_stiffness
    heading_hold_max_wz: float = 1.0
    upper_body_mode: str = "udp"  # "udp" | "default" | "random" | "motion" | "smooth" | "manual"
    upper_body_udp_port: int = 5555
    upper_body_timeout_s: float = 1.0
    upper_body_max_rate: float = 3.0  # rad/s, safety limit on target changes
    upper_body_amplitude: float = 0.6  # "random" mode
    upper_body_period_s: float = 2.0  # "random" mode
    upper_body_full_range_prob: float = 0.0  # "random" mode
    upper_body_motion_scale: float = 1.0  # "motion" mode amplitude (1 = limit to limit)
    upper_body_smooth_max_vel: float = 1.5  # "smooth" mode peak joint speed (rad/s)
    upper_body_smooth_scale: float = 1.0  # "smooth" mode: 1 = full poses, <1 = shrink towards rest
    upper_body_seed: int = 0  # "random" mode


def k1_loco_teleop_cfg(
    model: str,
    upper_body_mode: str = "udp",
    mjcf_path: str | None = None,
    init_z: float = 0.53,
    ankle_offset: float = 0.0,
    **policy_overrides,
) -> ControllerCfg:
    """Also serves full-body policies (no upper_body_* joints in the metadata: the
    policy drives all 22 joints). ``mjcf_path`` swaps the MuJoCo model (e.g. shoes).
    ``ankle_offset`` (rad) is added to both ankle pitches of the start pose (MuJoCo) and
    of the robot's prepare pose: in the heel shoes the sole is pitched toe-down, and
    without it the robot starts tipped forward."""
    """Controller cfg whose robot gains/default pose are copied from the export."""
    model_dir = os.path.join(os.path.dirname(__file__), "models", model)
    meta = _load_meta(model_dir)
    robot_names = [_norm_joint_name(n) for n in K1_CFG.joint_names]

    def robot_order(train_names: list[str], *fields: list[float]) -> list[list[float]]:
        out = []
        for values in fields:
            by_name = {_norm_joint_name(n): v for n, v in zip(train_names, values)}
            out.append([by_name[r] for r in robot_names])
        return out

    names = meta["joint_names"] + meta["upper_body_joint_names"]
    kp, kd = robot_order(
        names,
        meta["joint_stiffness"] + meta["upper_body_joint_stiffness"],
        meta["joint_damping"] + meta["upper_body_joint_damping"],
    )
    (default,) = robot_order(meta["all_joint_names"], meta["all_default_joint_pos"])

    @configclass
    class _Cfg(ControllerCfg):
        # effort_limit is only used by the MuJoCo controller. K1_CFG's values
        # (knee 40 Nm, hip pitch 30 Nm, ...) are 2-3x below the motor limits the
        # policy was trained with (booster-mjlab k1_constants.py, same as
        # booster_train); with them the policy shuffles and falls above ~0.7 m/s.
        robot = K1_CFG.replace(  # type: ignore
            default_joint_pos=default,
            joint_stiffness=kp,
            joint_damping=kd,
            effort_limit=K1_TRAIN_EFFORT_LIMIT,
        )
        vel_command: VelocityCommandCfg = VelocityCommandCfg(
            vx_max=1.0, vy_max=1.0, vyaw_max=1.0
        )
        policy: LocoTeleopPolicyCfg = LocoTeleopPolicyCfg(
            model_dir=model_dir, upper_body_mode=upper_body_mode
        )

    cfg = _Cfg()
    for key, value in policy_overrides.items():
        setattr(cfg.policy, key, value)
    if mjcf_path is not None:
        cfg.robot.mjcf_path = mjcf_path
    if ankle_offset:
        ankles = [i for i, n in enumerate(cfg.robot.joint_names) if n.endswith("Ankle_Pitch")]
        default = list(cfg.robot.default_joint_pos)
        prepare = list(cfg.robot.prepare_state.joint_pos)
        for i in ankles:
            default[i] += ankle_offset
            prepare[i] += ankle_offset
        cfg.robot.default_joint_pos = default
        cfg.robot.prepare_state = cfg.robot.prepare_state.replace(joint_pos=prepare)
    cfg.mujoco.init_pos = [0.0, 0.0, init_z]
    return cfg
