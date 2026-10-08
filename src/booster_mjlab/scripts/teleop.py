"""Keyboard teleop for loco-teleop policies: drive the legs' velocity command and the
upper-body joint targets from the terminal while watching in the viewer.

    uv run python -m booster_mjlab.scripts.teleop \
        Mjlab-LocoTeleop-Flat-Amp-Lower-Rand-Muon-Booster-K1 \
        --checkpoint-file logs/rsl_rl/<exp>/<run>/model_XXXX.pt \
        --dataset-root ~/booster_projects/catwalk4_k1.pkl

Keys are read from the terminal you launched from (works over ssh); the viser
viewer prints its URL (forward the port: ``ssh -L 8080:localhost:8080 ...``).

  Walking:     w/s  forward/back   a/d  left/right   q/e  turn   space  stop
  Arm poses:   1 default   2 arms forward   3 arms up   4 T-pose   5 right arm up
  Fine control: [ / ]  select upper joint    - / =  move it by 0.1 rad
               0  clear fine offsets          h  print help
  Auto arms:   r  toggle random arm poses (amplitude --auto-arms, every --auto-period s)

Side-by-side demo: run one process per policy on different --port values with the
same --auto-arms/--seed/--vx; both get the identical sequence of arm poses.
"""

from __future__ import annotations

import argparse
import atexit
import os
import sys
import threading

import torch

from booster_mjlab.scripts.loco_teleop_utils import (
    build_env_and_policy,
    load_task_cfgs,
    make_fixed_command,
    set_upper_targets,
    upper_joint_ids,
)

LIN_STEP = 0.1
YAW_STEP = 0.2
JOINT_STEP = 0.1
LIMITS = {"vx": (-1.5, 1.75), "vy": (-1.0, 1.0), "wz": (-1.5, 1.5)}

# Absolute targets (rad) layered over the default pose; unspecified joints keep default.
# Shoulder pitch < 0 raises the arm forward/up, shoulder roll 0 is arms out sideways.
PRESETS: dict[str, tuple[str, dict[str, float]]] = {
    "1": ("default", {}),
    "2": ("arms forward", {"Left_Shoulder_Pitch": -1.5, "Right_Shoulder_Pitch": -1.5}),
    "3": ("arms up", {"Left_Shoulder_Pitch": -3.0, "Right_Shoulder_Pitch": -3.0}),
    "4": ("T-pose", {"Left_Shoulder_Roll": 0.0, "Right_Shoulder_Roll": 0.0}),
    "5": ("right arm up", {"Right_Shoulder_Pitch": -3.0, "Right_Elbow_Pitch": -0.5}),
}


class TeleopState:
    def __init__(self, joint_names: list[str]):
        self.lock = threading.Lock()
        self.cmd = [0.0, 0.0, 0.0]
        self.joint_names = joint_names
        self.preset: dict[str, float] = {}
        self.preset_name = "default"
        self.offsets = [0.0] * len(joint_names)
        self.selected = 0
        self.auto_amp = 0.0
        self.auto_amp_toggle = 0.6

    def status(self) -> str:
        j = self.joint_names[self.selected]
        return (
            f"vx={self.cmd[0]:+.2f} vy={self.cmd[1]:+.2f} wz={self.cmd[2]:+.2f} | "
            f"pose={self.preset_name} | auto_arms={self.auto_amp:.1f} | "
            f"joint[{self.selected}]={j} offset={self.offsets[self.selected]:+.2f}"
        )

    def handle(self, key: str) -> None:
        with self.lock:
            c = self.cmd
            if key == "w":
                c[0] += LIN_STEP
            elif key == "s":
                c[0] -= LIN_STEP
            elif key == "a":
                c[1] += LIN_STEP
            elif key == "d":
                c[1] -= LIN_STEP
            elif key == "q":
                c[2] += YAW_STEP
            elif key == "e":
                c[2] -= YAW_STEP
            elif key == " ":
                self.cmd = [0.0, 0.0, 0.0]
            elif key in PRESETS:
                self.preset_name, self.preset = PRESETS[key]
            elif key == "[":
                self.selected = (self.selected - 1) % len(self.joint_names)
            elif key == "]":
                self.selected = (self.selected + 1) % len(self.joint_names)
            elif key == "-":
                self.offsets[self.selected] -= JOINT_STEP
            elif key == "=":
                self.offsets[self.selected] += JOINT_STEP
            elif key == "0":
                self.offsets = [0.0] * len(self.joint_names)
            elif key == "r":
                self.auto_amp = 0.0 if self.auto_amp > 0 else self.auto_amp_toggle
            elif key == "h":
                print("\r\n" + __doc__.split("\n\n", 2)[2] + "\r")
                return
            else:
                return
            for i, (name, (lo, hi)) in enumerate(LIMITS.items()):
                self.cmd[i] = round(min(max(self.cmd[i], lo), hi), 3)
        sys.stdout.write("\r\033[K" + self.status())
        sys.stdout.flush()


def _keyboard_loop(state: TeleopState) -> None:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    atexit.register(termios.tcsetattr, fd, termios.TCSADRAIN, old)
    tty.setcbreak(fd)
    while True:
        ch = os.read(fd, 1).decode(errors="ignore").lower()
        if ch:
            state.handle(ch)


class TeleopPolicy:
    """Wraps the trained policy: applies teleop state to the env, then acts."""

    def __init__(
        self, env, policy, state: TeleopState, auto_period_s: float = 2.0, seed: int = 0
    ):
        self.env = env.unwrapped
        self.policy = policy
        self.state = state
        self.ids, names = upper_joint_ids(self.env)
        assert names == state.joint_names
        robot = self.env.scene["robot"]
        self.default = robot.data.default_joint_pos[0, self.ids].clone()
        self.twist = self.env.command_manager.get_term("twist")
        # Auto arm poses depend only on (seed, step count), so two processes with the
        # same settings see the same sequence at the same sim time.
        self.auto_period = max(1, round(auto_period_s / self.env.step_dt))
        self.rng = torch.Generator().manual_seed(seed)
        self.auto_unit = torch.zeros(len(state.joint_names))
        self.steps = 0

    @torch.no_grad()
    def __call__(self, obs):
        s = self.state
        if self.steps % self.auto_period == 0:
            self.auto_unit = torch.rand(len(s.joint_names), generator=self.rng) * 2 - 1
        self.steps += 1
        with s.lock:
            cmd = torch.tensor(s.cmd, device=self.env.device)
            target = self.default.clone()
            for i, name in enumerate(s.joint_names):
                if name in s.preset:
                    target[i] = s.preset[name]
                target[i] += s.offsets[i] + s.auto_amp * self.auto_unit[i].item()
        self.twist.vel_command_b[:] = cmd
        set_upper_targets(self.env, self.ids, target.unsqueeze(0))
        return self.policy(obs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task")
    p.add_argument("--checkpoint-file", default=None, help="omit to run a zero policy")
    p.add_argument("--dataset-root", default=None, help="AMP motion .pkl for resets")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--viewer", choices=("viser", "native"), default="viser")
    p.add_argument("--port", type=int, default=8080, help="viser port")
    p.add_argument("--label", default=None, help="title shown in the viser panel")
    p.add_argument("--vx", type=float, default=0.0, help="initial forward command")
    p.add_argument("--auto-arms", type=float, default=0.0, help="random arm amplitude (rad), 0=off")
    p.add_argument("--auto-period", type=float, default=2.0, help="seconds between arm poses")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env_cfg, agent_cfg = load_task_cfgs(args.task, play=True, dataset_root=args.dataset_root)
    env_cfg.scene.num_envs = 1
    make_fixed_command(env_cfg)
    # Teleop owns the upper body: drop the random-target event.
    env_cfg.events.pop("upper_body_teleop", None)

    env, policy = build_env_and_policy(
        args.task, env_cfg, agent_cfg, args.checkpoint_file, args.device
    )
    _, names = upper_joint_ids(env.unwrapped)
    state = TeleopState(names)
    state.cmd[0] = args.vx
    state.auto_amp = args.auto_arms
    if args.auto_arms > 0:
        state.auto_amp_toggle = args.auto_arms
    teleop_policy = TeleopPolicy(env, policy, state, args.auto_period, args.seed)

    if sys.stdin.isatty():
        threading.Thread(target=_keyboard_loop, args=(state,), daemon=True).start()
        print(__doc__.split("\n\n", 2)[2])
        print("upper joints:", ", ".join(f"{i}:{n}" for i, n in enumerate(names)))
    else:
        print("[WARN] stdin is not a TTY: keyboard teleop disabled.")

    if args.viewer == "native":
        from mjlab.viewer import NativeMujocoViewer

        NativeMujocoViewer(env, teleop_policy).run()
    else:
        import viser
        from mjlab.viewer import ViserPlayViewer

        server = viser.ViserServer(port=args.port, label=args.label or "teleop")
        if args.label:
            server.gui.add_markdown(f"## {args.label}")
        ViserPlayViewer(env, teleop_policy, viser_server=server).run()
    env.close()


if __name__ == "__main__":
    main()
