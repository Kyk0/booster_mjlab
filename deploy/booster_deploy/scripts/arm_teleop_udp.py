"""Keyboard arm/head teleop for the k1_loco_teleop task (sends UDP to the policy).

    python3 scripts/arm_teleop_udp.py                       # sim on this machine
    python3 scripts/arm_teleop_udp.py --host <robot_ip>     # real robot

Keys:  1 default  2 arms forward  3 arms up  4 T-pose  5 right arm up
       [ / ] select joint   - / = move it 0.1 rad   0 clear offsets
       r  toggle random poses (every 2 s)   h help   Ctrl-C quit
Only stdlib; joint names/defaults/limits come from the model's deploy_meta.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import select
import socket
import sys
import termios
import time
import tty

PRESETS = {
    "1": ("default", {}),
    "2": ("arms forward", {"Left_Shoulder_Pitch": -1.5, "Right_Shoulder_Pitch": -1.5}),
    "3": ("arms up", {"Left_Shoulder_Pitch": -3.0, "Right_Shoulder_Pitch": -3.0}),
    "4": ("T-pose", {"Left_Shoulder_Roll": 0.0, "Right_Shoulder_Roll": 0.0}),
    "5": ("right arm up", {"Right_Shoulder_Pitch": -3.0, "Right_Elbow_Pitch": -0.5}),
}
STEP = 0.1


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--meta", default=os.path.join(here, "..", "tasks", "loco_teleop", "models", "armmotion_walk", "deploy_meta.json"))
    p.add_argument("--rate", type=float, default=30.0, help="send rate (Hz)")
    p.add_argument("--random-amp", type=float, default=0.6)
    p.add_argument("--random-period", type=float, default=2.0)
    args = p.parse_args()

    meta = json.load(open(args.meta))
    names = meta["upper_body_joint_names"]
    default = meta["upper_body_default_joint_pos"]
    limits = meta["upper_body_joint_limits"]

    preset_name, preset = PRESETS["1"]
    offsets = [0.0] * len(names)
    rand_on, rand_off, next_rand = False, [0.0] * len(names), 0.0
    sel = 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    print(__doc__.split("\n\n")[1])
    print("joints:", ", ".join(f"{i}:{n}" for i, n in enumerate(names)))
    print(f"sending to {args.host}:{args.port} at {args.rate:.0f} Hz")
    try:
        while True:
            if select.select([sys.stdin], [], [], 1.0 / args.rate)[0]:
                k = sys.stdin.read(1).lower()
                if k in PRESETS:
                    preset_name, preset = PRESETS[k]
                elif k == "[":
                    sel = (sel - 1) % len(names)
                elif k == "]":
                    sel = (sel + 1) % len(names)
                elif k == "-":
                    offsets[sel] -= STEP
                elif k == "=":
                    offsets[sel] += STEP
                elif k == "0":
                    offsets = [0.0] * len(names)
                elif k == "r":
                    rand_on, next_rand = not rand_on, 0.0
                    rand_off = [0.0] * len(names)
                elif k == "h":
                    print("\n" + __doc__.split("\n\n")[1])
                sys.stdout.write(
                    f"\r\033[Kpose={preset_name} random={'on' if rand_on else 'off'} "
                    f"joint[{sel}]={names[sel]} offset={offsets[sel]:+.1f}"
                )
                sys.stdout.flush()

            now = time.monotonic()
            if rand_on and now >= next_rand:
                rand_off = [random.uniform(-args.random_amp, args.random_amp) for _ in names]
                next_rand = now + args.random_period

            q = []
            for i, n in enumerate(names):
                v = preset.get(n, default[i]) + offsets[i] + (rand_off[i] if rand_on else 0.0)
                q.append(min(max(v, limits[i][0]), limits[i][1]))
            sock.sendto(json.dumps({"q": q}).encode(), (args.host, args.port))
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()


if __name__ == "__main__":
    main()
