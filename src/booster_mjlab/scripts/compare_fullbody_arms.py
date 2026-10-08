"""Full-body walk policy vs. arm-motion-trained policy under external (random) arm motion.

The full-body policy outputs all 22 joints. For the "random arms" conditions its arm
outputs are overwritten every step so that the arm PD targets follow the same arm
motion generator the loco-teleop policies were trained with (``upper_body_motion``);
the legs stay under the policy. Conditions:
  own        full-body policy drives its own arms (as trained)      [full-body only]
  still      arms held at the default pose                          [loco-teleop only]
  motion:S   random circles / sweeps / jumps, amplitude scale S     [both]

    uv run python -m booster_mjlab.scripts.compare_fullbody_arms \
        --fullbody logs/rsl_rl/k1_velocity_amp_symmetric_muon/2026-09-18_21-54-10/model_2000.pt \
        --armmotion logs/rsl_rl/k1_loco_teleop_lower_armmotion_walk_muon/<run>/model_3999.pt \
        --out compare_fullbody
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from booster_mjlab.scripts.loco_teleop_utils import build_env_and_policy, load_task_cfgs

FULLBODY_TASK = "Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-K1"
ARMMOTION_TASK = "Mjlab-LocoTeleop-Flat-Amp-Lower-ArmMotion-Walk-Muon-Booster-K1"
UPPER = (r"Head_.*", r".*_Shoulder_.*", r".*_Elbow_.*")


def _final_command_ranges(cfg) -> None:
    cfg.curriculum.pop("command_vel", None)
    r = cfg.commands["twist"].ranges
    r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (-1.0, 1.5), (-1.0, 1.0), (-1.0, 1.0)


def _add_arm_motion(cfg) -> None:
    from booster_mjlab.tasks.loco_teleop import mdp

    upper = SceneEntityCfg("robot", joint_names=UPPER)
    cfg.events["arm_motion_reset"] = EventTermCfg(
        func=mdp.upper_body_motion, mode="reset", params={"asset_cfg": upper, "reset": True, "amp_scale": 1.0}
    )
    cfg.events["arm_motion_step"] = EventTermCfg(
        func=mdp.upper_body_motion, mode="step", params={"asset_cfg": upper, "amp_scale": 1.0}
    )


def _set_arm_mode(base, condition: str) -> bool:
    """Configure the arm generator; returns whether it is active."""
    from booster_mjlab.tasks.loco_teleop import mdp

    em = base.event_manager
    if condition.startswith("motion:"):
        scale = float(condition.split(":")[1])
        for name in ("arm_motion_reset", "arm_motion_step"):
            em.get_term_cfg(name).func = mdp.upper_body_motion
            em.get_term_cfg(name).params["amp_scale"] = scale
        return True
    for name in ("arm_motion_reset", "arm_motion_step"):
        em.get_term_cfg(name).func = lambda env, env_ids, **kw: None
    return False


class ArmOverride:
    """Overwrite a full-body policy's arm outputs so its PD targets follow the generator."""

    def __init__(self, env, policy):
        base = env.unwrapped
        self.policy = policy
        self.robot = base.scene["robot"]
        term = base.action_manager.get_term("joint_pos")
        names = list(term._target_names)
        upper_names = set(self.robot.find_joints(UPPER)[1])
        self.act_idx = torch.tensor([i for i, n in enumerate(names) if n in upper_names], device=base.device)
        self.joint_ids = torch.as_tensor(
            [self.robot.joint_names.index(names[i]) for i in self.act_idx.tolist()], device=base.device
        )
        scale = term._scale
        self.scale = scale[:, self.act_idx] if isinstance(scale, torch.Tensor) else scale
        self.offset = term._offset[:, self.act_idx]
        self.active = False

    def __call__(self, obs):
        action = self.policy(obs)
        if self.active:
            target = self.robot.data.joint_pos_target[:, self.joint_ids]  # set by the generator
            action[:, self.act_idx] = (target - self.offset) / self.scale
        return action


@torch.no_grad()
def evaluate(env, policy, steps: int, seed: int) -> dict[str, float]:
    base = env.unwrapped
    torch.manual_seed(seed)
    base.seed(seed)
    obs, _ = env.reset()
    robot, twist = base.scene["robot"], base.command_manager.get_term("twist")
    falls = torch.zeros(base.num_envs, device=base.device)
    lin = ang = 0.0
    for _ in range(steps):
        obs, _, dones, extras = env.step(policy(obs))
        falls += (dones.bool() & ~extras["time_outs"].bool()).float()
        cmd = twist.vel_command_b
        lin += torch.norm(cmd[:, :2] - robot.data.root_link_lin_vel_b[:, :2], dim=-1).mean().item()
        ang += (cmd[:, 2] - robot.data.root_link_ang_vel_b[:, 2]).abs().mean().item()
    minutes = steps * base.step_dt / 60.0
    return {
        "falls_per_min": falls.mean().item() / minutes,
        "pct_envs_fell": 100.0 * (falls > 0).float().mean().item(),
        "lin_vel_err": lin / steps,
        "ang_vel_err": ang / steps,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fullbody", required=True)
    p.add_argument("--armmotion", required=True)
    p.add_argument("--fullbody-task", default=FULLBODY_TASK)
    p.add_argument("--armmotion-task", default=ARMMOTION_TASK)
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--seconds", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="compare_fullbody")
    args = p.parse_args()

    runs = [
        ("full-body walk (own arms policy)", args.fullbody_task, args.fullbody, ["own", "motion:0.5", "motion:1.0"]),
        ("arm-motion-trained walk", args.armmotion_task, args.armmotion, ["still", "motion:0.5", "motion:1.0"]),
    ]
    rows = []
    for label, task, ckpt, conditions in runs:
        cfg, agent = load_task_cfgs(task, play=False, dataset_root=None)
        cfg.scene.num_envs = args.num_envs
        _final_command_ranges(cfg)
        cfg.events.pop("upper_body_teleop", None)
        cfg.events.pop("upper_body_reset", None)
        if task == args.armmotion_task:  # still = hold default pose
            from booster_mjlab.tasks.loco_teleop.config.k1.env_cfgs import add_upper_body_events

            add_upper_body_events(cfg, 0.0)
        _add_arm_motion(cfg)
        env, policy = build_env_and_policy(task, cfg, agent, ckpt, args.device)
        override = ArmOverride(env, policy) if task == args.fullbody_task else None
        steps = int(args.seconds / env.unwrapped.step_dt)
        for cond in conditions:
            active = _set_arm_mode(env.unwrapped, cond)
            if override is not None:
                override.active = active
            res = evaluate(env, override or policy, steps, args.seed)
            row = {"policy": label, "arms": cond, **res}
            rows.append(row)
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        env.close()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = list(rows[0])
    table = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    table += ["| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |" for r in rows]
    (out / "comparison.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))


if __name__ == "__main__":
    main()
