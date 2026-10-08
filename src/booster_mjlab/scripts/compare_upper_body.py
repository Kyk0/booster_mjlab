"""Compare loco-teleop policies under increasing upper-body disturbance.

Every policy runs in the *same* env (the ``--env-task`` cfg) with the same seed; only
the upper-body disturbance changes between conditions. Reports, per policy x
condition: fall rate, velocity tracking errors and mean step reward.

Conditions (``--conditions``):
  near:X   targets = default ± X rad
  full:P   with probability P a target uniform over the full joint range
           (arms overhead/behind/sideways), else default ± 0.6 rad; targets
           jump every ``--resample`` seconds (low values = quick arm motions)
  motion:S big continuous arm motions (circles / sweeps / jumps over the whole
           range, see ``upper_body_motion``) with amplitude scale S

    uv run python -m booster_mjlab.scripts.compare_upper_body \
        --policy fixed=logs/rsl_rl/k1_loco_teleop_lower_muon/<run>/model_XXXX.pt \
        --policy rand=logs/rsl_rl/k1_loco_teleop_lower_rand_muon/<run>/model_XXXX.pt \
        --policy full=logs/rsl_rl/k1_loco_teleop_lower_fullarms_muon/<run>/model_XXXX.pt \
        --dataset-root ~/GMR/retargeting_data/catwalk4_k1.pkl \
        --conditions near:0 near:0.6 full:0.5 full:1.0 --num-envs 1024 --seconds 40
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from booster_mjlab.scripts.loco_teleop_utils import build_env_and_policy, load_task_cfgs

DEFAULT_ENV_TASK = "Mjlab-LocoTeleop-Flat-Amp-Lower-Rand-Muon-Booster-K1"


def parse_condition(spec: str) -> tuple[str, dict[str, float]]:
    kind, value = spec.split(":")
    if kind == "near":
        return kind, {"near_noise": float(value), "full_range_prob": 0.0}
    if kind == "full":
        return kind, {"near_noise": 0.6, "full_range_prob": float(value)}
    if kind == "motion":
        return kind, {"amp_scale": float(value)}
    raise ValueError(f"bad condition {spec!r}, use near:X, full:P or motion:S")


def _noop(env, env_ids, **kwargs) -> None:
    del env, env_ids, kwargs


def _set_condition(base, condition: str) -> None:
    """Route the upper body to either the jump events or the motion generator."""
    from booster_mjlab.tasks.loco_teleop import mdp

    kind, params = parse_condition(condition)
    em = base.event_manager
    jump_terms = ("upper_body_reset", "upper_body_teleop")
    motion_terms = ("upper_body_motion_reset", "upper_body_motion_step")
    for name in jump_terms:
        cfg = em.get_term_cfg(name)
        cfg.func = _noop if kind == "motion" else mdp.randomize_upper_body_full_range
        if kind != "motion":
            cfg.params.update(params, ramp_steps=0)
    for name in motion_terms:
        cfg = em.get_term_cfg(name)
        cfg.func = mdp.upper_body_motion if kind == "motion" else _noop
        if kind == "motion":
            cfg.params.update(params)


@torch.no_grad()
def evaluate(env, policy, condition: str, steps: int, seed: int) -> dict[str, float]:
    base = env.unwrapped
    _set_condition(base, condition)

    torch.manual_seed(seed)
    base.seed(seed)
    obs, _ = env.reset()
    robot = base.scene["robot"]
    twist = base.command_manager.get_term("twist")

    falls = torch.zeros(base.num_envs, device=base.device)
    lin_err = ang_err = rew_sum = 0.0
    for _ in range(steps):
        obs, rew, dones, extras = env.step(policy(obs))
        falls += (dones.bool() & ~extras["time_outs"].bool()).float()
        cmd = twist.vel_command_b
        lin_err += torch.norm(cmd[:, :2] - robot.data.root_link_lin_vel_b[:, :2], dim=-1).mean().item()
        ang_err += (cmd[:, 2] - robot.data.root_link_ang_vel_b[:, 2]).abs().mean().item()
        rew_sum += rew.mean().item()

    minutes = steps * base.step_dt / 60.0
    return {
        "falls_per_min": falls.mean().item() / minutes,
        "pct_envs_fell": 100.0 * (falls > 0).float().mean().item(),
        "lin_vel_err": lin_err / steps,
        "ang_vel_err": ang_err / steps,
        "mean_step_reward": rew_sum / steps,
    }


def _plot(rows: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ("falls_per_min", "pct_envs_fell", "lin_vel_err", "ang_vel_err")
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.2))
    for ax, m in zip(axes, metrics):
        for name in dict.fromkeys(r["policy"] for r in rows):
            pr = [r for r in rows if r["policy"] == name]
            ax.plot([r["condition"] for r in pr], [r[m] for r in pr], marker="o", label=name)
        ax.set_title(m)
        ax.set_xlabel("upper-body condition")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=120)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", action="append", required=True, help="NAME=CHECKPOINT (repeat)")
    p.add_argument("--env-task", default=DEFAULT_ENV_TASK)
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--conditions", nargs="+", default=["near:0", "near:0.6", "full:1.0", "motion:0.5", "motion:1.0"])
    p.add_argument("--resample", type=float, nargs=2, default=[0.3, 2.0], help="seconds between arm targets")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--seconds", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="compare_out")
    args = p.parse_args()

    policies = dict(s.split("=", 1) for s in args.policy)

    env_cfg, agent_cfg = load_task_cfgs(args.env_task, play=False, dataset_root=args.dataset_root)
    env_cfg.scene.num_envs = args.num_envs
    # Evaluate on the final curriculum command ranges from step 0.
    env_cfg.curriculum.pop("command_vel", None)
    ranges = env_cfg.commands["twist"].ranges
    ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z = (-1.0, 1.5), (-1.0, 1.0), (-1.0, 1.0)
    # Same upper-body event for every condition; its params are set per condition.
    from booster_mjlab.tasks.loco_teleop.config.k1.env_cfgs import (
        add_full_range_upper_body_events,
    )

    from mjlab.managers.event_manager import EventTermCfg

    add_full_range_upper_body_events(env_cfg)
    env_cfg.events["upper_body_teleop"].interval_range_s = tuple(args.resample)
    upper = env_cfg.events["upper_body_teleop"].params["asset_cfg"]
    env_cfg.events["upper_body_motion_reset"] = EventTermCfg(
        func=_noop, mode="reset", params={"asset_cfg": upper, "reset": True}
    )
    env_cfg.events["upper_body_motion_step"] = EventTermCfg(
        func=_noop, mode="step", params={"asset_cfg": upper}
    )
    for c in args.conditions:
        parse_condition(c)

    rows: list[dict] = []
    env = None
    for name, ckpt in policies.items():
        if env is None:
            env, policy = build_env_and_policy(args.env_task, env_cfg, agent_cfg, ckpt, args.device)
        else:
            _, policy = _load_policy(args, env, agent_cfg, ckpt)
        steps = int(args.seconds / env.unwrapped.step_dt)
        for condition in args.conditions:
            res = evaluate(env, policy, condition, steps, args.seed)
            row = {"policy": name, "condition": condition, **res}
            rows.append(row)
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cols = list(rows[0])
    table = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        table.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    (out / "comparison.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))
    _plot(rows, out)
    print(f"\nWrote {out}/comparison.csv, comparison.md, comparison.png")
    env.close()


def _load_policy(args, env, agent_cfg, ckpt: str):
    from dataclasses import asdict

    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_runner_cls

    runner_cls = load_runner_cls(args.env_task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(str(Path(ckpt).expanduser()), load_cfg={"actor": True}, strict=True, map_location=args.device)
    return env, runner.get_inference_policy(device=args.device)


if __name__ == "__main__":
    main()
