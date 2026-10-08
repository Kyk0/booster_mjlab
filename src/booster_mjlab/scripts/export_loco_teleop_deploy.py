"""Export a loco-teleop checkpoint for booster_deploy: TorchScript policy + metadata.

    uv run python -m booster_mjlab.scripts.export_loco_teleop_deploy \
        --checkpoint logs/rsl_rl/k1_loco_teleop_lower_rand_muon/<run>/model_4999.pt \
        --out ~/k1_train/booster_deploy/tasks/loco_teleop/models/rand

Writes ``policy.pt`` (TorchScript, observation normalizer baked in, obs -> actions)
and ``deploy_meta.json`` (joint names/order, default pose, PD gains, action scale,
upper-body joints and limits, observation layout). The export is checked against the
training-time inference policy on random observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from booster_mjlab.scripts.loco_teleop_utils import load_runner, load_task_cfgs
from booster_mjlab.tasks.loco_teleop.rl.runner import get_loco_teleop_metadata

DEFAULT_TASK = "Mjlab-LocoTeleop-Flat-Amp-Lower-Rand-Muon-Booster-K1"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    env_cfg, agent_cfg = load_task_cfgs(args.task, play=True, dataset_root=args.dataset_root)
    env_cfg.scene.num_envs = 1
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=env_cfg, device=args.device),
        clip_actions=agent_cfg.clip_actions,
    )
    runner = load_runner(args.task, env, agent_cfg, args.checkpoint, args.device)
    base = env.unwrapped

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_jit(str(out), "policy.pt")

    md = get_loco_teleop_metadata(base, run_path=str(Path(args.checkpoint).resolve()))
    robot = base.scene["robot"]
    limits = robot.data.soft_joint_pos_limits[0].cpu()
    names = list(robot.joint_names)
    md["upper_body_joint_limits"] = [
        limits[names.index(n)].tolist() for n in md["upper_body_joint_names"]
    ]
    md["all_joint_names"] = names
    md["all_default_joint_pos"] = robot.data.default_joint_pos[0].cpu().tolist()
    md["observation_dims"] = {
        name: int(dim[0])
        for name, dim in zip(
            base.observation_manager.active_terms["actor"],
            base.observation_manager.group_obs_term_dim["actor"],
        )
    }
    md["policy_dt"] = base.step_dt
    md["checkpoint"] = str(Path(args.checkpoint).resolve())
    (out / "deploy_meta.json").write_text(json.dumps(md, indent=2))

    # Parity check: TorchScript export vs. the training-time inference policy.
    jit = torch.jit.load(str(out / "policy.pt"), map_location="cpu")
    policy = runner.get_inference_policy(device=args.device)
    obs = env.get_observations()
    actor_obs = obs["actor"]
    gen = torch.Generator().manual_seed(0)
    max_err = 0.0
    for _ in range(16):
        x = torch.randn(actor_obs.shape, generator=gen).to(actor_obs.device)
        obs["actor"] = x
        with torch.no_grad():
            ref = policy(obs).cpu()
            got = jit(x.cpu())
        max_err = max(max_err, (ref - got).abs().max().item())
    print(f"[export] wrote {out}/policy.pt and deploy_meta.json")
    print(f"[export] obs dims {md['observation_dims']} -> {len(md['joint_names'])} actions")
    print(f"[export] max |jit - policy| over 16 random obs: {max_err:.2e}")
    if max_err > 1e-4:
        raise SystemExit("[export] parity check FAILED")
    env.close()


if __name__ == "__main__":
    main()
