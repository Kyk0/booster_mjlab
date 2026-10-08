"""Shared helpers for the loco-teleop scripts (teleop + robustness comparison)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

UPPER_JOINTS = (r"Head_.*", r".*_Shoulder_.*", r".*_Elbow_.*")


def load_task_cfgs(
    task_id: str, play: bool, dataset_root: str | None
) -> tuple[ManagerBasedRlEnvCfg, object]:
    """Load env + rl cfg, optionally pointing the AMP dataset at a local file."""
    import mjlab.tasks  # noqa: F401  (populates the registry)

    env_cfg = load_env_cfg(task_id, play=play)
    agent_cfg = load_rl_cfg(task_id)
    if dataset_root is not None:
        dataset_root = str(Path(dataset_root).expanduser())
        reset_term = env_cfg.events.get("reset_robot_from_motion")
        if reset_term is not None:
            reset_term.params["dataset_root"] = dataset_root
        if hasattr(agent_cfg, "dataset_root"):
            agent_cfg.dataset_root = dataset_root
    return env_cfg, agent_cfg


def make_fixed_command(env_cfg: ManagerBasedRlEnvCfg) -> None:
    """Make the twist command a plain externally-set value (no resampling/heading)."""
    twist = env_cfg.commands["twist"]
    twist.resampling_time_range = (1.0e9, 1.0e9)
    twist.heading_command = False
    twist.ranges.heading = None
    twist.rel_heading_envs = 0.0
    twist.rel_standing_envs = 0.0
    twist.rel_forward_envs = 0.0
    if hasattr(twist, "rel_world_envs"):
        twist.rel_world_envs = 0.0
    env_cfg.curriculum.pop("command_vel", None)


def load_runner(task_id: str, env, agent_cfg, checkpoint: str, device: str):
    """Build the task's runner on ``env`` and load the actor from ``checkpoint``."""
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(Path(checkpoint).expanduser()),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    return runner


def build_env_and_policy(
    task_id: str,
    env_cfg: ManagerBasedRlEnvCfg,
    agent_cfg,
    checkpoint: str | None,
    device: str,
):
    """Create the wrapped env and an inference policy (zero policy if no checkpoint)."""
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if checkpoint is None:
        action_shape = env.unwrapped.action_space.shape

        def policy(obs):
            del obs
            return torch.zeros(action_shape, device=env.unwrapped.device)

        return env, policy

    runner = load_runner(task_id, env, agent_cfg, checkpoint, device)
    return env, runner.get_inference_policy(device=device)


def upper_joint_ids(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, list[str]]:
    robot = env.scene["robot"]
    ids, names = robot.find_joints(UPPER_JOINTS, preserve_order=True)
    return torch.as_tensor(ids, device=env.device), list(names)


def set_upper_targets(env: ManagerBasedRlEnv, ids: torch.Tensor, target: torch.Tensor):
    """Write upper-body PD targets for all envs, clipped to soft joint limits."""
    robot = env.scene["robot"]
    limits = robot.data.soft_joint_pos_limits[:, ids]
    target = target.expand(env.num_envs, -1).clamp(limits[..., 0], limits[..., 1])
    robot.set_joint_position_target(target, joint_ids=ids)
