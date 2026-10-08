"""RL configuration for Booster K1 loco-teleop task."""

from booster_mjlab.amp.runners import AmpOnPolicyRunnerCfg
from booster_mjlab.tasks.velocity.config.k1.rl_cfg import booster_k1_amp_ppo_runner_cfg

# AMP style reward weight used for all loco-teleop and heel runs (upstream default: 0.3).
STYLE_REWARD_WEIGHT = 0.5


def set_style_weight(env_cfg, weight: float = STYLE_REWARD_WEIGHT):
    """Constant AMP style weight in the env curriculum term added by with_amp_obs_group."""
    params = env_cfg.curriculum["amp_style_weight"].params
    params["start_weight"] = params["end_weight"] = weight
    return env_cfg


def booster_k1_loco_teleop_runner_cfg(
    experiment_name: str, use_muon: bool = False
) -> AmpOnPolicyRunnerCfg:
    """Same networks / PPO / AMP settings as the K1 velocity AMP task, style weight 0.5."""
    cfg = booster_k1_amp_ppo_runner_cfg(use_muon=use_muon)
    cfg.style_reward_weight = STYLE_REWARD_WEIGHT
    cfg.experiment_name = experiment_name + ("_muon" if use_muon else "")
    return cfg
