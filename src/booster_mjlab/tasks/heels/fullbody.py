"""Full-body K1 walking (the standard velocity AMP task, arms driven by the policy) in the
printed 50 mm heel shoes. Same shoe and motion compensation as the loco-teleop Shoe50mm
task; meant to be warm-started from a bare-foot full-body checkpoint.

Task: Mjlab-Velocity-Flat-Amp-DA-Shoe50mm-Muon-Booster-K1
"""

from mjlab.tasks.registry import register_mjlab_task

from booster_mjlab.amp.config import DEFAULT_AMP_DATASET
from booster_mjlab.tasks.velocity.config import with_amp_obs_group
from booster_mjlab.tasks.velocity.config.k1.env_cfgs import booster_k1_flat_env_cfg
from booster_mjlab.tasks.velocity.config.k1.rl_cfg import booster_k1_amp_ppo_symmetric_runner_cfg
from booster_mjlab.tasks.loco_teleop.config.k1.rl_cfg import STYLE_REWARD_WEIGHT, set_style_weight
from booster_mjlab.tasks.velocity.rl.runner import VelocityAmpOnPolicyRunner

from .shoe import SHOE_MOTION_TRANSFORM, apply_shoe


def _env_cfg(runner_cfg, play: bool):
    cfg = booster_k1_flat_env_cfg(play=play)
    apply_shoe(cfg)
    return set_style_weight(with_amp_obs_group(
        cfg,
        dataset_root=runner_cfg.dataset_root,
        speed_factor=runner_cfg.speed_factor,
        dataset_weights=runner_cfg.dataset_weights,
        augmentations=runner_cfg.dataset_augmentations,
        dataset_transform=runner_cfg.dataset_transform,
        include_base_lin_vel=True,
    ))


_rl = booster_k1_amp_ppo_symmetric_runner_cfg(use_muon=True)
_rl.dataset_root = DEFAULT_AMP_DATASET  # LAFAN normal walking, like the bare-foot run
_rl.dataset_transform = SHOE_MOTION_TRANSFORM
_rl.warm_start = True
_rl.style_reward_weight = STYLE_REWARD_WEIGHT  # as in the bare-foot run it starts from
_rl.experiment_name = "k1_velocity_amp_symmetric_shoe50mm_muon"

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Amp-DA-Shoe50mm-Muon-Booster-K1",
    env_cfg=_env_cfg(_rl, play=False),
    play_env_cfg=_env_cfg(_rl, play=True),
    rl_cfg=_rl,
    runner_cls=VelocityAmpOnPolicyRunner,
)
