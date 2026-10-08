from mjlab.tasks.registry import register_mjlab_task

from booster_mjlab.tasks.velocity.config import with_amp_obs_group
from booster_mjlab.tasks.loco_teleop.rl.runner import LocoTeleopAmpOnPolicyRunner

from booster_mjlab.amp.config import DEFAULT_AMP_DATASET

from .env_cfgs import booster_k1_loco_teleop_flat_env_cfg
from .rl_cfg import booster_k1_loco_teleop_runner_cfg, set_style_weight


def _with_amp(env_cfg, runner_cfg):
    return set_style_weight(with_amp_obs_group(
        env_cfg,
        dataset_root=runner_cfg.dataset_root,
        speed_factor=runner_cfg.speed_factor,
        dataset_weights=runner_cfg.dataset_weights,
        augmentations=runner_cfg.dataset_augmentations,
        dataset_transform=runner_cfg.dataset_transform,
        include_base_lin_vel=True,
    ))


# Variant -> (upper-body target noise in rad, full range, arm motion, experiment name).
#   Lower:          upper body held at the default pose during training.
#   Lower-Rand:     upper body jumps to default ± 0.6 rad every 1-3 s.
#   Lower-FullArms: upper body jumps anywhere in its joint range every 0.3-2 s.
#   Lower-ArmMotion: big arm circles / sweeps / jumps over the whole range
#                    (AMP style from --agent.dataset-root).
#   Lower-ArmMotion-Walk: same, AMP style always from LAFAN normal walking/running.
#   Lower-ArmMotion-Walk-Shoe50mm: Walk wearing the printed 50 mm heel shoe; AMP/reset
#                    motions shoe-compensated; warm-started from a Walk checkpoint.
_VARIANTS = {
    "Flat-Amp-Lower": (0.0, False, False, "k1_loco_teleop_lower"),
    "Flat-Amp-Lower-Rand": (0.6, False, False, "k1_loco_teleop_lower_rand"),
    "Flat-Amp-Lower-FullArms": (0.0, True, False, "k1_loco_teleop_lower_fullarms"),
    "Flat-Amp-Lower-ArmMotion": (0.0, False, True, "k1_loco_teleop_lower_armmotion"),
    "Flat-Amp-Lower-ArmMotion-Walk": (0.0, False, True, "k1_loco_teleop_lower_armmotion_walk"),
    "Flat-Amp-Lower-ArmMotion-Walk-Shoe50mm": (0.0, False, True, "k1_loco_teleop_lower_armmotion_walk_shoe50mm"),
}
_SHOE = {"Flat-Amp-Lower-ArmMotion-Walk-Shoe50mm"}
# Variants whose AMP style dataset is fixed in the task (not left to the CLI).
_DATASETS = {
    "Flat-Amp-Lower-ArmMotion-Walk": DEFAULT_AMP_DATASET,
    "Flat-Amp-Lower-ArmMotion-Walk-Shoe50mm": DEFAULT_AMP_DATASET,
}

for use_muon in (False, True):
    optimizer = "-Muon" if use_muon else ""
    for name, (noise, full_range, arm_motion, experiment_name) in _VARIANTS.items():
        rl_cfg = booster_k1_loco_teleop_runner_cfg(experiment_name, use_muon=use_muon)
        if name in _DATASETS:
            rl_cfg.dataset_root = _DATASETS[name]
        shoe = name in _SHOE
        if shoe:
            from booster_mjlab.tasks.heels.shoe import SHOE_MOTION_TRANSFORM

            rl_cfg.dataset_transform = SHOE_MOTION_TRANSFORM
            rl_cfg.warm_start = True
        register_mjlab_task(
            task_id=f"Mjlab-LocoTeleop-{name}{optimizer}-Booster-K1",
            env_cfg=_with_amp(
                booster_k1_loco_teleop_flat_env_cfg(
                    upper_body_noise=noise, full_range=full_range, arm_motion=arm_motion, shoe=shoe
                ),
                rl_cfg,
            ),
            play_env_cfg=_with_amp(
                booster_k1_loco_teleop_flat_env_cfg(
                    play=True,
                    upper_body_noise=noise,
                    full_range=full_range,
                    arm_motion=arm_motion,
                    shoe=shoe,
                ),
                rl_cfg,
            ),
            rl_cfg=rl_cfg,
            runner_cls=LocoTeleopAmpOnPolicyRunner,
        )
