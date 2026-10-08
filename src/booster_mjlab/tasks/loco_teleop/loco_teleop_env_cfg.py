# Adapted from booster_mjlab tasks/velocity/velocity_env_cfg.py (upstream terms apply).
"""Loco-teleop task configuration (robot-agnostic template).

Flat-ground velocity tracking. Same terms as ``tasks/velocity/velocity_env_cfg.py``
(flat variant) except the upper-body posture reward, which makes no sense once the
upper body is not policy-controlled. Robot-specific names are filled in by
``config/<robot>/env_cfgs.py``.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from booster_mjlab.tasks.loco_teleop import mdp
from booster_mjlab.terrains.contact import terrain_collisions

# Shared target foot height for the foot reward terms.
TARGET_FOOT_HEIGHT = 0.06


def scene_cfg() -> SceneCfg:
    # Robot entity and sensors are set per robot.
    return SceneCfg(
        terrain=TerrainEntityCfg(collisions=terrain_collisions(), terrain_type="plane"),
        num_envs=1,
        extent=2.0,
    )


def observations_cfg() -> dict[str, ObservationGroupCfg]:
    # Joint pos/vel cover ALL joints, so the policy sees where the upper body is.
    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"biased": True},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
        ),
    }
    critic_terms = {
        **actor_terms,
        # Ground-truth joint state without encoder bias or noise.
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
        ),
        "foot_height": ObservationTermCfg(
            func=mdp.foot_height,
            params={"sensor_name": "foot_height_scan"},
        ),
        "foot_air_time": ObservationTermCfg(
            func=mdp.foot_air_time,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "foot_contact": ObservationTermCfg(
            func=mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "foot_contact_forces": ObservationTermCfg(
            func=mdp.foot_contact_forces,
            params={"sensor_name": "feet_ground_contact"},
        ),
    }
    return {
        "actor": ObservationGroupCfg(
            terms=actor_terms, concatenate_terms=True, enable_corruption=True
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms, concatenate_terms=True, enable_corruption=False
        ),
    }


def actions_cfg() -> dict:
    # Joint set and scale are set per robot.
    return {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,
            use_default_offset=True,
        )
    }


def commands_cfg() -> dict:
    return {
        "twist": mdp.UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(1.0, 4.0),
            rel_standing_envs=0.2,
            rel_heading_envs=0.3,
            rel_forward_envs=0.1,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.5, 0.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.5, 0.5),
                heading=(-math.pi, math.pi),
            ),
        )
    }


def events_cfg() -> dict[str, EventTermCfg]:
    # Body/geom names are set per robot. Upper-body target events are added per robot.
    return {
        "terrain_contact": EventTermCfg(
            func=mdp.randomize_terrain_contact,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("terrain"),
                "solref_ranges": {0: (0.006, 0.03), 1: (0.95, 1.05)},
                "solimp_ranges": {0: (0.88, 0.92), 1: (0.94, 0.99), 2: (0.003, 0.01)},
                "shared_random": True,
            },
        ),
        # Replaced by the AMP motion reset when AMP is enabled.
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.01, 0.05),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (0.01, 0.3),
                    "roll": (-0.1, 0.1),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.5, 0.5),
                },
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.1, 0.1),
                "velocity_range": (-0.1, 0.1),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.5, 4.0),
            params={
                "velocity_range": {
                    "x": (-0.28, 0.28),
                    "y": (-0.28, 0.28),
                    "z": (-0.2, 0.2),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=mdp.dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),
                "operation": "abs",
                "ranges": (0.75, 1.25),
                "shared_random": True,
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=mdp.dr.encoder_bias,
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.015, 0.015)},
        ),
        "pd_gains": EventTermCfg(
            mode="startup",
            func=mdp.dr.pd_gains,
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
                "operation": "scale",
                "kp_range": (0.8, 1.2),
                "kd_range": (0.8, 1.2),
            },
        ),
        "trunk_inertia": EventTermCfg(
            mode="startup",
            func=mdp.dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=()),
                "alpha_range": (-0.05, 0.05),
                "t_range": (-0.05, 0.05),
            },
        ),
        "limb_inertia": EventTermCfg(
            mode="startup",
            func=mdp.dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=()),
                "alpha_range": (-0.05, 0.05),
                "t_range": (-0.025, 0.025),
            },
        ),
    }


def rewards_cfg() -> dict[str, RewardTermCfg]:
    return {
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=2.25,
            params={
                "command_name": "twist",
                "std": math.sqrt(0.25),
                "std_at_rest": math.sqrt(0.01),
                "relative_std": 0.75,
                "progress_weight": 0.5,
            },
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=2.0,
            params={"command_name": "twist", "std": math.sqrt(0.5)},
        ),
        "upright": RewardTermCfg(
            func=mdp.variable_upright,
            weight=1.0,
            params={
                "command_name": "twist",
                "std_standing": math.sqrt(0.20),
                "std_walking": math.sqrt(0.25),
                "std_running": math.sqrt(0.35),
                "walking_threshold": 0.05,
                "running_threshold": 1.5,
                "asset_cfg": SceneEntityCfg("robot", body_names=()),
            },
        ),
        "body_ang_vel": RewardTermCfg(
            func=mdp.body_angular_velocity_penalty,
            weight=-0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=())},
        ),
        "angular_momentum": RewardTermCfg(
            func=mdp.angular_momentum_penalty,
            weight=-0.005,
            params={"sensor_name": "robot/root_angmom"},
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        "air_time": RewardTermCfg(
            func=mdp.feet_air_time,
            weight=0.1,
            params={
                "sensor_name": "feet_ground_contact",
                "threshold_min": 0.05,
                "threshold_max": 0.5,
                "command_name": "twist",
                "command_threshold": 0.2,
            },
        ),
        "foot_clearance": RewardTermCfg(
            func=mdp.feet_clearance,
            weight=-2.0,
            params={
                "target_height": TARGET_FOOT_HEIGHT,
                "height_sensor_name": "foot_height_scan",
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", site_names=()),
            },
        ),
        "foot_swing_height": RewardTermCfg(
            func=mdp.feet_swing_height,
            weight=-0.25,
            params={
                "sensor_name": "feet_ground_contact",
                "height_sensor_name": "foot_height_scan",
                "target_height": TARGET_FOOT_HEIGHT,
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        ),
        "foot_slip": RewardTermCfg(
            func=mdp.feet_slip,
            weight=-0.2,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", site_names=()),
            },
        ),
        "soft_landing": RewardTermCfg(
            func=mdp.soft_landing,
            weight=-0.001,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        ),
        "self_collisions": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-1.0,
            params={"sensor_name": "self_collision"},
        ),
        # Joint set is restricted to policy-controlled joints per robot.
        "standing_pose_l1": RewardTermCfg(
            func=mdp.standing_pose_deviation_l1,
            weight=-0.5,
            params={
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }


def terminations_cfg() -> dict[str, TerminationTermCfg]:
    return {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.stochastic_bad_orientation,
            params={"limit_angle": math.radians(63.0), "probability": 0.02},
        ),
        "illegal_contact": TerminationTermCfg(
            func=mdp.illegal_contact,
            params={"sensor_name": "non_foot_ground_contact"},
        ),
    }


def curriculum_cfg() -> dict[str, CurriculumTermCfg]:
    # mjlab restores common_step_counter on resume; keep absolute stage steps.
    return {
        "soft_landing_weight": CurriculumTermCfg(
            func=mdp.reward_curriculum,
            params={
                "reward_name": "soft_landing",
                "stages": [
                    {"step": 0, "weight": -0.0001},
                    {"step": 1000 * 24, "weight": -0.001},
                    {"step": 7000 * 24, "weight": -0.005},
                ],
            },
        ),
        "command_vel": CurriculumTermCfg(
            func=mdp.commands_vel,
            params={
                "command_name": "twist",
                "velocity_stages": [
                    {"step": 0, "lin_vel_x": (-1.0, 1.2), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-1.0, 1.0)},
                    {"step": 5000 * 24, "lin_vel_x": (-1.0, 1.5), "lin_vel_y": (-1.25, 1.25), "ang_vel_z": (-1.25, 1.25)},
                    {"step": 10000 * 24, "lin_vel_x": (-1.25, 1.5), "lin_vel_y": (-1.5, 1.5), "ang_vel_z": (-1.5, 1.5)},
                    {"step": 15000 * 24, "lin_vel_x": (-1.5, 1.75), "lin_vel_y": (-1.75, 1.75), "ang_vel_z": (-1.5, 1.5)},
                ],
            },
        ),
    }


def make_loco_teleop_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base loco-teleop task configuration."""
    return ManagerBasedRlEnvCfg(
        scene=scene_cfg(),
        observations=observations_cfg(),
        actions=actions_cfg(),
        commands=commands_cfg(),
        events=events_cfg(),
        rewards=rewards_cfg(),
        terminations=terminations_cfg(),
        curriculum=curriculum_cfg(),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",  # Set per robot.
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=50,
            njmax=300,
            contact_sensor_maxmatch=64,
            mujoco=MujocoCfg(
                timestep=0.005, iterations=10, ls_iterations=20, ccd_iterations=60
            ),
        ),
        # Physics at 200 Hz, policy at 200 / 4 = 50 Hz.
        decimation=4,
        episode_length_s=20.0,
    )
