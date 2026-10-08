"""Booster K1 loco-teleop environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef

from booster_mjlab.robots import K1_ACTION_SCALE, get_k1_robot_cfg
from booster_mjlab.robots.booster_k1.sensors import (
    FootClearanceSensorCfg,
    FootSoleGridPatternCfg,
)
from booster_mjlab.tasks.loco_teleop import mdp
from booster_mjlab.tasks.loco_teleop.loco_teleop_env_cfg import (
    make_loco_teleop_env_cfg,
)

# Policy-controlled joints and externally driven joints (regex on k1.xml joint names).
LEG_JOINTS = (r".*_Hip_.*", r".*_Knee_.*", r".*_Ankle_.*")
UPPER_JOINTS = (r"Head_.*", r".*_Shoulder_.*", r".*_Elbow_.*")

FOOT_SITES = ("left_foot", "right_foot")
FOOT_SOLE_SITES = ("left_foot_sole", "right_foot_sole")
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")

# K1_ACTION_SCALE has one entry per actuator group; keep only the leg groups, since
# mjlab raises on scale keys that match none of the action's joints.
LEG_ACTION_SCALE = {
    k: v
    for k, v in K1_ACTION_SCALE.items()
    if any(part in k for part in ("Hip", "Knee", "Ankle"))
}

# How often the upper-body targets jump to a new random pose during training.
UPPER_BODY_RESAMPLE_S = (1.0, 3.0)
# Full-range variant: faster resampling to cover quick teleop motions.
UPPER_BODY_FULL_RANGE_RESAMPLE_S = (0.3, 2.0)


# Adapted from booster_mjlab tasks/velocity/config/k1/env_cfgs.py (upstream terms apply).
def _k1_sensors() -> tuple:
    feet_ground = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_foot_link|right_foot_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    nonfoot_ground = ContactSensorCfg(
        name="non_foot_ground_contact",
        primary=ContactMatch(
            mode="body",
            entity="robot",
            pattern=r".*",
            exclude=("left_foot_link", "right_foot_link"),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    self_collision = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    foot_height_scan = FootClearanceSensorCfg(
        name="foot_height_scan",
        frame=tuple(
            ObjRef(type="site", name=s, entity="robot") for s in FOOT_SOLE_SITES
        ),
        pattern=FootSoleGridPatternCfg(),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
    )
    return (feet_ground, nonfoot_ground, self_collision, foot_height_scan)


def add_upper_body_events(cfg: ManagerBasedRlEnvCfg, noise: float) -> None:
    """Drive the upper body: hold default pose (noise=0) or jump to random poses."""
    upper = SceneEntityCfg("robot", joint_names=UPPER_JOINTS)
    cfg.events["upper_body_reset"] = EventTermCfg(
        func=mdp.set_upper_body_targets,
        mode="reset",
        params={"asset_cfg": upper, "noise": noise},
    )
    cfg.events.pop("upper_body_teleop", None)
    if noise > 0.0:
        cfg.events["upper_body_teleop"] = EventTermCfg(
            func=mdp.set_upper_body_targets,
            mode="interval",
            interval_range_s=UPPER_BODY_RESAMPLE_S,
            params={"asset_cfg": upper, "noise": noise},
        )


def add_full_range_upper_body_events(cfg: ManagerBasedRlEnvCfg) -> None:
    """Upper body over its whole joint range, with quick jumps (see the event fn)."""
    upper = SceneEntityCfg("robot", joint_names=UPPER_JOINTS)
    for name, mode, extra in (
        ("upper_body_reset", "reset", {}),
        ("upper_body_teleop", "interval", {"interval_range_s": UPPER_BODY_FULL_RANGE_RESAMPLE_S}),
    ):
        cfg.events[name] = EventTermCfg(
            func=mdp.randomize_upper_body_full_range,
            mode=mode,
            params={"asset_cfg": upper},
            **extra,
        )


def add_motion_upper_body_events(cfg: ManagerBasedRlEnvCfg) -> None:
    """Upper body: big circles / sweeps / jumps over the whole range (see event fn)."""
    upper = SceneEntityCfg("robot", joint_names=UPPER_JOINTS)
    cfg.events["upper_body_reset"] = EventTermCfg(
        func=mdp.upper_body_motion, mode="reset", params={"asset_cfg": upper, "reset": True}
    )
    cfg.events["upper_body_teleop"] = EventTermCfg(
        func=mdp.upper_body_motion, mode="step", params={"asset_cfg": upper}
    )


def booster_k1_loco_teleop_flat_env_cfg(
    play: bool = False,
    upper_body_noise: float = 0.0,
    full_range: bool = False,
    arm_motion: bool = False,
    shoe: bool = False,
) -> ManagerBasedRlEnvCfg:
    """K1 on flat ground: legs by policy, upper body by events (or teleop)."""
    cfg = make_loco_teleop_env_cfg()

    # Scene.
    cfg.scene.entities = {"robot": get_k1_robot_cfg()}
    if shoe:
        from booster_mjlab.tasks.heels.shoe import apply_shoe

        apply_shoe(cfg)
    cfg.scene.sensors = _k1_sensors()
    cfg.viewer.body_name = "Trunk"

    # Actions: legs only (12 of 22 joints).
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.actuator_names = LEG_JOINTS
    joint_pos_action.scale = LEG_ACTION_SCALE

    # Upper body.
    if arm_motion:
        add_motion_upper_body_events(cfg)
    elif full_range:
        add_full_range_upper_body_events(cfg)
    else:
        add_upper_body_events(cfg, upper_body_noise)

    # Commands.
    cfg.commands["twist"].viz.z_offset = 1.15

    # K1 names for robot-agnostic terms.
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
    cfg.events["trunk_inertia"].params["asset_cfg"].body_names = ("Trunk",)
    cfg.events["limb_inertia"].params["asset_cfg"].body_names = (r"(?!Trunk$).*",)
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("Trunk",)
    for name in ("foot_clearance", "foot_slip"):
        cfg.rewards[name].params["asset_cfg"].site_names = FOOT_SITES

    # Only penalize joints the policy can actually move.
    cfg.rewards["standing_pose_l1"].params["asset_cfg"].joint_names = LEG_JOINTS
    cfg.rewards["dof_pos_limits"].params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)
    }

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.terminations.pop("illegal_contact", None)
        ranges = cfg.commands["twist"].ranges
        ranges.lin_vel_x = (-2.5, 2.5)
        ranges.lin_vel_y = (-2.0, 2.0)
        ranges.ang_vel_z = (-3.7, 3.7)

    return cfg
