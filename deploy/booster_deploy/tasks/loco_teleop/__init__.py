import os

from booster_deploy.utils.registry import register_task

from .loco_teleop import k1_loco_teleop_cfg

_MODELS = os.path.join(os.path.dirname(__file__), "models")


def _have(model: str) -> bool:
    return os.path.isfile(os.path.join(_MODELS, model, "policy.pt"))


# armmotion_walk: legs trained with big arm circles / sweeps / jumps over the whole range,
# AMP style from LAFAN normal walking/running (lafan_locomotion_k1).
if _have("armmotion_walk"):
    # arms from UDP teleop (scripts/arm_teleop_udp.py), default pose without packets
    register_task("k1_loco_teleop_walk", k1_loco_teleop_cfg("armmotion_walk", upper_body_mode="udp", upper_body_max_rate=15.0))
    # arms move by themselves, like in training (sim stress test)
    register_task("k1_loco_teleop_walk_auto", k1_loco_teleop_cfg("armmotion_walk", upper_body_mode="motion", upper_body_max_rate=15.0))
    # smooth, self-collision-free arm motion from safe_arm_motion.json (sim tests and real-robot demos)
    register_task("k1_loco_teleop_walk_smooth", k1_loco_teleop_cfg("armmotion_walk", upper_body_mode="smooth"))

# Full-body walking policies (standard velocity task, the policy drives all 22 joints).
# Arms are the policy's own, so the upper-body source is unused ("default").
# built by scripts/build_shoe_deploy_xml.py (machine-specific paths, not in git)
_SHOE_XML = os.path.join(os.path.dirname(__file__), "assets", "K1_22dof_shoe50mm.xml")
_SHOE_Z = 0.53 + 0.0585  # spawn higher by the shoe's sole drop
_SHOE_ANKLE = 0.2726  # sole pitch in the shoe (15.62 deg): keeps the soles flat at start / prepare
if _have("fullbody_walk"):
    register_task("k1_walk", k1_loco_teleop_cfg("fullbody_walk", upper_body_mode="default"))
    if os.path.isfile(_SHOE_XML):
        # bare-foot trained policy wearing the shoes (comparison only)
        register_task(
            "k1_walk_flat_in_shoes",
            k1_loco_teleop_cfg("fullbody_walk", upper_body_mode="default", mjcf_path=_SHOE_XML, init_z=_SHOE_Z, ankle_offset=_SHOE_ANKLE),
        )
if _have("fullbody_walk_shoe50mm"):
    # warm-started in the 50 mm heel shoes: use THIS one on the robot when it wears the shoes
    register_task(
        "k1_walk_shoe50mm",
        k1_loco_teleop_cfg(
            "fullbody_walk_shoe50mm",
            upper_body_mode="default",
            mjcf_path=_SHOE_XML if os.path.isfile(_SHOE_XML) else None,
            init_z=_SHOE_Z,
            ankle_offset=_SHOE_ANKLE,
        ),
    )
