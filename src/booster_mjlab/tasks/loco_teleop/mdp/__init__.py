"""MDP term functions for the loco-teleop task."""

from mjlab.tasks.velocity.mdp import *  # noqa: F403

from booster_mjlab.mdp.terminations import stochastic_bad_orientation
from booster_mjlab.mdp.terrain import randomize_terrain_contact
from booster_mjlab.tasks.velocity.mdp.rewards import (
    standing_pose_deviation_l1,
    track_linear_velocity,
    variable_upright,
)
from booster_mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

from .events import (
    randomize_upper_body_full_range,
    set_upper_body_targets,
    upper_body_motion,
)
