"""Booster K1 walking in 3D-printed 50 mm heel shoes.

``shoe`` puts the shoes on the robot (collision + mass) and shifts the AMP / reset motion
data so the shoe sole is flat where the bare sole was. ``fullbody`` registers the
full-body velocity task in the shoes; the loco-teleop package has a shoe variant too.
"""
from . import fullbody

__all__ = ["fullbody"]
