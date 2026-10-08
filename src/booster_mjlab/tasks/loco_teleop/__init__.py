"""Loco-teleop task: legs-only walking policy, upper body free for teleop.

The policy walks with the legs (AMP style on legs only); the head and arms are not
policy-controlled and are driven externally (teleop at deploy time, random targets
during training to make the legs robust to upper-body motion).
"""
