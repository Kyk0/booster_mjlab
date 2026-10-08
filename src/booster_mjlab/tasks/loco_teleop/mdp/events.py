"""Loco-teleop event terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def set_upper_body_targets(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    noise: float = 0.0,
) -> None:
    """Set PD targets of joints no action term controls (teleop stand-in).

    mjlab zeroes every joint target on reset and never touches joints outside the
    action terms, so without this the upper body would servo to 0 rad. Use
    ``mode="reset"`` to hold the default pose and ``mode="interval"`` to move the
    upper body around while the legs balance.

    Targets are ``default_joint_pos + U(-noise, noise)``, clipped to soft joint limits.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    joint_ids = asset_cfg.joint_ids
    if not isinstance(joint_ids, slice):
        # A list would be paired elementwise with env_ids instead of outer-indexed.
        joint_ids = torch.as_tensor(joint_ids, device=env.device)

    target = asset.data.default_joint_pos[env_ids][:, joint_ids].clone()
    if noise > 0.0:
        target += (torch.rand_like(target) * 2.0 - 1.0) * noise
        limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids]
        target = target.clamp(limits[..., 0], limits[..., 1])

    asset.set_joint_position_target(target, joint_ids=joint_ids, env_ids=env_ids)


def randomize_upper_body_full_range(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    near_noise: float = 0.6,
    full_range_prob: float = 0.7,
    ramp_steps: int = 1500 * 24,
) -> None:
    """Teleop stand-in covering the whole upper-body workspace.

    Each call draws a new target per env: with probability ``p`` uniform over the
    soft joint limits of every upper-body joint (arms overhead, behind, sideways,
    head turned), otherwise ``default + U(-near_noise, near_noise)``. ``p`` ramps
    linearly from 0 to ``full_range_prob`` over ``ramp_steps`` env steps so the
    legs learn to walk before facing the full range. Targets jump, so the arms
    move as fast as the motors allow (quick teleop motions).
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    joint_ids = asset_cfg.joint_ids
    if not isinstance(joint_ids, slice):
        joint_ids = torch.as_tensor(joint_ids, device=env.device)

    default = asset.data.default_joint_pos[env_ids][:, joint_ids]
    limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids]
    lo, hi = limits[..., 0], limits[..., 1]

    near = default + (torch.rand_like(default) * 2.0 - 1.0) * near_noise
    full = lo + torch.rand_like(default) * (hi - lo)
    p = full_range_prob * min(1.0, env.common_step_counter / max(1, ramp_steps))
    use_full = torch.rand(len(env_ids), 1, device=env.device) < p
    target = torch.where(use_full, full, near).clamp(lo, hi)

    asset.set_joint_position_target(target, joint_ids=joint_ids, env_ids=env_ids)


_MODE_JUMP, _MODE_CIRCLE, _MODE_SWEEP = 0, 1, 2


def upper_body_motion(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    reset: bool = False,
    mode_probs: tuple[float, float, float] = (0.25, 0.5, 0.25),
    segment_s: tuple[float, float] = (1.0, 4.0),
    freq_hz: tuple[float, float] = (0.2, 1.5),
    jump_hold_s: tuple[float, float] = (0.3, 1.5),
    amp_scale: float | None = None,
    ramp_steps: int = 1000 * 24,
    circle_pairs: tuple[tuple[str, str], ...] = (
        ("Left_Shoulder_Pitch", "Left_Shoulder_Roll"),
        ("Right_Shoulder_Pitch", "Right_Shoulder_Roll"),
    ),
) -> None:
    """Teleop stand-in with large continuous motions over the whole joint range.

    Call every step (``mode="interval"`` with the step period) and on reset
    (``reset=True``). Every ``segment_s`` each env picks a mode:
      jump    jump to a uniform random pose every ``jump_hold_s``
      circle  shoulder pitch/roll trace circles (cos/sin, same frequency), the
              other joints follow independent sines
      sweep   every joint follows an independent sine
    Sine amplitudes go up to half the joint range (i.e. limit to limit) and are
    scaled by ``amp_scale`` (default: ramps 0.3 -> 1.0 over ``ramp_steps``).
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    joint_ids = asset_cfg.joint_ids
    if not isinstance(joint_ids, slice):
        joint_ids = torch.as_tensor(joint_ids, device=env.device)
    dev = env.device

    st = getattr(env, "_upper_body_motion", None)
    if st is None:
        lim = asset.data.soft_joint_pos_limits[0][joint_ids]
        names = [asset.joint_names[i] for i in joint_ids.tolist()]
        pairs = [(names.index(a), names.index(b)) for a, b in circle_pairs if a in names and b in names]
        n, j = env.num_envs, len(names)
        z = lambda *s: torch.zeros(*s, device=dev)  # noqa: E731
        st = {
            "lo": lim[:, 0], "hi": lim[:, 1], "pairs": pairs,
            "mode": torch.zeros(n, dtype=torch.long, device=dev),
            "t": z(n), "seg": z(n), "jump_t": z(n),
            "w": z(n, j), "phase": z(n, j), "amp": z(n, j), "center": z(n, j),
            "jump": asset.data.default_joint_pos[:, joint_ids].clone(),
        }
        env._upper_body_motion = st

    lo, hi = st["lo"], st["hi"]
    half = (hi - lo) / 2
    scale = amp_scale
    if scale is None:
        scale = min(1.0, 0.3 + 0.7 * env.common_step_counter / max(1, ramp_steps))

    def u(shape, a, b):
        return a + torch.rand(shape, device=dev) * (b - a)

    def uniform_pose(k):
        return lo + torch.rand(k, len(lo), device=dev) * (hi - lo)

    # New segment on reset or when the current one is over.
    if not reset:
        st["t"][env_ids] += env.step_dt
    new = env_ids if reset else env_ids[st["t"][env_ids] >= st["seg"][env_ids]]
    if len(new):
        k, j = len(new), len(lo)
        st["mode"][new] = torch.multinomial(
            torch.tensor(mode_probs, device=dev), k, replacement=True
        )
        st["t"][new] = 0.0
        st["seg"][new] = u(k, *segment_s)
        amp = scale * u((k, j), 0.3, 1.0) * half
        st["amp"][new] = amp
        st["center"][new] = u((k, j), 0.0, 1.0) * (hi - lo - 2 * amp) + lo + amp
        w = 2 * torch.pi * u((k, j), *freq_hz)
        phase = u((k, j), 0.0, 2 * torch.pi)
        circle = st["mode"][new] == _MODE_CIRCLE
        for a, b in st["pairs"]:  # roll follows pitch a quarter period later
            w[circle, b] = w[circle, a]
            phase[circle, b] = phase[circle, a] + torch.pi / 2
        st["w"][new], st["phase"][new] = w, phase
        st["jump"][new] = uniform_pose(k)
        st["jump_t"][new] = u(k, *jump_hold_s)

    # Jump mode: hold a pose, re-draw when its hold time is up.
    is_jump = st["mode"][env_ids] == _MODE_JUMP
    jump_ids = env_ids[is_jump]
    if len(jump_ids) and not reset:
        st["jump_t"][jump_ids] -= env.step_dt
        redo = jump_ids[st["jump_t"][jump_ids] <= 0]
        if len(redo):
            st["jump"][redo] = uniform_pose(len(redo))
            st["jump_t"][redo] = u(len(redo), *jump_hold_s)

    t = st["t"][env_ids].unsqueeze(-1)
    sine = st["center"][env_ids] + st["amp"][env_ids] * torch.sin(
        st["w"][env_ids] * t + st["phase"][env_ids]
    )
    target = torch.where(is_jump.unsqueeze(-1), st["jump"][env_ids], sine)
    target = target.clamp(lo, hi)
    asset.set_joint_position_target(target, joint_ids=joint_ids, env_ids=env_ids)
