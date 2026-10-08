"""Booster K1 wearing the printed 50 mm heel shoe.

Shoe meshes ``xmls/meshes/{Left,Right}_Shoe_50mm.STL`` are built from the printing files in
``assets/heels/`` (``build_heel50mm.py``) and are expressed in the foot_link frame: the
same Left_Foot.STL frame as mjlab's k1.xml, so they mount
without an offset. In the shoe the sole is pitched toe-down by SHOE_PITCH and sits
SHOE_DROP below the bare sole.

What changes on the robot:
- ``<side>_foot_collision`` uses the shoe mesh instead of the foot mesh, so only the
  shoe touches the ground, and every term keyed on that geom (friction randomization,
  collision settings) applies to the shoe;
- a visible copy of the shoe;
- the shoe mass (uniform density) is merged into the foot link's inertia;
- the ``<side>_foot_sole`` sites move to the shoe underside (foot height scan, clearance).

What changes in the motion data (AMP expert + reset poses), as in the BeyondMimic shoe
project: ankle pitch += SHOE_PITCH (clamped to the soft limit) so the shoe sole is flat
where the bare sole was flat, and the root is raised by SHOE_DROP.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from booster_mjlab import BOOSTER_MJLAB_SRC_PATH
from booster_mjlab.robots.booster_k1.k1_constants import K1_JOINT_ORDER

MESH_DIR = BOOSTER_MJLAB_SRC_PATH / "robots" / "booster_k1" / "xmls" / "meshes"
SHOE_MASS = 0.35  # kg per shoe
SHOE_PITCH = np.deg2rad(15.62)  # sole pitch in the shoe (toe down)
SHOE_DROP = 0.0585  # m, shoe sole below the bare sole
ANKLE_PITCH_SOFT_MAX = 0.3105  # 0.9 x hard limit 0.345
SOLE_SITE_XY = (0.027364, 0.001141)  # k1.xml <side>_foot_sole (y mirrored on the right)


def _shoe_file(side: str) -> str:
    return str(MESH_DIR / f"{side.capitalize()}_Shoe_50mm.STL")


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _merge_mass(body, mesh_file: str, mass: float) -> None:
    """Add a uniform-density solid of ``mass`` kg (mesh in body frame) to the body inertial."""
    import trimesh

    tm = trimesh.load(mesh_file)
    tm.density = mass / tm.volume
    m2, c2, i2 = mass, np.asarray(tm.center_mass), np.asarray(tm.moment_inertia)
    m1, c1 = float(body.mass), np.asarray(body.ipos, dtype=float)
    r = _quat_to_mat(np.asarray(body.iquat, dtype=float))
    i1 = r @ np.diag(np.asarray(body.inertia, dtype=float)) @ r.T
    m = m1 + m2
    c = (m1 * c1 + m2 * c2) / m

    def shift(i, mi, ci):
        d = ci - c
        return i + mi * (d @ d * np.eye(3) - np.outer(d, d))

    evals, evecs = np.linalg.eigh(shift(i1, m1, c1) + shift(i2, m2, c2))
    if np.linalg.det(evecs) < 0:
        evecs[:, 0] *= -1
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, evecs.flatten())
    body.mass, body.ipos, body.iquat, body.inertia = m, c, q, evals


def _underside_z(mesh_file: str, x: float, y: float) -> float:
    """Height of the shoe's underside below (x, y) in the foot frame (vertical ray)."""
    import trimesh

    # Ground contact uses the convex hull (it spans the arch gap), so ray the hull.
    tri = trimesh.load(mesh_file).convex_hull.triangles  # (n, 3, 3)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    # Barycentric coordinates of (x, y) in each triangle's xy projection.
    v0, v1 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    v2 = np.array([x, y]) - a[:, :2]
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    ok = np.abs(den) > 1e-12
    den = np.where(ok, den, 1.0)
    u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    inside = ok & (u >= 0) & (v >= 0) & (u + v <= 1)
    if not inside.any():
        raise ValueError(f"no shoe surface under ({x}, {y}) in {mesh_file}")
    z = a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (c[:, 2] - a[:, 2])
    return float(z[inside].min())


def shoe_spec_fn(base_spec_fn, mass: float = SHOE_MASS):
    def spec_fn() -> mujoco.MjSpec:
        spec = base_spec_fn()
        for side, sy in (("left", 1.0), ("right", -1.0)):
            f = _shoe_file(side)
            mesh = spec.add_mesh()
            mesh.name, mesh.file = f"{side.capitalize()}_Shoe_50mm", f
            body = spec.body(f"{side}_foot_link")
            spec.geom(f"{side}_foot_collision").meshname = mesh.name
            vis = body.add_geom()
            vis.name = f"{side}_shoe_visual"
            vis.type = mujoco.mjtGeom.mjGEOM_MESH
            vis.meshname = mesh.name
            vis.contype = vis.conaffinity = 0
            vis.group = 2
            vis.rgba = (0.95, 0.95, 0.95, 1.0)
            if mass > 0:
                _merge_mass(body, f, mass)
            site = spec.site(f"{side}_foot_sole")
            x, y = SOLE_SITE_XY[0], sy * SOLE_SITE_XY[1]
            site.pos = (x, y, _underside_z(f, x, y))
        return spec

    return spec_fn


def apply_shoe(cfg, mass: float = SHOE_MASS) -> None:
    """Put the shoes on the K1 of an env cfg (robot spec + spawn height)."""
    robot = cfg.scene.entities["robot"]
    robot.spec_fn = shoe_spec_fn(robot.spec_fn, mass)
    x, y, z = robot.init_state.pos
    robot.init_state = dataclasses.replace(robot.init_state, pos=(x, y, z + SHOE_DROP))


_ANKLE_PITCH = [K1_JOINT_ORDER.index(n) for n in ("Left_Ankle_Pitch", "Right_Ankle_Pitch")]


def shoe_motion_transform(motion):
    """Flat-foot reference -> shoe reference: ankle pitch += SHOE_PITCH, root += SHOE_DROP."""
    dof = np.array(motion.dof_pos, dtype=np.float32, copy=True)
    dof[:, _ANKLE_PITCH] = np.minimum(dof[:, _ANKLE_PITCH] + SHOE_PITCH, ANKLE_PITCH_SOFT_MAX)
    root = np.array(motion.root_pos, dtype=np.float32, copy=True)
    root[:, 2] += SHOE_DROP
    return dataclasses.replace(motion, dof_pos=dof, root_pos=root)


SHOE_MOTION_TRANSFORM = "booster_mjlab.tasks.heels.shoe:shoe_motion_transform"
