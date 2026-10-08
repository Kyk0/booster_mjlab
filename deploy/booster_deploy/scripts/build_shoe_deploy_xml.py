"""Build the booster_deploy MuJoCo model of the K1 wearing the printed 50 mm heel shoes.

Starts from booster_assets K1_22dof.xml (motor actuators, the model the MuJoCo
controller expects), adds the shoe meshes to <side>_foot_link (same Left_Foot.STL frame
as the shoe meshes) as collision + visual geoms, and merges the shoe mass into the foot
link inertia (uniform density) -- matching the booster-mjlab Shoe50mm training tasks.

Run with a python that has mujoco + trimesh (e.g. the booster-mjlab venv):

    PYTHONPATH=<booster_assets>/src <booster_mjlab>/.venv/bin/python \
        scripts/build_shoe_deploy_xml.py --shoe-dir <booster_mjlab>/src/booster_mjlab/robots/booster_k1/xmls/meshes
"""

from __future__ import annotations

import argparse
import os

import mujoco
import numpy as np
import trimesh


def merge_mass(body, mesh_file: str, mass: float) -> None:
    tm = trimesh.load(mesh_file)
    tm.density = mass / tm.volume
    m2, c2, i2 = mass, np.asarray(tm.center_mass), np.asarray(tm.moment_inertia)
    m1, c1 = float(body.mass), np.asarray(body.ipos, dtype=float)
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(body.iquat, dtype=float))
    r = r.reshape(3, 3)
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


def main() -> None:
    from booster_assets import BOOSTER_ASSETS_DIR

    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=f"{BOOSTER_ASSETS_DIR}/robots/K1/K1_22dof.xml")
    p.add_argument("--shoe-dir", required=True, help="dir with Left_Shoe_50mm.STL / Right_Shoe_50mm.STL")
    p.add_argument("--mass", type=float, default=0.35, help="kg per shoe")
    p.add_argument("--out", default=os.path.join(here, "..", "tasks", "loco_teleop", "assets", "K1_22dof_shoe50mm.xml"))
    a = p.parse_args()

    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    spec = mujoco.MjSpec.from_file(a.base)
    spec.meshdir = os.path.join(os.path.dirname(os.path.abspath(a.base)), "meshes")  # absolute: XML lives elsewhere
    base_mass = spec.compile().body_subtreemass[1]
    for side in ("left", "right"):
        src = os.path.join(os.path.expanduser(a.shoe_dir), f"{side.capitalize()}_Shoe_50mm.STL")
        dst = os.path.join(out_dir, os.path.basename(src))
        trimesh.load(src).export(dst)  # keep a copy next to the XML
        mesh = spec.add_mesh()
        mesh.name, mesh.file = f"{side}_shoe", dst
        body = spec.body(f"{side}_foot_link")
        col = body.add_geom()
        col.name, col.type, col.meshname = f"{side}_shoe_collision", mujoco.mjtGeom.mjGEOM_MESH, mesh.name
        col.rgba = (0.95, 0.95, 0.95, 1.0)
        col.friction = (0.8, 0.005, 0.0001)
        merge_mass(body, dst, a.mass)
    model = spec.compile()
    with open(a.out, "w") as f:
        f.write(spec.to_xml())
    print(f"[shoe_xml] wrote {a.out}: total mass {base_mass:.3f} -> {model.body_subtreemass[1]:.3f} kg")


if __name__ == "__main__":
    main()
