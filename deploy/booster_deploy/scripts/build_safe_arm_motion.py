"""Build a library of self-collision-free upper-body poses and transitions.

Samples head/arm poses inside the joint limits, keeps those where no arm/head part
comes closer than ``--margin`` to the trunk, head, legs or the other arm (checked
on the visual meshes' convex hulls, for several leg poses of a walking cycle), then
connects pairs of poses whose straight joint-space path is collision-free too.
The result is used by the ``smooth`` upper-body mode (no MuJoCo needed at runtime).

    python scripts/build_safe_arm_motion.py \
        --meta tasks/loco_teleop/models/armmotion_walk/deploy_meta.json \
        --out tasks/loco_teleop/safe_arm_motion.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import mujoco
import numpy as np

sys.path.append(".")

# Moving upper-body parts that can hit something. Left/Right_Arm_1 (the shoulder
# pitch housings) are left out: they spin in place inside the trunk shell, and
# their convex hulls overlap the head / the other shoulder without a real contact.
UPPER_BODIES = (
    "Head_1", "Head_2",
    "Left_Arm_2", "Left_Arm_3", "left_hand_link",
    "Right_Arm_2", "Right_Arm_3", "right_hand_link",
)
SHOULDER_HOUSINGS = ("Left_Arm_1", "Right_Arm_1")
# Leg poses (hip pitch, hip roll, knee, ankle pitch) covering a walking cycle.
LEG_POSES = (
    (-0.4, 0.0, 0.8, -0.4),
    (-0.9, 0.1, 1.2, -0.3),
    (0.2, -0.1, 0.4, -0.5),
    (-0.6, 0.25, 1.0, -0.4),
)


def norm(name: str) -> str:
    return name.lower().lstrip("a")


class Checker:
    def __init__(self, mjcf: str, upper_names: list[str], margin: float, leg_configs=None):
        # Visual-only meshes get no convex hull at compile time, and distances to
        # them are unreliable; mark them collidable so the compiler builds hulls.
        spec = mujoco.MjSpec.from_file(mjcf)
        for g in spec.geoms:
            if g.group == 1:
                g.contype, g.conaffinity = 1, 1
        m = spec.compile()
        self.m, self.d = m, mujoco.MjData(m)
        body = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""  # noqa: E731
        upper_ids = {b for b in range(m.nbody) if body(b) in UPPER_BODIES}

        def chain(b, n=2):  # body and its n ancestors
            out = {b}
            for _ in range(n):
                b = int(m.body_parentid[b])
                out.add(b)
            return out

        # Check distances between visual meshes (group 1): arm/head parts vs the
        # rest of the robot and vs the other arm. Kinematic neighbours (body + 2
        # ancestors) and parts of the same arm are skipped.
        visual = [
            g for g in range(m.ngeom)
            if m.geom_group[g] == 1 and m.geom_bodyid[g] > 0
            and body(int(m.geom_bodyid[g])) not in SHOULDER_HOUSINGS
        ]
        self.upper_ids = upper_ids
        self.chain = {b: chain(b) for b in range(m.nbody)}
        self.side = {b: ("L" if "left" in body(b).lower() else "R" if "right" in body(b).lower() else "") for b in range(m.nbody)}
        pairs = []
        for g1 in visual:
            b1 = int(m.geom_bodyid[g1])
            if b1 not in upper_ids:
                continue
            for g2 in visual:
                b2 = int(m.geom_bodyid[g2])
                if g2 == g1 or (b2 in upper_ids and g2 < g1):
                    continue
                if self._pair_counts(b1, b2):
                    pairs.append((g1, g2))
        self.pairs = np.array(pairs)
        self.margin = margin
        self.thr = np.full(len(pairs), margin)  # per-pair clearance, see set_rest_pose
        self.rbound = m.geom_rbound.copy()
        self._fromto = np.zeros(6)
        jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
        qadr = {norm(n): int(m.jnt_qposadr[j]) for j, n in enumerate(jnames) if n}
        self.upper_q = np.array([qadr[norm(n)] for n in upper_names])
        legs = {}
        for side in ("left", "right"):
            legs[side] = [qadr[f"{side}_{j}"] for j in ("hip_pitch", "hip_roll", "knee_pitch", "ankle_pitch")]
        self.leg_q = legs
        # Full 12-joint leg configurations (robot order) to check against; default:
        # the hand-made walking-cycle poses in LEG_POSES.
        self.leg_names = [n for n in jnames if n and any(k in n for k in ("Hip", "Knee", "Ankle"))]
        self.leg_adr = np.array([qadr[norm(n)] for n in self.leg_names])
        self.leg_configs = None if leg_configs is None else np.asarray(leg_configs)
        self.base = np.array(m.qpos0)
        self.base[3:7] = (1, 0, 0, 0)

    def _pair_counts(self, b1: int, b2: int) -> bool:
        if b1 in self.chain[b2] or b2 in self.chain[b1]:
            return False  # kinematic neighbours always touch near their joint
        s1, s2 = self.side[b1], self.side[b2]
        if b1 in self.upper_ids and b2 in self.upper_ids and s1 and s1 == s2:
            return False  # same arm
        return True

    def free(self, upper: np.ndarray) -> bool:
        return all(self.clear() for _ in self._leg_configs(upper))

    def distances(self) -> np.ndarray:
        m, d = self.m, self.d
        return np.array([
            mujoco.mj_geomDistance(m, d, int(a), int(b), 1.0, self._fromto) for a, b in self.pairs
        ])

    def _leg_configs(self, upper: np.ndarray):
        """Yield after setting qpos/kinematics for every checked leg pose."""
        if self.leg_configs is not None:
            for legs in self.leg_configs:
                q = self.base.copy()
                q[self.upper_q] = upper
                q[self.leg_adr] = legs
                self.d.qpos[:] = q
                mujoco.mj_kinematics(self.m, self.d)
                yield
            return
        for lp in LEG_POSES:
            for side, other in (("left", "right"), ("right", "left")):
                q = self.base.copy()
                q[self.upper_q] = upper
                q[self.leg_q[side]] = lp
                q[self.leg_q[other]] = LEG_POSES[0]
                q[self.leg_q[side][1]] = lp[1] if side == "left" else -lp[1]
                self.d.qpos[:] = q
                mujoco.mj_kinematics(self.m, self.d)
                yield

    def set_rest_pose(self, upper: np.ndarray, slack: float = 0.01) -> None:
        """Pairs already close in the rest pose may not get closer than rest - slack.

        Convex hulls (e.g. of the trunk) are much bigger than the real parts, so a
        fixed margin would reject the pose the robot always stands in. The rest
        distance is the minimum over all checked leg poses.
        """
        rest = np.full(len(self.pairs), np.inf)
        for _ in self._leg_configs(upper):
            rest = np.minimum(rest, self.distances())
        self.thr = np.minimum(self.margin, rest - slack)

    def clear(self) -> bool:
        """True if every checked pair keeps its clearance (bounding-sphere prefilter)."""
        m, d = self.m, self.d
        g1, g2 = self.pairs[:, 0], self.pairs[:, 1]
        centers = np.linalg.norm(d.geom_xpos[g1] - d.geom_xpos[g2], axis=1)
        near = np.nonzero(centers - self.rbound[g1] - self.rbound[g2] < self.thr)[0]
        for k in near:
            a, b = self.pairs[k]
            if mujoco.mj_geomDistance(m, d, int(a), int(b), self.margin + 0.05, self._fromto) < self.thr[k]:
                return False
        return True

    def path_free(self, a: np.ndarray, b: np.ndarray, step: float) -> bool:
        n = max(2, int(np.ceil(np.abs(b - a).max() / step)) + 1)
        return all(self.free(a + (b - a) * t) for t in np.linspace(0, 1, n)[1:-1])


def record_leg_configs(task: str, n: int, seconds: float = 20.0) -> np.ndarray:
    """Leg joint configurations seen while walking (0 / 0.3 / 0.6 / 0.9 m/s), FPS-subsampled."""
    import tasks.loco_teleop  # noqa: F401
    from booster_deploy.controllers.mujoco_controller import MujocoController
    from booster_deploy.utils.registry import get_task

    samples = []
    for vx in (0.0, 0.3, 0.6, 0.9):
        cfg = get_task(task)
        ctrl = MujocoController(cfg)
        ctrl.update_vel_command = lambda: None
        ctrl.policy.cfg.upper_body_mode = "default"
        ctrl.vel_command.lin_vel_x = vx
        ctrl.update_state()
        ctrl.start()
        names = list(ctrl.robot.cfg.joint_names)
        leg_idx = [i for i, nm in enumerate(names) if any(k in nm for k in ("Hip", "Knee", "Ankle"))]
        for i in range(int(seconds / ctrl.cfg.policy_dt)):
            ctrl.update_state()
            if not ctrl.is_running:
                break
            ctrl.ctrl_step(ctrl.policy_step())
            if i > 100 and i % 5 == 0:
                samples.append(ctrl.mj_data.qpos[7:][leg_idx].copy())
    samples = np.array(samples)
    chosen = [samples.mean(0)]
    dist = np.linalg.norm(samples - chosen[0], axis=1)
    for _ in range(n - 1):
        k = int(dist.argmax())
        chosen.append(samples[k])
        dist = np.minimum(dist, np.linalg.norm(samples - samples[k], axis=1))
    print(f"[safe_arm] {len(samples)} walking leg samples -> {n} leg configs")
    return np.array(chosen)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mjcf", default=None)
    p.add_argument("--margin", type=float, default=0.05, help="planning clearance (m)")
    p.add_argument("--legs-from-task", default="k1_loco_teleop_walk_smooth",
                   help="record leg poses by walking this booster_deploy task in MuJoCo ('' = built-in)")
    p.add_argument("--leg-configs", type=int, default=16)
    p.add_argument("--candidates", type=int, default=4000)
    p.add_argument("--poses", type=int, default=120)
    p.add_argument("--neighbors", type=int, default=12)
    p.add_argument("--stones", type=int, default=40, help="tree poses to always keep")
    p.add_argument("--tree-iters", type=int, default=600)
    p.add_argument("--tree-step", type=float, default=0.2)
    p.add_argument("--range-frac", type=float, default=0.9, help="use this fraction of each joint range")
    p.add_argument("--path-step", type=float, default=0.05, help="rad between path checks")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    meta = json.load(open(args.meta))
    names = meta["upper_body_joint_names"]
    lim = np.array(meta["upper_body_joint_limits"])
    default = np.array(meta["upper_body_default_joint_pos"])
    if args.mjcf is None:
        from booster_assets import BOOSTER_ASSETS_DIR

        args.mjcf = f"{BOOSTER_ASSETS_DIR}/robots/K1/K1_22dof.xml"
    leg_configs = record_leg_configs(args.legs_from_task, args.leg_configs) if args.legs_from_task else None
    chk = Checker(args.mjcf, names, args.margin, leg_configs)
    chk.set_rest_pose(default)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    mid, half = lim.mean(1), (lim[:, 1] - lim[:, 0]) / 2 * args.range_frac
    cand = mid + (rng.random((args.candidates, len(names))) * 2 - 1) * half
    safe = [c for c in cand if chk.free(c)]
    print(f"[safe_arm] {len(safe)}/{len(cand)} sampled poses collision-free ({time.time() - t0:.0f}s)")
    default_ok = chk.free(default)
    print(f"[safe_arm] default pose collision-free: {default_ok}")

    # Stepping stones out of the rest pose: grow a tree of small collision-free
    # moves (RRT-style) from the default pose, so it connects to the library.
    tree = [default]
    for _ in range(args.tree_iters):
        goal = mid + (rng.random(len(names)) * 2 - 1) * half
        k = int(np.argmin([np.abs(t - goal).max() for t in tree]))
        step = goal - tree[k]
        step *= min(1.0, args.tree_step / max(np.abs(step).max(), 1e-9))
        q = np.clip(tree[k] + step, lim[:, 0], lim[:, 1])
        if chk.free(q) and chk.path_free(tree[k], q, args.path_step):
            tree.append(q)
    stones = tree[1:]
    far = max((np.abs(t - default).max() for t in stones), default=0.0)
    print(f"[safe_arm] rest-pose tree: {len(stones)} poses, reaching {far:.2f} rad from default")

    # Farthest-point subsample for coverage, starting from the default pose.
    safe = np.array(safe + stones)
    # Keep tree poses spread out (every k-th) so the rest pose stays connected.
    every = max(1, len(stones) // max(1, args.stones))
    chosen = [default] + stones[::every][: args.stones]
    dist = np.linalg.norm(safe - default, axis=1)
    for c in chosen[1:]:
        dist = np.minimum(dist, np.linalg.norm(safe - c, axis=1))
    for _ in range(min(args.poses - len(chosen), len(safe))):
        i = int(dist.argmax())
        chosen.append(safe[i])
        dist = np.minimum(dist, np.linalg.norm(safe - safe[i], axis=1))
    poses = np.array(chosen)

    # Edges: k nearest neighbours with a collision-free straight path.
    edges: dict[int, list[int]] = {i: [] for i in range(len(poses))}
    for i in range(len(poses)):
        order = np.argsort(np.linalg.norm(poses - poses[i], axis=1))[1 : args.neighbors + 1]
        for j in order:
            j = int(j)
            if j in edges[i]:
                continue
            if chk.path_free(poses[i], poses[j], args.path_step):
                edges[i].append(j)
                edges[j].append(i)
    n_edges = sum(len(v) for v in edges.values()) // 2
    reach = {0}
    stack = [0]
    while stack:
        for j in edges[stack.pop()]:
            if j not in reach:
                reach.add(j)
                stack.append(j)
    print(f"[safe_arm] {len(poses)} poses, {n_edges} safe transitions, {len(reach)} reachable from default ({time.time() - t0:.0f}s)")

    keep = sorted(reach)
    remap = {old: new for new, old in enumerate(keep)}
    json.dump(
        {
            "joint_names": names,
            "margin_m": args.margin,
            "range_frac": args.range_frac,
            "mjcf": args.mjcf,
            "leg_configs_checked": (leg_configs.round(3).tolist() if leg_configs is not None else LEG_POSES),
            "poses": poses[keep].round(4).tolist(),
            "edges": [[remap[j] for j in edges[i] if j in remap] for i in keep],
        },
        open(args.out, "w"),
    )
    span = poses[keep]
    for k, n in enumerate(names):
        print(f"  {n:22s} covered [{span[:, k].min():+.2f}, {span[:, k].max():+.2f}]  limits [{lim[k, 0]:+.2f}, {lim[k, 1]:+.2f}]")
    print(f"[safe_arm] wrote {args.out}")


if __name__ == "__main__":
    main()
