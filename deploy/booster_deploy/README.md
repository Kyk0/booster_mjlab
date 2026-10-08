# booster_deploy task: K1 loco-teleop and heel-shoe walking

Drop-in task for [booster_deploy](https://github.com/BoosterRobotics/booster_deploy)
(Apache-2.0): copy `tasks/loco_teleop` into `booster_deploy/tasks/` and the files in
`scripts/` into `booster_deploy/scripts/`.

Simulation only so far (booster_deploy MuJoCo). Not tested on the real robot yet.

## Tasks

| task | model | upper body |
|---|---|---|
| `k1_loco_teleop_walk` | `models/armmotion_walk` (legs only) | UDP teleop (`scripts/arm_teleop_udp.py`), default pose if silent for 1 s |
| `k1_loco_teleop_walk_smooth` | same | smooth self-collision-free motion from `safe_arm_motion.json` |
| `k1_loco_teleop_walk_auto` | same | fast random motion like in training (sim stress test) |
| `k1_walk` | `models/fullbody_walk` (full body, bare feet) | policy |
| `k1_walk_shoe50mm` | `models/fullbody_walk_shoe50mm` (warm-started in 50 mm heel shoes) | policy |
| `k1_walk_flat_in_shoes` | bare-foot model in the shoe sim (comparison only) | policy |

`models/<name>/policy.pt` + `deploy_meta.json` come from
`booster_mjlab.scripts.export_loco_teleop_deploy`; joint order, default pose, PD gains and
action scale are read from the json, so they match training. The models were trained with
AMP style from LAFAN1 motions (CC BY-NC-ND 4.0): non-commercial use.

Policy details:
- observations built exactly as in training; legs = default + action x scale
- targets ramp in over 1.5 s from the start pose; upper-body targets clipped to joint
  limits and rate limited; stop if the trunk tilts past ~60 deg
- heading hold when the yaw command is 0
- MuJoCo torque limits = the training motor limits (the K1 config limits in booster_deploy
  are 2-3x lower and make the policy shuffle and fall in sim)
- shoe tasks: +0.2726 rad on both ankle pitches of the start / prepare pose

## MuJoCo

```bash
cd <booster_deploy>
export PYTHONPATH=<booster_assets>/src

# sim + legs; type "x y yaw" + Enter to walk, e.g. 0.5 0 0
python scripts/deploy.py --task k1_loco_teleop_walk --mujoco
# second terminal: arms (keys 1-5 poses, [ ] select joint, - = move, r random)
python scripts/arm_teleop_udp.py

# headless check
python scripts/sim_check_loco_teleop.py --task k1_loco_teleop_walk_auto --vx 0.5
# viser viewer in the browser
python scripts/deploy_viser.py --task k1_loco_teleop_walk_smooth
```

Shoe sim: build the MuJoCo model with shoes once (needs mujoco + trimesh), then run
`k1_walk_shoe50mm`:

```bash
python scripts/build_shoe_deploy_xml.py --shoe-dir <booster_mjlab>/src/booster_mjlab/robots/booster_k1/xmls/meshes
python scripts/deploy.py --task k1_walk_shoe50mm --mujoco
```

## Real robot (untested)

```bash
rsync -av tasks/loco_teleop <robot>:~/booster_deploy/tasks/
rsync -av scripts/arm_teleop_udp.py <robot>:~/booster_deploy/scripts/
# on the robot
source /opt/booster/BoosterRos2Interface/install/setup.bash
cd ~/booster_deploy && python3 scripts/deploy.py --task k1_loco_teleop_walk
# arms, from a machine on the robot network
python3 scripts/arm_teleop_udp.py --host <robot_ip>
```

`x` = prepare pose, `r` = start the policy, `w/s a/d q/e` = velocity, `space` = stop.
First runs: robot in a harness, arms at default, zero command; then standing, slow walk,
slow arm poses.

## Safe arm motion library

`tasks/loco_teleop/safe_arm_motion.json`: 149 upper-body poses and 413 transitions, built
by `scripts/build_safe_arm_motion.py` on the robot meshes' convex hulls. Every pose and
every transition (checked every 0.05 rad) keeps >= 5 cm between the arms/head and the
trunk, legs and other arm, for 16 leg poses recorded while walking. At runtime: min-jerk
moves between neighbouring poses, peak speed 1.5 rad/s.
