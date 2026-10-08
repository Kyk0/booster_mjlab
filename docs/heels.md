# K1 walking in 50 mm heel shoes

The K1 wearing 3D-printed 50 mm heel shoes (350 g each). A policy trained bare-foot gets
less stable in the shoes; warm-starting it in the shoes for a few hundred iterations
fixes most of it. Works for both the full-body walking policy and the
[loco-teleop](loco_teleop.md) policy.

Shoe CAD / print files: [k1-heel-shoes](https://github.com/Kyk0/k1-heel-shoes).

Simulation only so far (mjlab + booster_deploy MuJoCo). Not tested on the real robot yet.

Page with videos: [`pages/heels.html`](../pages/heels.html)

## What changes

`src/booster_mjlab/tasks/heels/shoe.py`

Robot (`apply_shoe(cfg)`):
- `<side>_foot_collision` uses the shoe mesh instead of the foot mesh, so only the shoe
  touches the ground; friction randomization keyed on that geom applies to the shoe
- the shoe mass (uniform density) is merged into the foot link inertia
- `<side>_foot_sole` sites move to the shoe underside (foot height, clearance terms)
- spawn height + 58.5 mm

Reference motions (AMP expert data and reset poses, `shoe_motion_transform`):
- ankle pitch + 15.62 deg (the shoe sole pitch), clamped to the soft limit, so the shoe
  sole is flat where the bare sole was flat
- root + 58.5 mm (shoe sole below the bare sole)

AMP style reward weight 0.5 in all these tasks (upstream default 0.3), same as the bare-foot
runs they start from (train those with 0.5 too).

Training: warm start from a bare-foot checkpoint (actor + critic weights; optimizer and
iteration counter start fresh, so curricula restart).

## Tasks

| task | base |
|---|---|
| `Mjlab-Velocity-Flat-Amp-DA-Shoe50mm-Muon-Booster-K1` | full-body velocity task (`Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-K1`), symmetric runner |
| `Mjlab-LocoTeleop-Flat-Amp-Lower-ArmMotion-Walk-Shoe50mm-Muon-Booster-K1` | loco-teleop, legs only, random arm motion |

## Warm start

The checkpoint is loaded from `logs/rsl_rl/<experiment>/<load-run>/<load-checkpoint>`,
so copy the bare-foot checkpoint into the shoe experiment first:

```bash
mkdir -p logs/rsl_rl/k1_velocity_amp_symmetric_shoe50mm_muon/warmstart
cp <bare-foot run>/model_2000.pt logs/rsl_rl/k1_velocity_amp_symmetric_shoe50mm_muon/warmstart/

uv run train Mjlab-Velocity-Flat-Amp-DA-Shoe50mm-Muon-Booster-K1 --env.scene.num-envs 4096 \
    --agent.max-iterations 1500 \
    --agent.resume True --agent.load-run warmstart --agent.load-checkpoint model_2000.pt
```

Loco-teleop: same with experiment `k1_loco_teleop_lower_armmotion_walk_shoe50mm_muon` and
a `...-Lower-ArmMotion-Walk-Muon-...` checkpoint.

## Results (sim)

- Loco-teleop, 512 envs, 40 s, full arm motion: falls 28% bare-foot policy in shoes ->
  3.9% after warm start (model 750). Plateau after ~300 iterations (vs ~4000 from scratch).
  Still a bit less stable than bare feet (0-2%).
- Full-body walk, booster_deploy MuJoCo in shoes: the bare-foot policy drifts 1.4 m while
  standing; the warm-started one (1400 iterations) stands still and tracks 0.5 -> 0.50,
  1.0 -> 0.95 m/s.

## Deploy

booster_deploy tasks `k1_walk_shoe50mm` and `k1_loco_teleop_*` in
[`deploy/booster_deploy/`](../deploy/booster_deploy/README.md). In the shoes the start /
prepare pose gets +0.2726 rad on both ankle pitches (without it the robot starts tipped
forward). `scripts/build_shoe_deploy_xml.py` builds the MuJoCo model with shoes for sim2sim.
