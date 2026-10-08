# K1 loco-teleop: legs walk, upper body is free

A lower-body walking policy for the Booster K1: the policy drives the 12 leg joints from a
velocity command, the head and arms (10 joints) are driven from outside (teleop). During
training the arms are moved by a random motion generator over their whole range, so the
legs learn to keep balance whatever the arms do.

Simulation only so far (mjlab training + booster_deploy MuJoCo sim2sim). Not tested on
the real robot yet.

Page with videos: [`pages/loco-teleop.html`](../pages/loco-teleop.html)

## Compared to the default `Mjlab-Velocity-Flat-Amp` K1 task

| | default velocity task | loco-teleop |
|---|---|---|
| actions | 22 joints | **12 leg joints** |
| actor obs | 75 | **65**: pos/vel of all 22 joints, last action (12), command, IMU |
| arms | policy output, kept near default by `upper_body_posture` | **part of the environment**, moved by a generator |
| rewards | ... + `upper_body_posture` | `upper_body_posture` removed; `standing_pose_l1`, `dof_pos_limits` on legs only |
| AMP | legs-only discriminator, style weight 0.3 | same discriminator, **style weight 0.5** |
| algo | PPO + Muon, MLP 512-256-128, 4096 envs | same |

## Arm motion during training

`src/booster_mjlab/tasks/loco_teleop/mdp/events.py::upper_body_motion`, an mjlab event
with `mode="step"` (every policy step, 50 Hz) plus one on reset, vectorized over all envs.
It sets the PD targets of the head and arm joints. Every 1-4 s each env picks one of:

- circles (50%): shoulder pitch/roll on a circle, the other joints sines
- sweeps (25%): every joint its own sine
- jumps (25%): random pose over the full range every 0.3-1.5 s

Frequency 0.2-1.5 Hz, amplitude 30-100% of half the joint range. Curriculum: amplitude
x0.3 -> x1.0 over the first 1000 iterations. Arm speed in sim: ~7 rad/s typical, peaks
15-18 rad/s. No motion dataset: the parameters are random per segment.

## Tasks

| task | arms during training |
|---|---|
| `Mjlab-LocoTeleop-Flat-Amp-Lower-Muon-Booster-K1` | held at the default pose |
| `...-Lower-Rand-Muon-...` | default ± 0.6 rad, new pose every 1-3 s |
| `...-Lower-FullArms-Muon-...` | random pose anywhere in range every 0.3-2 s |
| `...-Lower-ArmMotion-Muon-...` | the generator above, AMP style from `--agent.dataset-root` |
| `...-Lower-ArmMotion-Walk-Muon-...` | the generator above, AMP style from LAFAN walking (default dataset) |
| `...-Lower-ArmMotion-Walk-Shoe50mm-Muon-...` | same, wearing 50 mm heel shoes, see [heels.md](heels.md) |

Every task also exists without `-Muon` (Adam).

## Results (sim)

512 envs, 40 s, walking, % of envs that fell:

| arms during evaluation | trained arms still | trained Rand | trained ArmMotion |
|---|---|---|---|
| circles over the full range | 100% | 25% | 0.7-2.5% |

In booster_deploy MuJoCo with the arms up: ArmMotion keeps the trunk pitch at -1..-2 deg,
Rand leans -16..-18 deg.

Training: ~2 s per iteration (4096 envs x 24 steps) on one RTX 6000, 4000 iterations.

## Commands

```bash
# train
uv run train Mjlab-LocoTeleop-Flat-Amp-Lower-ArmMotion-Walk-Muon-Booster-K1 \
    --env.scene.num-envs 4096 --agent.max-iterations 4000

# viser viewer with keyboard teleop for velocity and arms
uv run python -m booster_mjlab.scripts.teleop Mjlab-LocoTeleop-Flat-Amp-Lower-ArmMotion-Walk-Muon-Booster-K1 \
    --checkpoint-file <run>/model_3999.pt --port 8080 --vx 0.5

# robustness comparison between checkpoints (falls under different arm conditions)
uv run python -m booster_mjlab.scripts.compare_upper_body --policy a=<ckpt> --policy b=<ckpt> \
    --env-task Mjlab-LocoTeleop-Flat-Amp-Lower-ArmMotion-Walk-Muon-Booster-K1 \
    --conditions near:0 full:1.0 motion:1.0 --out compare_out

# full-body policy with its arm actions overridden by the same generator
uv run python -m booster_mjlab.scripts.compare_fullbody_arms --help

# export for booster_deploy (TorchScript + deploy_meta.json)
uv run python -m booster_mjlab.scripts.export_loco_teleop_deploy --checkpoint <run>/model_3999.pt \
    --task Mjlab-LocoTeleop-Flat-Amp-Lower-ArmMotion-Walk-Muon-Booster-K1 \
    --out deploy/booster_deploy/tasks/loco_teleop/models/armmotion_walk
```

Videos: `scripts/render_pair.py`, `render_fullbody_arms.py`, `render_shoe_compare.py`
(offscreen MuJoCo, `--light` for a white background), `make_sbs_video.py` /
`make_grid_video.py` to put them side by side.

## Deploy

[`deploy/booster_deploy/`](../deploy/booster_deploy/README.md): a task for
[booster_deploy](https://github.com/BoosterRobotics/booster_deploy) with UDP arm teleop,
a library of self-collision-free arm poses for smooth demo motion, and MuJoCo checks.
