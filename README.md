# OC1 biped velocity policy — trained on a Mac

The OC1 biped (10 joints, 26.3 kg, `urdf/oc1_bipedal.urdf`) trained to walk by following
velocity commands. It uses the exact training setup from `~/Desktop/g1_mac_rl`
(the Unitree-G1-Flat task from unitree_rl_mjlab re-built for CPU MuJoCo):

- same PPO (asymmetric actor-critic, 512-256-128 ELU, 24 steps × 2048 envs, 5 epochs,
  4 mini-batches, adaptive LR 1e-3 / KL 0.01, entropy 0.01, clip 0.2, γ 0.99, λ 0.95)
- same 16 reward terms and weights, commands, curriculum, pushes, friction / CoM /
  encoder randomization, observation noise, 20 s episodes, 70° fall limit
- same PD recipe (Kp = armature·ω², Kd = 2·ζ·armature·ω, 10 Hz, ζ = 2,
  action_scale = 0.25·torque_limit/Kp), 200 Hz physics, 50 Hz policy
- same checkpoint + video schedule: `oc1_ppo_<steps>_steps.pt/.onnx` and
  `videos/step_<steps>.mp4` every 1M steps, 3000 iterations by default

## Setup

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
```


## Pretrained policy

A trained policy is included, so you can try the robot without training anything:

- `pretrained/policy.onnx`: the trained walking policy
- `pretrained/demo.mp4`: full video of the policy walking

Pass it with `--policy`:

```bash
.venv/bin/mjpython scripts/play.py --policy pretrained/policy.onnx
.venv/bin/python scripts/record_video.py --policy pretrained/policy.onnx --view grid
.venv/bin/python scripts/eval_policy.py --policy pretrained/policy.onnx
```

Without `--policy`, the scripts use the newest run in `runs/`, so they only work after you have trained your own policy.


## Train

```bash
caffeinate -i .venv/bin/python scripts/train.py                        # new run
caffeinate -i .venv/bin/python scripts/train.py --resume latest --iterations 5000
pkill -f scripts/train.py            # stop (saves a checkpoint + video first)
.venv/bin/tensorboard --logdir runs  # training curves
```

```
runs/<date>/
├── checkpoints/   oc1_ppo_<steps>_steps.pt (+ .onnx)   every 1M steps (--save-every-steps)
├── videos/        step_<steps>.mp4                      recorded at every checkpoint
├── logs/          progress.csv, TensorBoard events, config.json, train.log
└── policy.onnx    newest policy (default for play.py / eval_policy.py / record_video.py)
```

## Watch / record / score
By default these use your newest run in `runs/`. To use the included policy, add `--policy pretrained/policy.onnx`.

```bash
.venv/bin/mjpython scripts/play.py                         # ↑/↓ speed, ←/→ turn, ,/. sideways, 0 stop
.venv/bin/python scripts/record_video.py --view grid       # front / right / back / top
.venv/bin/python scripts/record_video.py --view all        # one video per camera angle
.venv/bin/python scripts/eval_policy.py --train-conditions # falls, speed error, reward per term
```

Policy input: 41 numbers (base gyro 3, gravity 3, command 3, gait clock 2, joint pos 10,
joint vel 10, last action 10). Output: 10 joint actions in `JOINT_NAMES` order
(right leg first: hip pitch, hip roll, hip yaw, knee, ankle; then the left leg).
Joint target = default pose + action × action_scale.

## What is specific to the OC1 (`oc1_rl/robot.py`)

| | |
|---|---|
| Base frame | The URDF root (`torso_ss`) faces +y with the left leg on −x. A `base` link is added at the hip centre, rotated −90° in yaw so the robot faces +x, left is +y, up is +z. |
| Motors | Hip pitch, hip roll, knee: Robstride RS04 (120 N·m peak, armature 0.04 → Kp 158, Kd 10.1). Hip yaw, ankle: RS03 (60 N·m, armature 0.02 → Kp 79, Kd 5.0). Action scale 0.19 rad. The URDF `effort="1"` placeholder is ignored. **Armature values are estimates** — replace with datasheet rotor inertia × gear ratio² if you have them. |
| Home pose | Slight knee bend as on the G1: hip 0.1, knee 0.3, ankle 0.2 rad (signs follow each joint's axis). Base height 0.79 m. Joint limits are the URDF's ±0.5 rad. |
| Foot contact | 7 spheres (r = 1 cm) on each sole, friction 0.6 (randomized 0.3–1.6), like the G1. Foot meshes don't touch the floor; all other meshes do. |
| Collision meshes | Convex hulls of the URDF collision meshes, thinned to 64 points each (≤ 7 mm from the full hull). Full CAD meshes made each env's model copy 84 MB; now 0.9 MB. Self-collision stays on. |
| Posture tolerances | G1 values for the matching joints (the OC1 has no ankle roll, waist or arms). |

## Files

| File | What it does |
|---|---|
| `oc1_rl/robot.py` | Builds the MuJoCo model from the URDF: base frame, actuators and gains, foot spheres, sensors, floor. |
| `oc1_rl/env.py` | The task: 2048 robots stepped in parallel, observations, rewards, commands, pushes, randomization, resets (G1 env with the robot swapped). |
| `oc1_rl/ppo.py` | PPO (identical to `g1_rl/ppo.py`). |
| `scripts/train.py` | Training loop, checkpoints + ONNX + video every 1M steps, `--resume`. |
| `scripts/record_video.py`, `play.py`, `eval_policy.py` | Video, interactive viewer, scoring. |
| `scripts/joint_test.py`, `check_collisions.py` | Earlier model-checking tools (unchanged). |


