"""Record an mp4 of a policy walking (offscreen, no window needed).

  .venv/bin/python scripts/record_video.py --policy runs/<run>/checkpoints/oc1_ppo_1000000_steps.onnx
  .venv/bin/python scripts/record_video.py --view grid     # front/side/back/top in one video
  .venv/bin/python scripts/record_video.py --view orbit    # camera circles the robot 360 deg
  .venv/bin/python scripts/record_video.py --view all      # one video per angle

Views (relative to the robot's heading): follow, orbit, front, back, left, right, top, grid, all.

The command schedule: stand, walk forward, walk faster while turning, walk sideways.
"""

import argparse
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oc1_rl import robot  # noqa: E402
from oc1_rl.env import STEP_DT, EnvCfg, OC1VelocityEnv, yaw_of  # noqa: E402
from oc1_rl.paths import resolve_policy  # noqa: E402

# (start time s, vx, vy, wz)
SCHEDULE = [(0.0, 0.0, 0.0, 0.0), (1.0, 0.5, 0.0, 0.0), (4.0, 1.0, 0.0, 0.5), (7.0, 0.0, 0.4, 0.0)]


def command_at(t):
  return np.array(next(c for s, *c in reversed(SCHEDULE) if t >= s))


# View name -> (azimuth offset from the robot's heading in degrees, elevation in degrees).
# Azimuth equal to the heading puts the camera behind the robot.
VIEWS = {
  "front": (180.0, -10.0),
  "back": (0.0, -10.0),
  "left": (-90.0, -10.0),
  "right": (90.0, -10.0),
  "top": (0.0, -89.0),
}
GRID = ("front", "right", "back", "top")
ALL_VIEWS = ("follow", "orbit", "front", "back", "left", "right", "top", "grid")


def _camera_angles(view, heading_deg, t, seconds):
  if view == "follow":
    return 120.0, -15.0  # Fixed world angle, like the training progress videos.
  if view == "orbit":
    return heading_deg + 180.0 + 360.0 * t / seconds, -15.0
  offset, elevation = VIEWS[view]
  return heading_deg + offset, elevation


def record(policy_path, out_path, seconds=10.0, width=640, height=480, label=None, view="follow"):
  sess = ort.InferenceSession(str(policy_path))
  model = robot.make_model(visual=True)
  model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
  model.vis.global_.offheight = max(model.vis.global_.offheight, height)
  env = OC1VelocityEnv(EnvCfg(num_envs=1, nthread=1, obs_noise=False, domain_randomization=False,
                             pushes=False, terminate_on_timeout=False, seed=0), model=model)
  env.reset_envs(np.array([0]))
  data = mujoco.MjData(model)
  panels = GRID if view == "grid" else (view,)
  ph, pw = (height // 2, width // 2) if view == "grid" else (height, width)
  renderer = mujoco.Renderer(model, height=ph, width=pw)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  cam.trackbodyid = model.body(robot.BASE_BODY).id
  cam.distance = 3.0

  frames, fell = [], False
  env.command_override = command_at(0.0)
  env._update_commands()
  obs, _ = env.observations()
  for i in range(int(seconds / STEP_DT)):
    env.command_override = command_at(i * STEP_DT)
    obs, _, _, done, _, _ = env.step(sess.run(None, {"obs": obs})[0])
    fell |= bool(done[0])
    mujoco.mj_setState(model, data, env.state[0], env.state_spec)
    mujoco.mj_forward(model, data)
    heading = np.degrees(yaw_of(env.qpos()[:1, 3:7])[0])
    images = []
    for panel in panels:
      cam.azimuth, cam.elevation = _camera_angles(panel, heading, i * STEP_DT, seconds)
      cam.distance = 2.5 if panel == "top" else 3.0
      renderer.update_scene(data, cam)
      images.append(renderer.render())
    if view == "grid":
      frames.append(np.concatenate([np.concatenate(images[:2], axis=1),
                                    np.concatenate(images[2:], axis=1)], axis=0))
    else:
      frames.append(images[0])
  renderer.close()
  env.close()

  out_path = Path(out_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimsave(out_path, frames, fps=round(1 / STEP_DT))
  print(f"Saved progress video: {out_path}" + (f" ({label})" if label else "")
        + (" — robot fell at least once" if fell else ""), flush=True)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--policy", default="latest", help="ONNX path, run dir or 'latest'")
  p.add_argument("--out", default=None)
  p.add_argument("--seconds", type=float, default=10.0)
  p.add_argument("--view", default="follow", choices=ALL_VIEWS + ("all",))
  args = p.parse_args()
  path = resolve_policy(args.policy)
  # Default output: the run's videos/ folder (or the current directory).
  run_dir = path.parent.parent if path.parent.name == "checkpoints" else path.parent
  default_dir = run_dir / "videos" if (run_dir / "videos").is_dir() else Path(".")
  if args.view == "all":
    out_dir = Path(args.out) if args.out else default_dir / f"{path.stem}_views"
    for v in ALL_VIEWS:
      record(path, out_dir / f"{v}.mp4", args.seconds, view=v)
    return
  out = args.out or default_dir / f"{path.stem}_{args.view}.mp4"
  record(path, out, args.seconds, view=args.view)


if __name__ == "__main__":
  main()
