"""Watch a policy walk in the MuJoCo viewer (use `mjpython` on macOS).

  .venv/bin/mjpython scripts/play.py                    # latest trained run
  .venv/bin/mjpython scripts/play.py --policy runs/<run>/policy.onnx

Keys: Up/Down forward speed, Left/Right turn, ,/. sideways, 0 stop, Backspace reset.

When the turn rate is 0 the robot holds its current heading (same heading controller
as in training), so it walks in a straight line instead of slowly drifting.
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oc1_rl import robot  # noqa: E402
from oc1_rl.env import STEP_DT, EnvCfg, OC1VelocityEnv, wrap_to_pi, yaw_of  # noqa: E402
from oc1_rl.paths import resolve_policy  # noqa: E402

KEYS = {265: (0, 0.1), 264: (0, -0.1), 263: (2, 0.2), 262: (2, -0.2), 44: (1, 0.1), 46: (1, -0.1)}
LIMITS = np.array([[-1.0, 2.0], [-1.0, 1.0], [-1.0, 1.0]])


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--policy", default="latest", help="ONNX path, run dir or 'latest'")
  p.add_argument("--vx", type=float, default=0.5)
  p.add_argument("--vy", type=float, default=0.0)
  p.add_argument("--wz", type=float, default=0.0)
  args = p.parse_args()

  path = resolve_policy(args.policy)
  print(f"policy: {path}")
  sess = ort.InferenceSession(str(path))

  model = robot.make_model(visual=True)
  env = OC1VelocityEnv(EnvCfg(num_envs=1, nthread=1, obs_noise=False, domain_randomization=False,
                             pushes=False, terminate_on_timeout=False), model=model)
  cmd = np.array([args.vx, args.vy, args.wz])
  lock = threading.Lock()
  reset_requested = [False]

  def on_key(k):
    with lock:
      if k in KEYS:
        i, dv = KEYS[k]
        cmd[i] = np.clip(round(cmd[i] + dv, 2), *LIMITS[i])
      elif k == 48:
        cmd[:] = 0.0
      elif k == 259:  # Backspace
        reset_requested[0] = True
      else:
        return
      print(f"cmd  vx={cmd[0]:+.2f}  vy={cmd[1]:+.2f}  wz={cmd[2]:+.2f}")

  def heading():
    return float(yaw_of(env.qpos()[:1, 3:7])[0])

  def command_with_heading_hold():
    nonlocal heading_target
    c = cmd.copy()
    if c[2] == 0.0:
      c[2] = np.clip(0.5 * wrap_to_pi(heading_target - heading()), *LIMITS[2])
    else:
      heading_target = heading()
    return c

  heading_target = heading()
  env.command_override = command_with_heading_hold()
  env._update_commands()
  obs, _ = env.observations()

  data = mujoco.MjData(model)
  print(__doc__)
  with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = model.body(robot.BASE_BODY).id
    viewer.cam.distance = 3.0
    while viewer.is_running():
      t0 = time.perf_counter()
      with lock:
        if reset_requested[0]:
          env.reset_envs(np.array([0]))
          heading_target = heading()
          reset_requested[0] = False
        env.command_override = command_with_heading_hold()
      obs, *_ = env.step(sess.run(None, {"obs": obs})[0])
      with viewer.lock():
        mujoco.mj_setState(model, data, env.state[0], env.state_spec)
        mujoco.mj_forward(model, data)
      viewer.sync()
      time.sleep(max(0.0, STEP_DT - (time.perf_counter() - t0)))
  env.close()


if __name__ == "__main__":
  main()
