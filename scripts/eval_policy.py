"""Evaluate an ONNX policy in the batched environment and report per-term rewards.

  python scripts/eval_policy.py --policy runs/<run>/policy.onnx
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oc1_rl.env import EnvCfg, OC1VelocityEnv, REWARD_WEIGHTS, STEP_DT, quat_rotate_inverse  # noqa: E402
from oc1_rl.paths import resolve_policy  # noqa: E402


def load_batched_session(path):
  """Open an ONNX policy, relaxing a fixed batch size of 1 to a dynamic one."""
  import onnx

  model = onnx.load(str(path))
  for v in list(model.graph.input) + list(model.graph.output):
    v.type.tensor_type.shape.dim[0].dim_param = "batch"
  return ort.InferenceSession(model.SerializeToString())


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--policy", default="latest", help="ONNX path, run dir or 'latest'")
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--seconds", type=float, default=19.0)
  p.add_argument("--train-conditions", action="store_true",
                 help="enable obs noise, domain randomization and pushes")
  args = p.parse_args()

  path = resolve_policy(args.policy)
  print(f"policy: {path}")
  sess = load_batched_session(path)
  t = args.train_conditions
  env = OC1VelocityEnv(EnvCfg(num_envs=args.num_envs, obs_noise=t, domain_randomization=t,
                             pushes=t, seed=123))
  obs, _ = env.reset()
  sums = {k: 0.0 for k in REWARD_WEIGHTS}
  lin_err, falls, steps = 0.0, 0, int(args.seconds / STEP_DT)
  t0 = time.time()
  for _ in range(steps):
    act = sess.run(None, {"obs": obs})[0]
    cmd = env.command.copy()
    obs, _, _, done, time_out, extras = env.step(act)
    q = env.qpos()[:, 3:7]
    v = quat_rotate_inverse(q, env.qvel()[:, 0:3])
    lin_err += np.mean(np.linalg.norm(v[:, :2] - cmd[:, :2], axis=1))
    falls += int(np.sum(done & ~time_out))
  # Include partially finished episodes.
  for k, v in env.episode_sums.items():
    sums[k] = float(np.mean(v)) / args.seconds
  dt = time.time() - t0
  print(f"{args.num_envs} envs x {args.seconds:.0f}s simulated in {dt:.1f}s "
        f"({args.num_envs * steps / dt:.0f} steps/s)")
  print(f"falls: {falls} ({falls / args.num_envs:.2f} per env)")
  print(f"mean xy velocity tracking error: {lin_err / steps:.3f} m/s")
  print("reward per second by term (last episode):")
  for k, v in sums.items():
    print(f"  {k:24s} {v:+.4f}")
  print(f"  {'TOTAL':24s} {sum(sums.values()):+.4f}")
  env.close()


if __name__ == "__main__":
  main()
