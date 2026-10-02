"""Train the OC1 velocity-tracking policy from scratch on CPU (Apple Silicon friendly).

  .venv/bin/python scripts/train.py                      # new run
  .venv/bin/python scripts/train.py --resume latest      # continue the latest run

Each run lives in runs/<date>/ with:
  checkpoints/  oc1_ppo_<steps>_steps.pt (+ .onnx) every --save-every-steps
  videos/       step_<steps>.mp4 recorded at every checkpoint
  logs/         progress.csv, TensorBoard events, config.json
  policy.onnx   the most recent policy (used by play.py / eval_policy.py)
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oc1_rl.env import NUM_ACTOR_OBS, NUM_CRITIC_OBS, EnvCfg, OC1VelocityEnv  # noqa: E402
from oc1_rl.paths import RUNS_DIR, latest_run  # noqa: E402
from oc1_rl.ppo import PPO, ActorCritic, OnnxPolicy, PPOCfg  # noqa: E402
from oc1_rl.robot import NUM_JOINTS  # noqa: E402


def export_onnx(ac, path):
  policy = OnnxPolicy(ac).eval()
  dummy = torch.zeros(1, NUM_ACTOR_OBS)
  kwargs = dict(input_names=["obs"], output_names=["actions"],
                dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}})
  try:
    torch.onnx.export(policy, dummy, str(path), dynamo=False, **kwargs)
  except Exception:
    torch.onnx.export(policy, dummy, str(path), dynamo=True, external_data=False, **kwargs)


SCRIPTS_DIR = Path(__file__).resolve().parent


def save(run_dir, it, total_steps, ac, ppo, env):
  name = f"oc1_ppo_{total_steps}_steps"
  ckpt = run_dir / "checkpoints" / f"{name}.pt"
  torch.save({"iteration": it, "total_steps": total_steps, "model": ac.state_dict(),
              "optimizer": ppo.optimizer.state_dict(), "lr": ppo.lr,
              "common_step": env.common_step}, ckpt)
  export_onnx(ac, ckpt.with_suffix(".onnx"))
  export_onnx(ac, run_dir / "policy.onnx")
  print(f"Saved checkpoint: {ckpt}", flush=True)
  return ckpt


def record_video(run_dir, ckpt, total_steps, previous):
  """Render the checkpoint to videos/step_<steps>.mp4 in a separate process."""
  if previous is not None:
    previous.wait()
  out = run_dir / "videos" / f"step_{total_steps}.mp4"
  return subprocess.Popen([sys.executable, "-W", "ignore", str(SCRIPTS_DIR / "record_video.py"),
                           "--policy", str(ckpt.with_suffix(".onnx")), "--out", str(out)])


def find_checkpoint(run_dir):
  ckpts = list((run_dir / "checkpoints").glob("oc1_ppo_*_steps.pt"))
  if ckpts:
    return max(ckpts, key=lambda f: int(f.stem.split("_")[2]))
  return max(run_dir.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[1]))


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--num-envs", type=int, default=2048)
  p.add_argument("--iterations", type=int, default=3000, help="total iterations (incl. resumed)")
  p.add_argument("--nthread", type=int, default=os.cpu_count())
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--save-every-steps", type=int, default=1_000_000)
  p.add_argument("--no-video", action="store_true")
  p.add_argument("--resume", default=None, help="run dir, or 'latest'")
  args = p.parse_args()

  # Stop cleanly (checkpoint + video) on Ctrl+C or `pkill -f scripts/train.py`,
  # including when running in the background where SIGINT is normally ignored.
  def _stop(*_):
    raise KeyboardInterrupt
  signal.signal(signal.SIGINT, _stop)
  signal.signal(signal.SIGTERM, _stop)

  torch.manual_seed(args.seed)
  torch.set_num_threads(args.nthread)

  cfg = PPOCfg()
  env = OC1VelocityEnv(EnvCfg(num_envs=args.num_envs, nthread=args.nthread, seed=args.seed))
  ac = ActorCritic(NUM_ACTOR_OBS, NUM_CRITIC_OBS, NUM_JOINTS, cfg)
  ppo = PPO(ac, cfg, args.num_envs, NUM_ACTOR_OBS, NUM_CRITIC_OBS, NUM_JOINTS)

  start_it = 0
  if args.resume:
    run_dir = latest_run() if args.resume == "latest" else Path(args.resume)
    ckpt = find_checkpoint(run_dir)
    state = torch.load(ckpt)
    ac.load_state_dict(state["model"])
    ppo.optimizer.load_state_dict(state["optimizer"])
    ppo.lr = state["lr"]
    env.common_step = state["common_step"]
    start_it = state["iteration"] + 1
    print(f"Resumed from {ckpt}")
  else:
    run_dir = RUNS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  for sub in ("checkpoints", "videos", "logs"):
    (run_dir / sub).mkdir(parents=True, exist_ok=True)
  if not args.resume:
    (run_dir / "logs" / "config.json").write_text(json.dumps(
      {"args": vars(args), "ppo": asdict(cfg), "env": asdict(env.cfg)}, indent=2, default=str))
  print(f"Run directory: {run_dir}")

  steps_per_it = cfg.num_steps_per_env * args.num_envs
  total_steps = state.get("total_steps", start_it * steps_per_it) if args.resume else 0
  next_save = (total_steps // args.save_every_steps + 1) * args.save_every_steps
  video_proc = None
  last_saved = total_steps if args.resume else -1

  try:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(run_dir / "logs"))
  except ImportError:
    writer = None
  csv_path = run_dir / "logs" / "progress.csv"
  csv_file = open(csv_path, "a", newline="")
  csv_writer = None

  obs, critic_obs = (torch.from_numpy(x) for x in env.reset())
  ep_logs, rew_buf, len_buf = [], [], []
  cur_rew = np.zeros(args.num_envs)
  cur_len = np.zeros(args.num_envs)
  t_start = time.time()
  it = start_it
  end_it = args.iterations

  try:
    for it in range(start_it, end_it):
      t0 = time.time()
      for _ in range(cfg.num_steps_per_env):
        actions = ppo.act(obs, critic_obs)
        o, co, rew, done, time_out, extras = env.step(actions.numpy())
        obs, critic_obs = torch.from_numpy(o), torch.from_numpy(co)
        ppo.process_step(torch.from_numpy(rew), torch.from_numpy(done.astype(np.float32)),
                         torch.from_numpy(time_out.astype(np.float32)))
        cur_rew += rew
        cur_len += 1
        if done.any():
          rew_buf.extend(cur_rew[done].tolist())
          len_buf.extend(cur_len[done].tolist())
          cur_rew[done] = 0
          cur_len[done] = 0
          ep_logs.append(extras)
      t_collect = time.time() - t0

      ppo.compute_returns(critic_obs)
      stats = ppo.update()
      t_iter = time.time() - t0

      rew_buf, len_buf = rew_buf[-2000:], len_buf[-2000:]
      steps = steps_per_it
      total_steps += steps
      row = {
        "iteration": it,
        "total_steps": total_steps,
        "wall_time_min": (time.time() - t_start) / 60,
        "fps": steps / t_iter,
        "mean_episode_reward": float(np.mean(rew_buf)) if rew_buf else 0.0,
        "mean_episode_length_s": float(np.mean(len_buf)) * 0.02 if len_buf else 0.0,
        "fell_frac": float(np.mean([e["fell_frac"] for e in ep_logs if "fell_frac" in e]))
        if ep_logs else 0.0,
        "action_std": ac.std.detach().mean().item(),
        "lr": ppo.lr,
        **stats,
      }
      for k in ep_logs[0]["episode"] if ep_logs else []:
        row[f"rew/{k}"] = float(np.mean([e["episode"][k] for e in ep_logs]))
      ep_logs = []

      if writer:
        for k, v in row.items():
          if k != "iteration":
            writer.add_scalar(k, v, it)
      if csv_writer is None:
        csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()), extrasaction="ignore")
        if csv_path.stat().st_size == 0:
          csv_writer.writeheader()
      csv_writer.writerow(row)
      csv_file.flush()

      done_its = it - start_it + 1
      eta = (time.time() - t_start) / done_its * (end_it - it - 1) / 3600
      print(f"it {it:5d} | {total_steps / 1e6:6.1f}M | {row['fps']:6.0f} steps/s (collect {t_collect:4.1f}s, total {t_iter:4.1f}s)"
            f" | ep reward {row['mean_episode_reward']:7.2f} | ep len {row['mean_episode_length_s']:5.1f}s"
            f" | fell {row['fell_frac']:.2f} | track_lin {row.get('rew/track_linear_velocity', 0):.3f}"
            f" | gait {row.get('rew/foot_gait', 0):.3f} | std {row['action_std']:.2f} | ETA {eta:.1f}h",
            flush=True)

      if total_steps >= next_save:
        ckpt = save(run_dir, it, total_steps, ac, ppo, env)
        last_saved = total_steps
        if not args.no_video:
          video_proc = record_video(run_dir, ckpt, total_steps, video_proc)
        next_save = (total_steps // args.save_every_steps + 1) * args.save_every_steps
  except KeyboardInterrupt:
    print("\nInterrupted — saving checkpoint.")
  if last_saved != total_steps:
    ckpt = save(run_dir, it, total_steps, ac, ppo, env)
    if not args.no_video:
      video_proc = record_video(run_dir, ckpt, total_steps, video_proc)
  if video_proc is not None:
    video_proc.wait()
  csv_file.close()
  env.close()


if __name__ == "__main__":
  main()
