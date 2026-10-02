"""Joint torque vs max and rated torque — one small chart per joint.

  walking:   torque vs forward speed
    python scripts/torque_map.py --mode walk --policy pretrained/policy.onnx
    python scripts/torque_map.py --mode walk --policy <onnx> --min -1 --max 2 --points 13
  holding:   torque vs push force
    python scripts/torque_map.py --mode hold --policy pretrained/pos_hold_policy.onnx
    python scripts/torque_map.py --mode hold --policy <onnx> --min 0 --max 300 --points 13

Each chart has three lines:
  red    MAX torque     RS04 120 N.m, RS03 60 N.m   (hip pitch, hip roll, knee = RS04;
  black  RATED torque   RS04  40 N.m, RS03 20 N.m    hip yaw, ankle = RS03)
  blue   torque the robot actually used (highest seen; --metric rms for the sustained value)

Data points:  --points  how many speeds / forces between --min and --max
              --robots  how many robots are tested at each point (more = more reliable)
Robots that fall are left out (the terminal and CSV say how many).
Hold mode: each robot gets one push on the torso (--duration s, from --direction), measured
over the push and the 2 s after it. A walking policy in hold mode just stands still.

Output: logs/torque_maps/<mode>_<policy>_<time>.png and .csv
"""

import argparse
import csv
import importlib
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oc1_rl import robot  # noqa: E402

JN = list(robot.JOINT_NAMES)
MAX_T = robot.EFFORT_LIMIT.astype(float)


def find_module(cls_name):
  """First module under oc1_rl/ that defines `cls_name` (e.g. oc1_rl/walk/env.py)."""
  pat = re.compile(rf"^class\s+{cls_name}\b", re.M)
  for f in sorted((ROOT / "oc1_rl").rglob("*.py")):
    rel = f.relative_to(ROOT).with_suffix("")
    if "__pycache__" in rel.parts or not pat.search(f.read_text(errors="ignore")):
      continue
    try:
      return importlib.import_module(".".join(rel.parts))
    except Exception:  # noqa: BLE001
      continue
  return None


def label(j):
  return j.replace("_pitch", "").replace("right_", "R ").replace("left_", "L ").replace("_", " ")


def load_policy(path):
  import onnx
  m = onnx.load(str(path))
  for v in list(m.graph.input) + list(m.graph.output):
    v.type.tensor_type.shape.dim[0].dim_param = "batch"
  return ort.InferenceSession(m.SerializeToString())


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--policy", required=True)
  p.add_argument("--mode", required=True, choices=["walk", "hold"])
  p.add_argument("--min", type=float, default=None, help="walk: m/s (default -1)  hold: N (default 0)")
  p.add_argument("--max", type=float, default=None, help="walk: m/s (default 2)   hold: N (default 250)")
  p.add_argument("--points", type=int, default=10, help="number of speeds / forces to test")
  p.add_argument("--robots", type=int, default=8, help="robots tested at each point")
  p.add_argument("--metric", default="peak", choices=["peak", "rms"],
                 help="peak = highest torque seen (default); rms = sustained torque")
  p.add_argument("--duration", type=float, default=0.2, help="hold: push length (s)")
  p.add_argument("--direction", default="all", choices=["all", "front", "back", "side"],
                 help="hold: push direction (all = 8 directions spread over the robots)")
  p.add_argument("--rated-rs04", type=float, default=40.0)
  p.add_argument("--rated-rs03", type=float, default=20.0)
  p.add_argument("--seconds", type=float, default=5.0, help="walk: seconds of walking")
  p.add_argument("--dr", action="store_true", help="sensor noise + randomization")
  p.add_argument("--task", default=None, choices=["walk", "pos_hold"],
                 help="env the policy was trained in (default: pos_hold if 'pos_hold' is in the path)")
  p.add_argument("--nthread", type=int, default=os.cpu_count())
  p.add_argument("--out", default=None)
  p.add_argument("--show", action="store_true")
  a = p.parse_args()

  lo = a.min if a.min is not None else (-1.0 if a.mode == "walk" else 0.0)
  hi = a.max if a.max is not None else (2.0 if a.mode == "walk" else 250.0)
  xs = np.round(np.linspace(lo, hi, a.points), 4)
  rated = np.where(MAX_T >= 100, a.rated_rs04, a.rated_rs03).astype(float)
  task = a.task or ("pos_hold" if "pos_hold" in str(a.policy) else "walk")
  if a.mode == "walk" and task == "pos_hold":
    raise SystemExit("--mode walk needs a walking policy")

  W = find_module("OC1VelocityEnv")
  if W is None:
    raise SystemExit("walking env (class OC1VelocityEnv) not found in oc1_rl/")
  n = a.points * a.robots
  x_of = np.repeat(xs, a.robots)
  kw = dict(num_envs=n, nthread=a.nthread, obs_noise=a.dr, domain_randomization=a.dr,
            pushes=False, terminate_on_timeout=False, seed=0)
  if task == "pos_hold":
    Pm = find_module("OC1PositionHoldEnv")
    if Pm is None:
      raise SystemExit("position-hold env (class OC1PositionHoldEnv) not found in oc1_rl/")
    f = Pm.PosHoldCfg.__dataclass_fields__
    kw.update({k: v for k, v in dict(anchor_offset_prob=0.0, reanchor_prob=0.0,
                                     max_drift=1e9).items() if k in f})
    env = Pm.OC1PositionHoldEnv(Pm.PosHoldCfg(**kw))
  else:
    env = W.OC1VelocityEnv(W.EnvCfg(**kw))
  sess = load_policy(a.policy)
  if sess.get_inputs()[0].shape[1] != env.observations()[0].shape[1]:
    raise SystemExit("this policy does not match the environment (different number of inputs)")

  DT = W.STEP_DT
  base = env.model.body(robot.BASE_BODY).id
  mass = float(env.model.body_subtreemass[base])
  has_xfrc = hasattr(env, "xfrc") and hasattr(env, "_ctrl_spec")
  zero = np.zeros((n, 3))

  if a.mode == "walk":
    cmd = np.stack([x_of, np.zeros(n), np.zeros(n)], 1)
    stand, total = 1.0, 1.0 + a.seconds
    window = (stand + 1.5, total)          # skip the first 1.5 s of speeding up
    push = None
  else:
    cmd = zero
    stand, t_push = 0.0, 1.5
    total = t_push + a.duration + 2.0
    window = (t_push, total)
    ang = {"front": 180.0, "back": 0.0, "side": 90.0}.get(a.direction)
    ang = (np.radians(np.full(n, ang)) if ang is not None
           else np.radians(np.tile(np.arange(0, 360, 45.0), n // 8 + 1)[:n]))
    yaw = W.yaw_of(env.qpos()[:, 3:7]) + ang
    push = np.stack([x_of * np.cos(yaw), x_of * np.sin(yaw), np.zeros(n)], 1)

  print(f"policy: {a.policy}   mode: {a.mode}   {a.points} points x {a.robots} robots = {n} robots")
  t0 = time.time()
  if task != "pos_hold":
    env.command_override = zero
    env._update_commands()
  obs, _ = env.observations()
  fell = np.zeros(n, bool)
  peak = np.zeros((n, len(JN)))
  sq = np.zeros((n, len(JN)))
  cnt = np.zeros(n)
  for k in range(round(total / DT)):
    t = k * DT
    if task != "pos_hold":
      env.command_override = cmd if t >= stand else zero
    if push is not None:
      on = (t >= t_push - 1e-9) & (t < t_push + a.duration - 1e-9) & ~fell
      if has_xfrc:
        xf = np.zeros((n, env.model.nbody, 6))
        xf[on, base, 0:3] = push[on]
        env.xfrc = xf if on.any() else None
      elif abs(t - t_push) < DT / 2:
        env.qvel()[:, 0:3] += push * a.duration / mass * (~fell)[:, None]
    obs, _, _, done, _, _ = env.step(sess.run(None, {"obs": obs})[0])
    fell |= done
    if window[0] <= t < window[1]:
      target = env.default_q + env.action * env.action_scale
      sub = getattr(env, "_states_out", None)
      if sub is not None and sub.ndim == 3:        # all physics sub-steps
        q, qd = sub[..., env.qpos_slice][..., 7:], sub[..., env.qvel_slice][..., 6:]
      else:
        q, qd = env.qpos()[:, None, 7:], env.qvel()[:, None, 6:]
      tau = np.clip(robot.KP * (target[:, None] - q) - robot.KD * qd, -MAX_T, MAX_T)
      ok = ~fell
      peak[ok] = np.maximum(peak[ok], np.abs(tau[ok]).max(1))
      sq[ok] += (tau[ok] ** 2).mean(1)
      cnt[ok] += 1
  if has_xfrc:
    env.xfrc = None
  env.close()
  rms = np.sqrt(sq / np.maximum(cnt, 1)[:, None])
  val = peak if a.metric == "peak" else rms
  print(f"simulated in {time.time() - t0:.0f} s")

  # per point: worst robot (peak) or average robot (rms), fallen robots left out
  Y = np.full((a.points, len(JN)), np.nan)
  fall = np.zeros(a.points)
  for i, x in enumerate(xs):
    m = x_of == x
    ok = m & ~fell
    fall[i] = 100 * fell[m].mean()
    if ok.any():
      Y[i] = val[ok].max(0) if a.metric == "peak" else val[ok].mean(0)

  # ---------------- plot: one chart per joint (top row right leg, bottom row left leg)
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                       "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
                       "axes.labelcolor": "#52514e", "xtick.color": "#898781",
                       "ytick.color": "#898781", "axes.spines.top": False,
                       "axes.spines.right": False, "axes.grid": True, "grid.color": "#e1e0d9",
                       "font.size": 9.5, "axes.titleweight": "bold", "axes.titlesize": 11})
  half = len(JN) // 2
  fig, axes = plt.subplots(2, half, figsize=(3.9 * half, 7.4), sharex=True)
  xlabel = "forward speed (m/s)" if a.mode == "walk" else "push force (N)"
  what = "highest torque" if a.metric == "peak" else "sustained (RMS) torque"
  for j, ax in enumerate(axes.flat):
    ax.axhline(MAX_T[j], color="#d03b3b", lw=2, ls="--", label=f"max {MAX_T[j]:.0f}")
    ax.axhline(rated[j], color="#0b0b0b", lw=1.6, ls="--", label=f"rated {rated[j]:.0f}")
    ax.plot(xs, Y[:, j], color="#2a78d6", lw=2.2, marker="o", ms=4.5, label=what)
    bad = fall > 0
    ax.scatter(xs[bad], Y[bad, j], s=46, facecolor="#fcfcfb", edgecolor="#eb6834", lw=1.8,
               zorder=5, label="some robots fell here")
    ax.set_ylim(0, MAX_T[j] * 1.12)
    ax.set_title(label(JN[j]), loc="left")
    ax.text(0.98, 0.98, f"max {MAX_T[j]:.0f} · rated {rated[j]:.0f} N·m", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#898781")
    if j % half == 0:
      ax.set_ylabel("torque (N·m)")
    if j >= half:
      ax.set_xlabel(xlabel)
  h, l_ = axes.flat[0].get_legend_handles_labels()
  names = ["max torque", "rated torque", f"used by the robot ({what})",
           "some robots fell at this point (survivors were stumbling)"][:len(h)]
  fig.legend(h, names, loc="upper right", ncol=len(h), frameon=False,
             bbox_to_anchor=(0.995, 0.995), fontsize=9.5)
  title = "Walking" if a.mode == "walk" else f"Holding position, {a.duration:g} s push ({a.direction})"
  fig.suptitle(f"Joint torque — {title}", x=0.01, ha="left", y=0.985, fontsize=15, fontweight="bold")
  fig.text(0.01, 0.925, f"{a.policy}   ·   {a.points} points × {a.robots} robots"
           + (f"   ·   robots that fell are left out (up to {fall.max():.0f}% at one point)"
              if fall.max() > 0 else ""), color="#898781", fontsize=9)
  fig.tight_layout(rect=(0, 0, 1, 0.91))

  pol = Path(a.policy)
  name = pol.parent.name if pol.stem == "policy" else pol.stem
  out = Path(a.out) if a.out else ROOT / "logs" / "torque_maps" / (
    f"{a.mode}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', name)}_{datetime.now():%Y-%m-%d_%H-%M-%S}.png")
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out, dpi=120)
  with open(out.with_suffix(".csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["speed_m_s" if a.mode == "walk" else "force_N", "fell_pct"]
               + [f"{j}_{a.metric}_Nm" for j in JN])
    for i, x in enumerate(xs):
      w.writerow([x, round(fall[i], 1)] + [round(float(v), 2) for v in Y[i]])

  print(f"\n{'joint':12s} {'max':>5s} {'rated':>6s} {'highest used':>13s}")
  for j in range(len(JN)):
    top = np.nanmax(Y[:, j])
    flag = "  AT MAX" if top >= MAX_T[j] * 0.999 else ("  above rated" if top > rated[j] else "")
    print(f"{label(JN[j]):12s} {MAX_T[j]:5.0f} {rated[j]:6.0f} {top:13.1f}{flag}")
  print(f"\nplot: {out}\ndata: {out.with_suffix('.csv')}")
  if a.show:
    import subprocess
    subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(out)])


if __name__ == "__main__":
  main()