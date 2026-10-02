"""twin_viewer.py - 3D digital twin of the OC1 during hardware bring-up (runs on your laptop).

LIVE: the robot (or the sim) sends its state here while bringup.py runs.
    .venv/bin/mjpython scripts/hardware/twin_viewer.py
  then, on the Pi:
    python3 scripts/hardware/bringup.py --twin <this laptop's IP>
  or, to try it with the sim on the laptop (second terminal):
    .venv/bin/python scripts/hardware/bringup.py --sim --twin 127.0.0.1

REPLAY: play back any CSV that bringup.py saved.
    .venv/bin/mjpython scripts/hardware/twin_viewer.py --replay runs/hardware/<date>/policy_air_step.csv
    keys: space = pause, R = restart, [ / ] = slower / faster

What you see:
  solid robot        measured joint angles, body tilted like the IMU says (hung at a fixed height)
  blue ghost         where the joints are TOLD to go (target).  Solid and ghost apart = motor
                     not following (too little current, wrong sign, wrong zero, ...)
  link colours       torque of the motor that moves that link:
                     grey = motor off, green = low, orange = near rated, red = above rated
  yellow link        the joint the hand test wants you to move (the ghost shows which way)
  text, top left     step, instructions, stop reason
  text, top right    every joint: angle, target, torque % of rated, temperature

Find the laptop IP:  ipconfig getifaddr en0   (Mac).  The laptop and the Pi must be on the same
network, and macOS may ask to allow incoming connections for Python the first time - say Allow.
"""

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = next((d for d in Path(__file__).resolve().parents if (d / "oc1_rl").is_dir()),
            Path(__file__).resolve().parents[2])   # project folder, wherever this file is
sys.path.insert(0, str(ROOT))
from oc1_rl import robot  # noqa: E402

PORT = 5055
JOINTS = robot.JOINT_NAMES
N = len(JOINTS)
RATED = np.where(robot.EFFORT_LIMIT >= 100, 40.0, 20.0)
HANG_ABOVE_FLOOR = 0.10   # twin hangs with the feet this far above the floor (gantry)

GREY = np.array([0.45, 0.45, 0.5, 1.0])
YELLOW = np.array([1.0, 0.85, 0.1, 1.0])
GHOST = np.array([0.3, 0.6, 1.0, 0.28])


def torque_color(frac):
  """0 -> green, 1 (rated) -> orange, >= 1.5 -> red."""
  green, orange, red = np.array([0.2, 0.75, 0.3]), np.array([1.0, 0.6, 0.1]), np.array([0.9, 0.1, 0.1])
  if frac <= 1.0:
    c = green + (orange - green) * frac
  else:
    c = orange + (red - orange) * min((frac - 1.0) / 0.5, 1.0)
  return np.r_[c, 1.0]


def quat_from_gravity(g):
  """Body orientation (yaw = 0) that makes gravity appear as g in the body frame."""
  g = np.asarray(g, float)
  if not np.all(np.isfinite(g)) or np.linalg.norm(g) < 0.5:
    return np.array([1.0, 0, 0, 0])
  up = -g / np.linalg.norm(g)                  # world z seen from the body
  x = np.array([1.0, 0, 0]) - up[0] * up       # world x = body x made level
  if np.linalg.norm(x) < 1e-6:
    x = np.array([0, 1.0, 0]) - up[1] * up
  x /= np.linalg.norm(x)
  y = np.cross(up, x)
  R = np.stack([x, y, up])                     # rows = world axes in body frame -> R_world_body
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, R.flatten())
  return quat


class Twin:
  """Keeps the MuJoCo model in step with the robot state and draws the extras."""

  def __init__(self):
    self.m = robot.make_model(visual=True)
    self.d = mujoco.MjData(self.m)
    self.ghost = mujoco.MjData(self.m)
    self.rgba0 = self.m.geom_rgba.copy()
    self.link_geoms = []                       # geoms moved by joint i (its child body)
    for i in range(N):
      body = self.m.jnt_bodyid[self.m.joint(JOINTS[i]).id]
      self.link_geoms.append([g for g in range(self.m.ngeom)
                              if self.m.geom_bodyid[g] == body and self.m.geom_group[g] < 3])
    self.state = None
    self.t_rx = 0.0
    # Base height that puts the feet on the floor in the home pose, plus a gap (gantry).
    self.hang = robot.init_base_height(self.m) + HANG_ABOVE_FLOOR
    for d in (self.d, self.ghost):         # start in the home pose, not at height 0
      d.qpos[:3] = [0, 0, self.hang]
      d.qpos[3:7] = [1, 0, 0, 0]
      d.qpos[7:] = robot.DEFAULT_JOINT_POS
      mujoco.mj_kinematics(self.m, d)

  def apply(self, st):
    self.state = st
    q = np.array([np.nan if v is None else v for v in st.get("q", [np.nan] * N)], float)
    quat = quat_from_gravity([np.nan if v is None else v for v in st.get("grav", [0, 0, -1])])
    for d, joints in ((self.d, q), (self.ghost, st.get("target"))):
      d.qpos[:3] = [0, 0, self.hang]
      d.qpos[3:7] = quat
      if joints is not None:
        jq = np.array([np.nan if v is None else v for v in joints], float)
        d.qpos[7:] = np.where(np.isfinite(jq), jq, 0.0)
      mujoco.mj_kinematics(self.m, d)

    tau = np.array([np.nan if v is None else v for v in st.get("tau", [None] * N)], float)
    enabled = st.get("enabled") or [False] * N
    self.m.geom_rgba[:] = self.rgba0
    for i in range(N):
      if i == st.get("hl", -1):
        col = YELLOW
      elif not enabled[i] or not np.isfinite(tau[i]):
        col = GREY
      else:
        col = torque_color(abs(tau[i]) / RATED[i])
      for g in self.link_geoms[i]:
        self.m.geom_rgba[g] = col

  def add_ghost(self, scn, opt, pert):
    if not self.state or self.state.get("target") is None:
      return
    start = scn.ngeom
    mujoco.mjv_addGeoms(self.m, self.ghost, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
    for k in range(start, scn.ngeom):
      scn.geoms[k].rgba[:] = GHOST

  def texts(self, extra=""):
    st = self.state
    if st is None:
      return (f"Waiting for data on UDP port {PORT} ...\n"
              "Start bringup.py with --twin <this computer's IP>"), ""
    left = [st.get("step", ""), st.get("msg", "")]
    if st.get("stop"):
      left.append("STOP: " + st["stop"])
    g = np.array([np.nan if v is None else v for v in st.get("grav", [0, 0, -1])], float)
    tilt = np.degrees(np.arccos(np.clip(-g[2], -1, 1))) if np.all(np.isfinite(g)) else float("nan")
    left.append(f"body tilt {tilt:.1f} deg")
    if extra:
      left.append(extra)
    names, vals = ["joint   angle / target   torque   temp"], [""]
    q, tgt = st.get("q", [None] * N), st.get("target") or [None] * N
    tau, temp = st.get("tau", [None] * N), st.get("temp") or [None] * N
    en = st.get("enabled") or [False] * N

    def f(v, fmt):
      return fmt.format(v) if v is not None and np.isfinite(v) else "  --"
    for i, j in enumerate(JOINTS):
      short = j.replace("right_", "R ").replace("left_", "L ").replace("_pitch", "").replace("_", " ")
      pct = None if tau[i] is None else 100 * abs(tau[i]) / RATED[i]
      names.append(("> " if i == st.get("hl", -1) else "  ") + short)
      vals.append(f"{f(q[i], '{:+.2f}')} / {f(tgt[i], '{:+.2f}')}  {f(pct, '{:4.0f}%')}  "
                  f"{f(temp[i], '{:3.0f}C')}{'' if en[i] else '  off'}")
    return "\n".join(l for l in left if l), ("\n".join(names), "\n".join(vals))


def replay_frames(path):
  d = np.genfromtxt(path, delimiter=",", names=True)
  cols = d.dtype.names
  frames = []
  for row in np.atleast_1d(d):
    def pick(prefix):
      return [float(row[f"{prefix}_{j}"]) if f"{prefix}_{j}" in cols else None for j in JOINTS]
    tau = pick("tau")
    frames.append(dict(
      t=float(row["t"]), q=pick("q"), target=pick("target"), tau=tau,
      enabled=[v is not None and np.isfinite(v) for v in tau],
      grav=[float(row[f"grav_{a}"]) for a in "xyz"] if "grav_x" in cols else [0, 0, -1],
      step=f"REPLAY {Path(path).name}", msg="", hl=-1, stop=""))
  return frames


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--port", type=int, default=PORT)
  p.add_argument("--replay", default=None, help="a CSV saved by bringup.py")
  args = p.parse_args()
  import mujoco.viewer

  twin = Twin()
  frames, sock = None, None
  if args.replay:
    frames = replay_frames(args.replay)
    print(f"Replaying {len(frames)} frames ({frames[-1]['t']:.1f} s) from {args.replay}")
  else:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.setblocking(False)
    print(f"Listening on UDP port {args.port}.  Start bringup.py with --twin <this computer's IP>")

  ctl = dict(paused=False, restart=False, speed=1.0)

  def on_key(key):
    if key == 32:
      ctl["paused"] = not ctl["paused"]
    elif key in (ord("R"), ord("r")):
      ctl["restart"] = True
    elif key == ord("["):
      ctl["speed"] = max(0.125, ctl["speed"] / 2)
    elif key == ord("]"):
      ctl["speed"] = min(8.0, ctl["speed"] * 2)

  mujoco.mj_forward(twin.m, twin.d)
  with mujoco.viewer.launch_passive(twin.m, twin.d, key_callback=on_key,
                                    show_left_ui=False, show_right_ui=False) as v:
    v.cam.lookat[:] = [0, 0, twin.hang - 0.35]
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.2, 135, -15
    k, t_play, last = 0, 0.0, time.perf_counter()
    while v.is_running():
      now = time.perf_counter()
      extra = ""
      if frames:
        if ctl["restart"]:
          k, t_play, ctl["restart"] = 0, 0.0, False
        if not ctl["paused"]:
          t_play += (now - last) * ctl["speed"]
        while k < len(frames) - 1 and frames[k + 1]["t"] <= t_play:
          k += 1
        if k >= len(frames) - 1:
          k, t_play = 0, 0.0
        twin.apply(frames[k])
        extra = (f"t = {frames[k]['t']:.2f} s   speed x{ctl['speed']:g}"
                 f"{'   PAUSED' if ctl['paused'] else ''}   (space, R, [ ])")
      else:
        newest = None
        while True:
          try:
            newest = sock.recv(65536)
          except (BlockingIOError, OSError):
            break
        if newest:
          twin.apply(json.loads(newest))
          twin.t_rx = now
        if twin.state and now - twin.t_rx > 1.0:
          extra = f"NO DATA for {now - twin.t_rx:.0f} s"
      last = now
      with v.lock():
        v.user_scn.ngeom = 0
        twin.add_ghost(v.user_scn, v.opt, v.perturb)
        left, right = twin.texts(extra)
        texts = [(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, "")]
        if right:
          texts.append((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                        right[0], right[1]))
        if hasattr(v, "set_texts"):
          v.set_texts(texts)
      v.sync()
      time.sleep(1 / 60)


if __name__ == "__main__":
  main()
