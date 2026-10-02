#!/usr/bin/env python3
"""bringup.py - step-by-step hardware bring-up for the OC1 biped.

Follows the Sim-to-Real SOP.  For every step it:
  1. tells you what it will do and what you must do first,
  2. waits for you to type  y,
  3. runs the step and measures the result,
  4. checks the numbers against a pass limit (PASS / FAIL),
  5. asks you to confirm before moving to the next step.

Run it on the Raspberry Pi (motors on the CAN hub):
    python3 scripts/hardware/bringup.py
    python3 scripts/hardware/bringup.py --list          show all steps
    python3 scripts/hardware/bringup.py --start 9       continue from step 9
    python3 scripts/hardware/bringup.py --only 6        run just step 6

Try it without hardware first (MuJoCo copy of the robot, answers itself):
    .venv/bin/python scripts/hardware/bringup.py --sim --yes

Watch it live in 3D (digital twin) on your laptop:
    .venv/bin/mjpython scripts/hardware/twin_viewer.py                   on the laptop
    python3 scripts/hardware/bringup.py --twin <laptop IP>               on the Pi
    .venv/bin/python scripts/hardware/bringup.py --sim --twin 127.0.0.1  or sim on the laptop

Stop at any time:  press Enter while motors are moving, or Ctrl+C.
Every exit path (error, Ctrl+C, end of script) disables all motors.

Before the first run, fill in scripts/hardware/hw_config.py (motor ids, IMU).
Results go to runs/hardware/<date>/ (summary.txt + one CSV per moving step).
Fixes found during bring-up (motor zeros, flipped signs) are saved to
scripts/hardware/hw_offsets.json - nothing is written to the motors' flash.
"""

import argparse
import atexit
import csv
import json
import math
import os
import select
import signal
import socket
import struct
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="All-NaN")
warnings.filterwarnings("ignore", message="Mean of empty")

HERE = Path(__file__).resolve().parent
ROOT = next((d for d in Path(__file__).resolve().parents if (d / "oc1_rl").is_dir()),
            Path(__file__).resolve().parents[2])   # project folder, wherever this file is
sys.path.insert(0, str(HERE))
import hw_config as C  # noqa: E402

OFFSETS_FILE = HERE / "hw_offsets.json"

# ------------------------------------------------------------------ robot model (from training)

JOINTS = tuple(C.MOTORS)
N = len(JOINTS)
DEFAULT_Q = np.array([-0.1, 0.0, 0.0, -0.3, 0.2, 0.1, 0.0, 0.0, 0.3, 0.2])
MODEL = [C.MOTORS[j]["model"] for j in JOINTS]
MAX_TQ = np.array([C.MOTOR_SPECS[m]["t_max"] for m in MODEL])
RATED_TQ = np.array([C.MOTOR_SPECS[m]["rated"] for m in MODEL])
_ARM = np.where(MAX_TQ >= 100, 0.04, 0.02)
SIM_KP = _ARM * (2 * math.pi * 10) ** 2
ACTION_SCALE = 0.25 * MAX_TQ / SIM_KP          # ~0.19 rad per unit action
STEP_DT = 0.02                                 # 50 Hz policy
GAIT_PERIOD = 0.6
NUM_OBS = 41

# What "+" means for each joint (worked out from the URDF), used in the hand test.
PLUS_DIRECTION = {
  "right_hip_pitch": "swing the right leg BACKWARD",
  "right_hip_roll": "move the right foot OUTWARD (to the robot's right)",
  "right_hip_yaw": "turn the right toe to the robot's LEFT (inward)",
  "right_knee_pitch": "STRAIGHTEN the right knee (foot comes forward)",
  "right_ankle_pitch": "lift the right TOE UP",
  "left_hip_pitch": "swing the left leg FORWARD",
  "left_hip_roll": "move the left foot INWARD (to the robot's right)",
  "left_hip_yaw": "turn the left toe to the robot's LEFT (outward)",
  "left_knee_pitch": "BEND the left knee (foot goes backward)",
  "left_ankle_pitch": "lift the left TOE UP",
}

# ------------------------------------------------------------------ RobStride protocol
# (same frames as your robstride-can-hub-with-rpi scripts)

T_PING, T_MIT, T_FEEDBACK, T_ENABLE, T_DISABLE = 0, 1, 2, 3, 4
T_READ, T_WRITE = 17, 18
RUN_MODE, LOC_REF, LIMIT_SPD, LIMIT_CUR = 0x7005, 0x7016, 0x7017, 0x7018
MECH_POS, MECH_VEL, CAN_TIMEOUT = 0x7019, 0x701B, 0x7028
MODE_NAMES = {0: "off", 1: "calibrating", 2: "running"}

DEG = 180.0 / math.pi
USE_COLOR = sys.stdout.isatty()


def color(text, c):
  codes = {"g": "32", "r": "31", "y": "33", "b": "1", "c": "36"}
  return f"\033[{codes[c]}m{text}\033[0m" if USE_COLOR else text


def ext_id(comm_type, extra, target):
  return (comm_type << 24) | (extra << 8) | target


def to_u16(x, x_max):
  return int(max(0, min(65535, (x + x_max) / (2 * x_max) * 65535)))


def from_u16(code, x_max):
  return (code / 32767.0 - 1.0) * x_max


# ------------------------------------------------------------------ CAN ports

class SocketCanPort:
  def __init__(self, channel):
    self.channel = channel
    self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    try:
      self.sock.bind((channel,))
    except OSError as exc:
      raise RuntimeError(f"cannot open {channel} ({exc}).  Bring it up with:\n"
                         f"  sudo ip link set {channel} up type can bitrate 1000000")
    self.sock.setblocking(False)
    self.send_errors = 0

  def send(self, can_id, data):
    frame = struct.pack("=IB3x8s", can_id | socket.CAN_EFF_FLAG, 8, bytes(data).ljust(8, b"\0"))
    try:
      self.sock.send(frame)
    except OSError:
      self.send_errors += 1          # ENOBUFS: nobody acknowledging on this bus

  def recv(self, timeout):
    if not select.select([self.sock], [], [], max(0.0, timeout))[0]:
      return None
    try:
      raw = self.sock.recv(16)
    except (BlockingIOError, OSError):
      return None
    can_id, _dlc, data = struct.unpack("=IB3x8s", raw)
    return can_id & 0x1FFFFFFF, data

  def close(self):
    self.sock.close()


class UsbCanPort:
  def __init__(self, channel):
    from usbcan import UsbCanBus      # copy usbcan.py next to this file
    self.channel = channel
    self.bus = UsbCanBus(C.USBCAN_PORT)
    self.send_errors = 0

  def send(self, can_id, data):
    self.bus.send(can_id, bytes(data).ljust(8, b"\0"))

  def recv(self, timeout):
    return self.bus.recv(timeout=max(0.0, timeout))

  def close(self):
    self.bus.close()


# ------------------------------------------------------------------ hardware backend (raw motor units)

class HardwareBackend:
  """Talks to the real motors.  Positions here are the motors' own readings."""
  is_sim = False

  def __init__(self):
    self.ids = [C.MOTORS[j]["id"] for j in JOINTS]
    self.chan = [C.MOTORS[j]["channel"] for j in JOINTS]
    self.ports = {}
    self.enabled = np.zeros(N, bool)
    self.imu_handle = None
    self.last_fb = None

  # ---- ports
  def open_ports(self):
    out = {}
    for ch in sorted(set(self.chan)):
      if ch in self.ports:
        out[ch] = "open"
        continue
      try:
        self.ports[ch] = (UsbCanPort if C.CAN_BACKEND == "usbcan" else SocketCanPort)(ch)
        out[ch] = "open"
      except Exception as exc:  # noqa: BLE001
        out[ch] = f"ERROR: {exc}"
    return out

  def port_state(self, ch):
    p = Path(f"/sys/class/net/{ch}/operstate")
    return p.read_text().strip() if p.exists() else "missing"

  def send(self, i, comm_type, data, extra=None):
    self.ports[self.chan[i]].send(ext_id(comm_type, C.HOST_ID if extra is None else extra,
                                         self.ids[i]), data)

  def drain(self):
    for p in self.ports.values():
      while p.recv(0.0) is not None:
        pass

  def _collect(self, want, match, timeout):
    """Read frames until match() has filled every index in `want` or timeout."""
    got = {}
    deadline = time.perf_counter() + timeout
    while len(got) < len(want) and time.perf_counter() < deadline:
      idle = True
      for ch, p in self.ports.items():
        fr = p.recv(0.0)
        if fr is None:
          continue
        idle = False
        r = match(ch, *fr)
        if r is not None and r[0] in want:
          got[r[0]] = r[1]
      if idle:
        time.sleep(0.0002)
    return got

  def _index_of(self, ch, motor_id):
    for i in range(N):
      if self.chan[i] == ch and self.ids[i] == motor_id:
        return i
    return None

  # ---- parameters
  def read_param(self, index, joints=None):
    joints = range(N) if joints is None else joints
    joints = [i for i in joints if self.chan[i] in self.ports]
    self.drain()
    for i in joints:
      self.send(i, T_READ, struct.pack("<HHI", index, 0, 0))

    def match(ch, can_id, data):
      if (can_id >> 24) != T_READ or (can_id & 0xFF) != C.HOST_ID:
        return None
      if struct.unpack_from("<H", data, 0)[0] != index:
        return None
      i = self._index_of(ch, (can_id >> 8) & 0xFF)
      return None if i is None else (i, struct.unpack_from("<f", data, 4)[0])
    got = self._collect(set(joints), match, 0.15)
    out = np.full(N, np.nan)
    for i, v in got.items():
      out[i] = v
    return out

  def write_float(self, i, index, value):
    self.send(i, T_WRITE, struct.pack("<HHf", index, 0, float(value)))

  def write_u8(self, i, index, value):
    self.send(i, T_WRITE, struct.pack("<HHB3x", index, 0, int(value)))

  def write_u32(self, i, index, value):
    self.send(i, T_WRITE, struct.pack("<HHI", index, 0, int(value)))

  # ---- discovery
  def ping(self):
    return ~np.isnan(self.read_param(MECH_POS))

  def scan(self):
    """Type-0 ping to ids 1..127 on every open channel (like discover.py)."""
    found = []
    for ch, p in self.ports.items():
      while p.recv(0.0) is not None:
        pass
      for target in range(1, 128):
        p.send(ext_id(T_PING, C.HOST_ID, target), b"\0" * 8)
        deadline = time.perf_counter() + 0.01
        while time.perf_counter() < deadline:
          fr = p.recv(deadline - time.perf_counter())
          if fr and (fr[0] >> 24) == T_PING:
            found.append((ch, (fr[0] >> 8) & 0xFF))
            break
    return sorted(set(found))

  # ---- state
  def read_raw(self):
    return self.read_param(MECH_POS), self.read_param(MECH_VEL)

  def enable(self, joints, current, pos):
    """Start holding `pos` (raw) on these joints.  Same order as jog_hold2.Motor.start()."""
    joints = list(joints)
    mit = C.CONTROL_MODE == "mit"
    self.drain()
    for i in joints:
      self.write_u8(i, RUN_MODE, 0 if mit else 5)
    time.sleep(0.05)
    if not mit:
      for i in joints:
        self.write_float(i, LOC_REF, pos[i])
      time.sleep(0.05)
    for i in joints:
      self.send(i, T_ENABLE, b"\0" * 8)
    time.sleep(0.05)
    for i in joints:
      if not mit:
        self.write_float(i, LIMIT_SPD, C.SPEED_LIMIT)
        self.write_float(i, LIMIT_CUR, current[MODEL[i]])
      if C.CAN_TIMEOUT_RAW:
        self.write_u32(i, CAN_TIMEOUT, C.CAN_TIMEOUT_RAW)
    time.sleep(0.03)
    self.enabled[joints] = True
    self.command(pos, kp_scale=0.3)
    self.drain()

  def command(self, pos, kp_scale=1.0):
    """Send targets to every enabled motor, return the feedback that comes back."""
    en = np.nonzero(self.enabled)[0]
    for i in en:
      if C.CONTROL_MODE == "mit":
        s = C.MOTOR_SPECS[MODEL[i]]
        kp, kd = C.KP[MODEL[i]] * kp_scale, C.KD[MODEL[i]]
        can_id = ext_id(T_MIT, to_u16(0.0, s["t_max"]), self.ids[i])
        data = struct.pack(">HHHH", to_u16(pos[i], s["p_max"]), to_u16(0.0, s["v_max"]),
                           int(min(kp, s["kp_max"]) / s["kp_max"] * 65535),
                           int(min(kd, s["kd_max"]) / s["kd_max"] * 65535))
        self.ports[self.chan[i]].send(can_id, data)
      else:
        self.write_float(i, LOC_REF, pos[i])
    return self.feedback(set(en.tolist()))

  def feedback(self, want, timeout=0.006):
    def match(ch, can_id, data):
      if (can_id >> 24) != T_FEEDBACK:
        return None
      i = self._index_of(ch, (can_id >> 8) & 0xFF)
      if i is None:
        return None
      s = C.MOTOR_SPECS[MODEL[i]]
      p, v, t, temp = struct.unpack(">HHHH", data)
      return i, (from_u16(p, s["p_max"]), from_u16(v, s["v_max"]), from_u16(t, s["t_max"]),
                 temp / 10.0, (can_id >> 16) & 0x3F, (can_id >> 22) & 0x03)
    got = self._collect(want, match, timeout)
    fb = {k: np.full(N, np.nan) for k in ("pos", "vel", "tq", "temp")}
    fb["faults"], fb["mode"] = np.zeros(N, int), np.full(N, -1)
    fb["got"] = np.zeros(N, bool)
    for i, (p, v, t, temp, f, m) in got.items():
      fb["pos"][i], fb["vel"][i], fb["tq"][i], fb["temp"][i] = p, v, t, temp
      fb["faults"][i], fb["mode"][i], fb["got"][i] = f, m, True
    return fb

  def disable(self, joints=None):
    joints = np.nonzero(self.enabled)[0] if joints is None else list(joints)
    for _ in range(5):
      for i in joints:
        if self.chan[i] in self.ports:
          self.send(i, T_DISABLE, b"\0" * 8)
      time.sleep(0.005)
    self.enabled[joints] = False

  def disable_everything(self):
    """Disable every configured motor, enabled or not (used on exit)."""
    for _ in range(5):
      for i in range(N):
        if self.chan[i] in self.ports and self.ids[i] is not None:
          self.send(i, T_DISABLE, b"\0" * 8)
      time.sleep(0.005)
    self.enabled[:] = False

  def silence(self, seconds):
    time.sleep(seconds)

  # ---- IMU
  def imu_raw(self):
    if C.IMU_TYPE == "none":
      return None
    if self.imu_handle is None:
      self.imu_handle = C.open_imu()
    gyro, quat = C.read_imu(self.imu_handle)
    return np.asarray(gyro, float), np.asarray(quat, float)

  def simulate(self, *_a, **_k):
    pass                                   # the real world does this part

  def tick(self):
    pass

  def close(self):
    for p in self.ports.values():
      p.close()


# ------------------------------------------------------------------ sim backend (for testing this script)

class SimBackend:
  """MuJoCo copy of the robot hanging in a gantry.  Pretends to be the CAN bus."""
  is_sim = True

  def __init__(self, flip=(), swap=None, bad_zero=0.0):
    import mujoco
    sys.path.insert(0, str(ROOT))
    from oc1_rl import robot
    self.mj, self.robot = mujoco, robot
    self.m = robot.make_model(visual=False)
    self.d = mujoco.MjData(self.m)
    self.gain = self.m.actuator_gainprm[:, 0].copy()
    self.bias = self.m.actuator_biasprm[:, :3].copy()
    self.enabled = np.zeros(N, bool)
    self.held = True                      # base held by the gantry
    self.base_quat = np.array([1.0, 0, 0, 0])
    self.base_pos = np.array([0, 0, 1.2])
    self.base_w = np.zeros(3)
    self.true_sign = np.ones(N)
    for j in flip:
      self.true_sign[JOINTS.index(j)] = -1
    self.true_zero = np.array([C.MOTORS[j]["zero"] for j in JOINTS]) + \
      np.random.default_rng(0).uniform(-bad_zero, bad_zero, N)
    self.order = np.arange(N)
    if swap:
      a, b = (JOINTS.index(j) for j in swap)
      self.order[a], self.order[b] = b, a
    self.actions = []                     # scripted "person" actions
    self.t = 0.0
    self.last_cmd = np.zeros(N)
    self.estopped = False
    self.target = np.zeros(N)
    self.d.qpos[3] = 1.0
    self._apply_gains()
    self._hold_base()
    mujoco.mj_forward(self.m, self.d)

  def _apply_gains(self):
    for i in range(N):
      on = self.enabled[i] and not self.estopped
      self.m.actuator_gainprm[i, 0] = self.gain[i] if on else 0.0
      self.m.actuator_biasprm[i, 1:3] = self.bias[i, 1:3] if on else 0.0
      self.m.dof_damping[6 + i] = 0.0 if on else 2.0     # a switched-off motor still drags a bit

  def _hold_base(self):
    if self.held:
      self.d.qpos[0:3], self.d.qpos[3:7] = self.base_pos, self.base_quat
      self.d.qvel[0:6] = 0.0
      self.d.qvel[3:6] = self.base_w

  def _q(self):
    return self.d.qpos[7:].copy()

  def _raw(self, q):                      # joint -> what the motor would report
    return (self.true_zero + self.true_sign * q)[self.order]

  def _from_raw(self, raw):
    back = np.empty(N)
    back[self.order] = raw
    return (back - self.true_zero) * self.true_sign

  def tick(self):
    """Advance 20 ms (4 physics steps), playing any scripted person action."""
    for _ in range(4):
      for act in self.actions:
        act(self.t)
      self.actions = [a for a in self.actions if not getattr(a, "done", False)]
      if not self.estopped:
        for i in np.nonzero(self.enabled)[0]:
          if self.t - self.last_cmd[i] > C.CAN_TIMEOUT_RAW / 20000.0:
            self.enabled[i] = False       # motor watchdog
            self._apply_gains()
      self.d.ctrl[:] = self._from_raw(self.target) if self.enabled.any() else self._q()
      self.mj.mj_step(self.m, self.d)
      self._hold_base()
      self.t += self.m.opt.timestep
    time.sleep(0.0)

  # ---- same API as HardwareBackend
  def open_ports(self):
    return {ch: "open (sim)" for ch in sorted({C.MOTORS[j]["channel"] for j in JOINTS})}

  def port_state(self, ch):
    return "up"

  def ping(self):
    return np.full(N, not self.estopped)

  def scan(self):
    return []

  def read_raw(self):
    self.tick()
    if self.estopped:
      return np.full(N, np.nan), np.full(N, np.nan)
    vel = self.d.qvel[6:][self.order] * self.true_sign[self.order]
    return self._raw(self._q()), vel

  def enable(self, joints, current, pos):
    joints = list(joints)
    self.target = np.array(pos, float)
    self.enabled[joints] = True
    self.last_cmd[joints] = self.t
    self._apply_gains()

  def command(self, pos, kp_scale=1.0):
    self.target = np.array(pos, float)
    self.last_cmd[self.enabled] = self.t
    self.tick()
    fb = {k: np.full(N, np.nan) for k in ("pos", "vel", "tq", "temp")}
    fb["faults"], fb["mode"], fb["got"] = np.zeros(N, int), np.full(N, -1), np.zeros(N, bool)
    if self.estopped:
      return fb
    act = self.d.actuator_force.copy()
    raw_pos = self._raw(self._q())
    raw_vel = (self.d.qvel[6:] * self.true_sign)[self.order]
    raw_tq = (act * self.true_sign)[self.order]
    for i in range(N):
      fb["mode"][i] = 2 if self.enabled[i] else 0
      if self.enabled[i]:
        fb["pos"][i], fb["vel"][i], fb["tq"][i], fb["temp"][i] = raw_pos[i], raw_vel[i], raw_tq[i], 35.0
        fb["got"][i] = True
    return fb

  def feedback(self, want, timeout=0.0):
    return self.command(self.target)

  def disable(self, joints=None):
    joints = np.nonzero(self.enabled)[0] if joints is None else list(joints)
    self.enabled[joints] = False
    self._apply_gains()

  def disable_everything(self):
    self.disable(range(N))

  def silence(self, seconds):
    for _ in range(int(seconds / 0.02)):
      self.tick()

  def imu_raw(self):
    return self.d.qvel[3:6].copy(), self.d.qpos[3:7].copy()

  def close(self):
    pass

  # ---- the scripted "person" (only used with --sim)
  def simulate(self, what, **kw):
    t0, dur = self.t, kw.get("duration", 1.5)

    def bump(t):                            # smooth 0 -> 1 -> 0
      x = min(max((t - t0) / dur, 0.0), 1.0)
      return math.sin(math.pi * x) ** 2
    if what == "hand_move":
      i, amount, done = JOINTS.index(kw["joint"]), kw.get("amount", 0.25), [False]
      start = self.d.qpos[7 + i]

      def act(t):
        self.d.qpos[7 + i] = start + amount * bump(t)
        self.d.qvel[6 + i] = 0.0
        act.done = t - t0 > dur
      self.actions.append(act)
    elif what == "pose":                    # person puts the legs in a pose
      self.d.qpos[7:] = kw["q"]
      self.d.qvel[6:] = 0.0
    elif what == "tilt":                    # person tilts the robot about a base axis
      axis, ang, hold = np.asarray(kw["axis"], float), kw["angle"], kw.get("hold", True)
      q0 = self.base_quat.copy()

      def act(t):
        x = min(max((t - t0) / dur, 0.0), 1.0)
        a = ang * (0.5 - 0.5 * math.cos(math.pi * x)) if hold else ang * bump(t)
        dq = np.r_[math.cos(a / 2), math.sin(a / 2) * axis]
        new = np.empty(4)
        self.mj.mju_mulQuat(new, q0, dq)
        if 0 < x < 1:
          rate = (ang * 0.5 * math.pi / dur * math.sin(math.pi * x)) if hold else \
            (ang * math.pi / dur * math.sin(2 * math.pi * x))
        else:
          rate = 0.0
        self.base_quat = new
        self.base_w = axis * rate
        act.done = t - t0 > dur + 0.3
        if act.done:
          self.base_w = np.zeros(3)
      self.actions.append(act)
    elif what == "level":
      self.base_quat = np.array([1.0, 0, 0, 0])
      self.base_w = np.zeros(3)
    elif what == "estop":
      self.estopped = True
      self._apply_gains()
    elif what == "estop_release":
      self.estopped = False
      self.enabled[:] = False
      self._apply_gains()
    elif what == "lower":                   # gantry lowered: feet on the floor
      self.held = False
      self.d.qpos[0:2] = 0.0
      self.d.qpos[2] = self.robot.init_base_height(self.m) + 0.005
      self.d.qpos[3:7] = [1, 0, 0, 0]
      self.d.qpos[7:] = self._from_raw(self.target)
      self.d.qvel[:] = 0.0
    elif what == "raise":
      self.held = True
      self.base_quat = np.array([1.0, 0, 0, 0])
      self._hold_base()


# ------------------------------------------------------------------ robot (joint units, safety)

class StopRun(Exception):
  pass


TWIN_PORT = 5055


class TwinSender:
  """Streams the robot's state to twin_viewer.py (UDP, JSON, up to 50 per second).

  Sending never blocks and never raises: if the laptop is not listening, nothing happens."""

  def __init__(self, dest):
    host, _, port = dest.partition(":")
    self.addr = (host, int(port or TWIN_PORT))
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.sock.setblocking(False)
    self.last = 0.0
    self.state = dict(q=np.full(N, np.nan), target=None, tau=np.full(N, np.nan),
                      temp=np.full(N, np.nan), enabled=np.zeros(N, bool),
                      grav=np.array([0.0, 0.0, -1.0]), gyro=np.zeros(3),
                      step="", msg="", hl=-1, stop="")

  def update(self, force=False, **kw):
    self.state.update(kw)
    now = time.perf_counter()
    if not force and now - self.last < 0.02:
      return
    self.last = now

    def clean(v):
      if isinstance(v, np.ndarray):
        return [None if (isinstance(x, float) and not math.isfinite(x)) else x for x in v.tolist()]
      return v
    msg = {k: clean(v) for k, v in self.state.items()}
    msg["joints"] = list(JOINTS)
    msg["rated"] = RATED_TQ.tolist()
    try:
      self.sock.sendto(json.dumps(msg).encode(), self.addr)
    except OSError:
      pass


class NoTwin:
  def update(self, force=False, **kw):
    pass


TWIN = NoTwin()
REALTIME = True


def pace():
  """Wait one control period (the sim otherwise runs as fast as it can)."""
  if REALTIME:
    time.sleep(STEP_DT)


class Robot:
  """Joint-frame view of a backend: q = sign * (motor - zero).  All motion goes through here."""

  def __init__(self, backend):
    self.b = backend
    self.sign = np.array([float(C.MOTORS[j]["sign"]) for j in JOINTS])
    self.zero = np.array([float(C.MOTORS[j]["zero"]) for j in JOINTS])
    if OFFSETS_FILE.exists() and not backend.is_sim:
      saved = json.loads(OFFSETS_FILE.read_text())
      for i, j in enumerate(JOINTS):
        if j in saved:
          self.zero[i] = saved[j].get("zero", self.zero[i])
          self.sign[i] = saved[j].get("sign", self.sign[i])
    self.saved = {}
    self.imu_R = np.asarray(C.IMU_TO_BASE, float)
    self.current = C.CURRENT_LIMIT_BENCH
    self.target = DEFAULT_Q.copy()
    self.missed = 0

  def save_fix(self, i, **kw):
    self.saved.setdefault(JOINTS[i], {}).update(kw)
    if self.b.is_sim:
      return
    data = json.loads(OFFSETS_FILE.read_text()) if OFFSETS_FILE.exists() else {}
    data.setdefault(JOINTS[i], {}).update(kw)
    OFFSETS_FILE.write_text(json.dumps(data, indent=2))

  def to_joint(self, raw):
    return self.sign * (raw - self.zero)

  def to_raw(self, q):
    return self.zero + self.sign * q

  def read(self):
    p, v = self.b.read_raw()
    q = self.to_joint(p)
    TWIN.update(q=q, tau=np.full(N, np.nan), enabled=self.b.enabled.copy())
    return q, self.sign * v

  def imu(self):
    r = self.b.imu_raw()
    if r is None:
      return None, None
    gyro, quat = r
    w, u = quat[0], quat[1:]
    v = np.array([0.0, 0.0, -1.0])
    g = v * (2 * w * w - 1) - 2 * w * np.cross(u, v) + 2 * u * np.dot(u, v)
    if not self.b.is_sim:
      gyro, g = self.imu_R @ gyro, self.imu_R @ g
    TWIN.update(grav=g, gyro=gyro)
    return gyro, g

  def enable(self, joints, current=None):
    self.current = current or self.current
    p, _ = self.b.read_raw()
    if np.isnan(p[list(joints)]).any():
      missing = [JOINTS[i] for i in joints if np.isnan(p[i])]
      raise StopRun(f"motor(s) did not answer before enable: {', '.join(missing)}")
    self.target = self.to_joint(p)
    self.b.enable(joints, self.current, p)
    self.missed = 0
    return self.target.copy()

  def command(self, q_target):
    q_target = np.clip(q_target, -C.JOINT_TARGET_LIMIT, C.JOINT_TARGET_LIMIT)
    self.target = q_target
    fb = self.b.command(self.to_raw(q_target))
    en = self.b.enabled
    st = dict(q=self.to_joint(fb["pos"]), qd=self.sign * fb["vel"], tau=self.sign * fb["tq"],
              temp=fb["temp"], faults=fb["faults"], mode=fb["mode"], got=fb["got"],
              target=q_target.copy())
    self.missed = self.missed + 1 if (en & ~fb["got"]).any() else 0
    TWIN.update(q=st["q"], target=q_target.copy(), tau=st["tau"], temp=st["temp"],
                enabled=en.copy())
    return st

  def disable(self, joints=None):
    self.b.disable(joints)

  @property
  def enabled(self):
    return self.b.enabled


# ------------------------------------------------------------------ keyboard

def key_pressed():
  """True if Enter was pressed (non-blocking)."""
  if ARGS.yes or not sys.stdin.isatty():
    return False
  if select.select([sys.stdin], [], [], 0)[0]:
    sys.stdin.readline()
    return True
  return False


def ask(prompt, choices="yq", auto=None):
  if ARGS.yes:
    ans = auto or choices[0]
    print(color(f"{prompt} [{'/'.join(choices)}] -> {ans} (auto)", "c"))
    return ans
  while True:
    try:
      ans = input(color(f"{prompt} [{'/'.join(choices)}] ", "c")).strip().lower()[:1]
    except EOFError:
      ans = "q"
    if ans in choices:
      return ans
    print(f"  please type one of: {', '.join(choices)}")


def wait_enter(msg):
  if ARGS.yes:
    print(color(f"{msg}  (auto)", "c"))
    return
  input(color(f"{msg}  [press Enter] ", "c"))


# ------------------------------------------------------------------ control loop

class Rec:
  """What one moving test recorded (50 Hz)."""

  def __init__(self):
    self.rows = []
    self.stop = None

  def add(self, **kw):
    self.rows.append(kw)

  def arr(self, key):
    return np.array([r[key] for r in self.rows]) if self.rows else np.zeros((0, N))

  def __len__(self):
    return len(self.rows)


def run_loop(w, duration, target_fn, policy_mode=False, check_tilt=True, allow_key=True,
             tilt_soft=None, enter_ends=False):
  """50 Hz loop: target_fn(k, t, state) -> joint targets.  Safety checks every cycle.

  Stops early (sets rec.stop) on: Enter key, motor fault, overheating, lost feedback,
  joint past its limit, too much tilt.  A tilt past TILT_CUT_DEG disables all motors."""
  r = w.robot
  rec = Rec()
  steps = int(round(duration / STEP_DT))
  state = None
  t_next = time.perf_counter()
  t_start = t_next
  last = t_start
  tilt_soft = C.TILT_SOFT_DEG if tilt_soft is None else tilt_soft
  for k in range(steps):
    t = k * STEP_DT
    target = target_fn(k, t, state)
    t0 = time.perf_counter()
    state = r.command(target)
    rt = time.perf_counter() - t0
    gyro, grav = r.imu()
    tilt = float(np.degrees(np.arccos(np.clip(-grav[2], -1, 1)))) if grav is not None else 0.0
    now = time.perf_counter()
    rec.add(t=t, q=state["q"], target=state["target"], tau=state["tau"], qd=state["qd"],
            temp=state["temp"], got=state["got"], period=now - last, roundtrip=rt,
            gyro=gyro if gyro is not None else np.zeros(3),
            grav=grav if grav is not None else np.zeros(3), tilt=tilt,
            action=getattr(w, "last_action", np.zeros(N)).copy(),
            infer=getattr(w, "last_infer", 0.0))
    last = now
    en = r.enabled
    reason = None
    if allow_key and key_pressed():
      reason = "you pressed Enter"
    elif (state["faults"][en] != 0).any():
      bad = [f"{JOINTS[i]} 0x{state['faults'][i]:02X}" for i in np.nonzero(en & (state["faults"] != 0))[0]]
      reason = "motor fault: " + ", ".join(bad)
    elif np.nanmax(np.where(en, state["temp"], 0)) > C.MAX_TEMP_C:
      reason = f"motor too hot (> {C.MAX_TEMP_C} C)"
    elif r.missed >= C.MISSED_FRAMES_STOP:
      lost = [JOINTS[i] for i in np.nonzero(en & ~state["got"])[0]]
      reason = "no feedback for 3 steps from: " + ", ".join(lost)
    elif np.nanmax(np.abs(np.where(en, state["q"], 0))) > C.JOINT_LIMIT + 0.1:
      i = int(np.nanargmax(np.abs(np.where(en, state["q"], 0))))
      reason = f"{JOINTS[i]} past its limit ({state['q'][i]:+.2f} rad)"
    elif check_tilt and grav is not None and tilt > C.TILT_CUT_DEG:
      r.disable()
      reason = f"TILT {tilt:.0f} deg > {C.TILT_CUT_DEG:.0f} - motors disabled"
    elif check_tilt and grav is not None and policy_mode and tilt > tilt_soft:
      reason = f"tilt {tilt:.0f} deg > {tilt_soft:.0f} - policy stopped"
    if reason == "you pressed Enter" and enter_ends:
      break
    if reason:
      rec.stop = reason
      TWIN.update(force=True, stop=reason)
      print(color(f"  STOP: {reason}", "r"))
      break
    t_next += STEP_DT
    sleep = t_next - time.perf_counter()
    if sleep > 0:
      time.sleep(sleep)
    else:
      t_next = time.perf_counter()
  rec.state = state
  return rec


def smooth(x):
  x = min(max(x, 0.0), 1.0)
  return x * x * (3 - 2 * x)


def ramp(w, q_from, q_to, seconds, **kw):
  q_from, q_to = np.asarray(q_from, float), np.asarray(q_to, float)
  return run_loop(w, seconds, lambda k, t, s: q_from + (q_to - q_from) * smooth(t / seconds), **kw)


def hold(w, q, seconds, **kw):
  return run_loop(w, seconds, lambda k, t, s: q, **kw)


def hold_until_enter(w, q, msg, sim_action=None):
  """Keep holding pose q (sending commands) until the person presses Enter."""
  print(color(f"  {msg}  - motors are HOLDING. Press Enter when done.", "c"))
  if ARGS.yes:
    if sim_action:
      w.robot.b.simulate(sim_action)
    return hold(w, q, 0.1, allow_key=False)
  return run_loop(w, 600, lambda k, t, s: q, allow_key=True, enter_ends=True)


def park(w, q_start):
  """Ramp back to where the legs were when enabled, then disable."""
  r = w.robot
  if r.enabled.any():
    ramp(w, r.target, q_start, 2.0, check_tilt=False, allow_key=False)
    r.disable()


def save_rec(w, name, rec):
  if not rec.rows:
    return
  path = w.run_dir / f"{name}.csv"
  keys = ["t", "q", "target", "tau", "qd", "gyro", "grav", "tilt", "action", "period", "roundtrip"]
  with open(path, "w", newline="") as f:
    cw = csv.writer(f)
    head = []
    for k in keys:
      v = rec.rows[0][k]
      if np.ndim(v) == 0:
        head.append(k)
      elif len(v) == N:
        head += [f"{k}_{j}" for j in JOINTS]
      else:
        head += [f"{k}_{a}" for a in "xyz"]
    cw.writerow(head)
    for r in rec.rows:
      row = []
      for k in keys:
        v = r[k]
        row += [f"{float(v):.5f}"] if np.ndim(v) == 0 else [f"{float(x):.5f}" for x in v]
      cw.writerow(row)
  w.files.append(path.name)


# ------------------------------------------------------------------ policy

class Policy:
  def __init__(self, path):
    import onnxruntime as ort
    self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    self.inp = self.sess.get_inputs()[0]
    self.out = self.sess.get_outputs()[0]

  def __call__(self, obs):
    return self.sess.run(None, {self.inp.name: obs[None].astype(np.float32)})[0][0]


def build_obs(gyro, grav, cmd, k, q, qd, last_action):
  """Exactly the 41 inputs of OC1VelocityEnv.observations() (actor, no noise)."""
  cmd = np.asarray(cmd, float)
  if np.linalg.norm(cmd) < 0.1:
    phase = np.zeros(2)
  else:
    ph = (k * STEP_DT) % GAIT_PERIOD / GAIT_PERIOD
    phase = np.array([math.sin(2 * math.pi * ph), math.cos(2 * math.pi * ph)])
  return np.concatenate([gyro, grav, cmd, phase, q - DEFAULT_Q, qd, last_action]).astype(np.float32)


def policy_fn(w, cmd_fn, send=True):
  """Target function that runs the policy.  send=False: compute actions but hold home."""
  w.last_action = np.zeros(N)

  def fn(k, t, state):
    if state is None:
      return DEFAULT_Q.copy()
    gyro, grav = w.robot.imu()
    obs = build_obs(gyro, grav, cmd_fn(t), k, state["q"], state["qd"], w.last_action)
    t0 = time.perf_counter()
    a = np.asarray(w.policy(obs), float)
    w.last_infer = time.perf_counter() - t0
    w.last_action = a
    w.actions_seen.append(a)
    return DEFAULT_Q + a * ACTION_SCALE if send else DEFAULT_Q.copy()
  return fn


# ------------------------------------------------------------------ result helpers

class Result:
  def __init__(self):
    self.lines, self.ok = [], True

  def check(self, label, value, ok, limit=""):
    mark = color("PASS", "g") if ok else color("FAIL", "r")
    self.lines.append(f"  [{mark}] {label}: {value}" + (f"   (limit {limit})" if limit else ""))
    self.ok &= bool(ok)
    return ok

  def info(self, text):
    self.lines.append(f"         {text}")

  def fail(self, text):
    self.lines.append(f"  [{color('FAIL', 'r')}] {text}")
    self.ok = False


def joint_table(res, title, values, fmt="{:+.3f}", bad=None):
  res.info(title)
  for i, j in enumerate(JOINTS):
    flag = color("  <--", "r") if bad is not None and bad[i] else ""
    res.info(f"   {j:18s} {fmt.format(values[i]) if np.isfinite(values[i]) else '  --'}{flag}")


def need_ids(res):
  missing = [j for j in JOINTS if C.MOTORS[j]["id"] is None]
  if missing and not ARGS.sim:
    res.fail("motor id not set in hw_config.py for: " + ", ".join(missing))
    return False
  return True


def need_imu(w, res):
  if C.IMU_TYPE == "none" and not ARGS.sim:
    res.fail("IMU_TYPE is 'none' in hw_config.py - set up your IMU first")
    return False
  return True


def need_policy(w, res):
  if w.policy is None:
    res.fail("policy not loaded (step 2 must pass)")
    return False
  return True


# ================================================================== the steps
# Each step: (title, what it is for, what you must do first, function)

def s_checklist(w):
  res = Result()
  items = [
    "Robot is hanging in the gantry, both feet OFF the ground, legs free to swing",
    "Emergency stop (cuts motor power) is within reach and you have tested it once",
    "Power supply current limit set (bench supply) or battery charged and fused",
    "Nobody's hands or cables near the legs",
    "You know: press Enter (or Ctrl+C) in this window stops any movement",
  ]
  for it in items:
    a = ask(f"  {it}?", "yn")
    res.check(it, "yes" if a == "y" else "NO", a == "y")
  return res


def s_config(w):
  res = Result()
  res.check("joints in config", N, N == 10, "10")
  res.check("joint order", "same as training" if JOINTS == (
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee_pitch", "right_ankle_pitch",
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee_pitch", "left_ankle_pitch")
    else "DIFFERENT", JOINTS[0] == "right_hip_pitch" and JOINTS[5] == "left_hip_pitch")
  if need_ids(res):
    keys = [(C.MOTORS[j]["channel"], C.MOTORS[j]["id"]) for j in JOINTS]
    res.check("channel + id unique", "yes" if len(set(keys)) == N else "DUPLICATES",
              len(set(keys)) == N or ARGS.sim)
  res.check("control mode", C.CONTROL_MODE, C.CONTROL_MODE in ("csp", "mit"), "csp or mit")
  if C.CONTROL_MODE == "mit":
    res.info("MIT mode has not been tested on your rig yet - watch the first moves closely")
  res.info(f"IMU: {C.IMU_TYPE if not ARGS.sim else 'sim'}   offsets file: "
           f"{'found' if OFFSETS_FILE.exists() else 'none yet'}")
  path = Path(ARGS.policy or C.POLICY)
  path = path if path.is_absolute() else ROOT / path
  try:
    w.policy = Policy(path)
  except Exception as exc:  # noqa: BLE001
    res.fail(f"cannot load policy {path}: {exc}")
    return res
  shape_in, shape_out = w.policy.inp.shape[-1], w.policy.out.shape[-1]
  res.check("policy inputs", shape_in, shape_in == NUM_OBS, str(NUM_OBS))
  res.check("policy outputs", shape_out, shape_out == N, str(N))
  a = w.policy(build_obs(np.zeros(3), np.array([0, 0, -1.0]), np.zeros(3), 0, DEFAULT_Q,
                         np.zeros(N), np.zeros(N)))
  res.check("policy output at home pose", f"max |action| {np.abs(a).max():.2f}",
            np.all(np.isfinite(a)) and np.abs(a).max() < 3, "< 3")
  t0 = time.perf_counter()
  for _ in range(50):
    w.policy(np.zeros(NUM_OBS, np.float32))
  ms = (time.perf_counter() - t0) / 50 * 1000
  res.check("policy speed", f"{ms:.2f} ms per step", ms < 5, "< 5 ms")
  return res


def s_ports(w):
  res = Result()
  if not need_ids(res):
    return res
  if C.CAN_BACKEND == "socketcan":
    for ch in sorted({C.MOTORS[j]["channel"] for j in JOINTS}):
      st = w.robot.b.port_state(ch)
      res.check(f"{ch} link", st, st in ("up", "unknown"), "up")
  for ch, st in w.robot.b.open_ports().items():
    res.check(f"{ch} opened", st, st.startswith("open"))
  return res


def s_find(w):
  res = Result()
  if not need_ids(res):
    return res
  ok = w.robot.b.ping()
  for i, j in enumerate(JOINTS):
    c = C.MOTORS[j]
    res.check(f"{j:18s} {c['channel']} id {c['id'] if c['id'] is None else hex(c['id'])}",
              "answers" if ok[i] else "NO REPLY", ok[i])
  if not ok.all() and not w.robot.b.is_sim:
    res.info("scanning ids 1..127 on each channel to see what IS there ...")
    known = {(C.MOTORS[j]["channel"], C.MOTORS[j]["id"]) for j in JOINTS}
    for ch, mid in w.robot.b.scan():
      res.info(f"   found {ch} id 0x{mid:02X}" + ("" if (ch, mid) in known else "   <- not in hw_config"))
    res.info("fix: put the ids that were found into hw_config.py MOTORS")
  return res


def s_positions(w):
  res = Result()
  if not need_ids(res):
    return res
  q, qd = w.robot.read()
  far = np.abs(q) > C.JOINT_LIMIT + 0.3
  joint_table(res, "joint angles now (rad), motors off:", q, bad=far | ~np.isfinite(q))
  res.check("all readings present", int(np.isfinite(q).sum()), np.isfinite(q).all(), "10")
  res.check("all angles near the joint range", f"{int(far.sum())} far away", not far.any(),
            f"|q| < {C.JOINT_LIMIT + 0.3:.1f} rad")
  if far.any():
    res.info("a joint far outside +/-0.8 rad usually means that motor's zero was never set")
    res.info("-> step 'Zero check' can measure and save it")
  return res


def s_order_sign(w):
  res = Result()
  if not need_ids(res):
    return res
  r = w.robot
  print("  Motors are OFF.  For each joint you move ONLY that joint by hand, about 15-20 deg,")
  print("  in the direction shown, and back.  The script checks the right motor moved the right way.")
  for i, j in enumerate(JOINTS):
    while True:
      a = ask(f"\n  {color(j, 'b')}: {PLUS_DIRECTION[j]}, then back.  Ready to record 4 s?", "ysq")
      if a == "q":
        raise KeyboardInterrupt
      if a == "s":
        res.fail(f"{j}: skipped")
        break
      q_now, _ = r.read()
      show = np.nan_to_num(q_now)
      show[i] += 0.3
      TWIN.update(force=True, hl=i, target=show, msg=f"MOVE {j}: {PLUS_DIRECTION[j]}")
      if r.b.is_sim:
        r.b.simulate("hand_move", joint=j, amount=0.3, duration=2.0)
      print("  recording ... move it now")
      qs = []
      for _ in range(200):
        q, _ = r.read()
        qs.append(q)
        pace()
      qs = np.array(qs)
      dev = qs - qs[0]
      peak = np.nanmax(np.abs(dev), axis=0)
      moved = int(np.nanargmax(peak))
      signed = dev[np.nanargmax(np.abs(dev[:, moved])), moved]
      others = np.delete(peak, moved).max()
      print(f"  biggest movement: {JOINTS[moved]} {signed:+.2f} rad   (next biggest {others:.2f})")
      if peak[moved] < 0.08:
        print(color("  too small - move it further (15-20 deg)", "y"))
        continue
      if moved != i:
        print(color(f"  WRONG JOINT: you moved {j} but the script saw {JOINTS[moved]}.", "r"))
        print(f"  -> in hw_config.py, the ids of {j} and {JOINTS[moved]} are probably swapped")
        if ask("  try again after fixing? (r = record again, s = skip)", "rs", auto="s") == "s":
          res.fail(f"{j}: script saw {JOINTS[moved]} move")
          break
        continue
      if signed < 0:
        print(color(f"  DIRECTION REVERSED for {j}", "y"))
        if ask("  flip the sign of this joint (saved to hw_offsets.json)?", "yn") == "y":
          r.sign[i] *= -1
          r.save_fix(i, sign=float(r.sign[i]))
          print("  flipped - record again to confirm")
          continue
        res.fail(f"{j}: direction reversed")
        break
      res.check(f"{j}", f"correct joint, correct direction ({signed:+.2f} rad)", True)
      break
  TWIN.update(force=True, hl=-1, target=None, msg="")
  return res


ZERO_TOL = 0.015


def s_zero(w):
  res = Result()
  if not need_ids(res):
    return res
  r = w.robot
  print("  Put BOTH legs in the zero pose: legs straight down, knees straight, feet flat and")
  print("  parallel, toes forward (use your zeroing jig if you have one).")
  TWIN.update(force=True, target=np.zeros(N), msg="Put the legs in the zero pose (ghost)")
  wait_enter("  Holding them there?")
  if r.b.is_sim:
    r.b.simulate("pose", q=np.zeros(N))
  qs = []
  for _ in range(25):
    qs.append(r.read()[0])
    pace()
  qs = np.array(qs)
  q = np.nanmean(qs, 0)
  bad = np.abs(q) > ZERO_TOL
  joint_table(res, "angle in zero pose (rad):", q, bad=bad)
  res.check("all joints within zero tolerance", f"worst {np.nanmax(np.abs(q)):.3f} rad",
            not bad.any(), f"< {ZERO_TOL} rad")
  if bad.any():
    if ask("  save the measured zeros for the joints marked <-- (hw_offsets.json)?", "yn") == "y":
      raw_now = r.to_raw(q)
      for i in np.nonzero(bad)[0]:
        r.zero[i] = raw_now[i]
        r.save_fix(i, zero=float(raw_now[i]))
      q2 = np.nanmean(np.array([r.read()[0] for _ in range(25)]), 0)
      res.ok = True
      res.lines = [l for l in res.lines if "FAIL" not in l]
      res.check("after saving new zeros", f"worst {np.nanmax(np.abs(q2)):.3f} rad",
                np.nanmax(np.abs(q2)) < ZERO_TOL, f"< {ZERO_TOL} rad")
  return res


def record_imu(w, seconds, sim=None):
  r = w.robot
  if r.b.is_sim and sim:
    r.b.simulate(*sim[:1], **sim[1])
  g, gy = [], []
  for _ in range(int(seconds / STEP_DT)):
    if r.b.is_sim:
      r.b.tick()
    pace()
    gyro, grav = r.imu()
    g.append(grav)
    gy.append(gyro)
  return np.array(gy), np.array(g)


def s_imu(w):
  res = Result()
  if not need_imu(w, res):
    return res
  wait_enter("  Keep the robot still and level (hanging straight).")
  gy, g = record_imu(w, 2.0)
  bias = np.abs(gy.mean(0))
  res.check("gravity when level", np.round(g.mean(0), 3).tolist(),
            abs(g.mean(0)[0]) < 0.05 and abs(g.mean(0)[1]) < 0.05 and abs(g.mean(0)[2] + 1) < 0.05,
            "about [0, 0, -1]")
  res.check("gyro when still", f"{bias.max():.3f} rad/s", bias.max() < 0.03, "< 0.03 rad/s")
  tests = [
    ("tilt the robot NOSE DOWN about 20 deg and hold it there", ([0, 1, 0], 0.35),
     lambda gy, g: (g[-10:, 0].mean(), gy[:, 1][np.argmax(np.abs(gy[:, 1]))]),
     "gravity x goes positive, gyro y positive"),
    ("tilt the robot so its LEFT side goes UP about 20 deg and hold", ([1, 0, 0], 0.35),
     lambda gy, g: (-g[-10:, 1].mean(), gy[:, 0][np.argmax(np.abs(gy[:, 0]))]),
     "gravity y goes negative, gyro x positive"),
    ("turn the robot to its LEFT (counter-clockwise seen from above) about 30 deg",
     ([0, 0, 1], 0.5), lambda gy, g: (0.3, gy[:, 2][np.argmax(np.abs(gy[:, 2]))]),
     "gyro z positive"),
  ]
  for text, (axis, ang), measure, expect in tests:
    wait_enter(f"  Level again, then {text} - start moving after pressing Enter.")
    gy, g = record_imu(w, 3.0, sim=("tilt", dict(axis=axis, angle=ang, duration=1.5)))
    grav_val, gyro_val = measure(gy, g)
    res.check(text.split(" about")[0], f"gravity {grav_val:+.2f}, gyro peak {gyro_val:+.2f}",
              grav_val > 0.2 and gyro_val > 0.2, expect)
    if w.robot.b.is_sim:
      w.robot.b.simulate("level")
  if not res.ok:
    res.info("fix: change IMU_TO_BASE in hw_config.py (swap/negate rows) and repeat")
  return res


def s_enable_hold(w):
  res = Result()
  if not need_ids(res):
    return res
  r = w.robot
  for i, j in enumerate(JOINTS):
    if not ARGS.yes:
      wait_enter(f"  Next: enable {j} and hold for 2 s.")
    q0 = r.enable([i], C.CURRENT_LIMIT_BENCH)
    rec = hold(w, q0, 2.5, check_tilt=False)
    r.disable([i])
    got = rec.arr("got")[:, i].mean() if len(rec) else 0
    settled = rec.arr("q")[25:, i] if len(rec) > 25 else np.array([np.nan])
    drift = np.nanmax(np.abs(settled - q0[i]))           # after 0.5 s to settle
    temp = np.nanmax(rec.arr("temp")[:, i]) if len(rec) else np.nan
    ok = rec.stop is None and got > 0.9 and drift < 0.03
    res.check(f"{j:18s}", f"feedback {got * 100:3.0f}%  drift {drift:.3f} rad  temp {temp:.0f} C"
              + (f"  STOP: {rec.stop}" if rec.stop else ""), ok, "feedback >90%, drift <0.03")
  return res


def s_small_motion(w):
  res = Result()
  if not need_ids(res):
    return res
  r = w.robot
  amp, freq, dur = 0.1, 0.5, 4.0
  for i, j in enumerate(JOINTS):
    if not ARGS.yes:
      a = ask(f"  Next: {j} moves +/-{amp * DEG:.0f} deg slowly for {dur:.0f} s.  Go?", "ysq")
      if a == "q":
        raise KeyboardInterrupt
      if a == "s":
        res.fail(f"{j}: skipped")
        continue
    q0 = r.enable([i], C.CURRENT_LIMIT_BENCH)
    centre = np.clip(q0[i], -C.JOINT_TARGET_LIMIT + amp, C.JOINT_TARGET_LIMIT - amp)
    start = q0.copy()
    mid = q0.copy()
    mid[i] = centre
    ramp(w, start, mid, 1.0, check_tilt=False)

    def fn(k, t, s, i=i, mid=mid):
      q = mid.copy()
      q[i] += amp * math.sin(2 * math.pi * freq * t)
      return q
    rec = run_loop(w, dur, fn, check_tilt=False)
    park(w, start)
    if len(rec) < 10:
      res.fail(f"{j}: stopped early ({rec.stop})")
      continue
    tgt, q = rec.arr("target")[:, i], rec.arr("q")[:, i]
    err = np.sqrt(np.nanmean((q - tgt) ** 2))
    good = np.isfinite(q)
    corr = np.corrcoef(tgt[good], q[good])[0, 1] if good.sum() > 10 else np.nan
    lags = [np.nanmean((q[s:] - tgt[:len(tgt) - s]) ** 2) for s in range(0, 15)]
    lag_ms = int(np.argmin(lags)) * STEP_DT * 1000
    peak = np.nanmax(np.abs(rec.arr("tau")[:, i]))
    ok = rec.stop is None and err < 0.04 and corr > 0.9
    res.check(f"{j:18s}", f"error {err:.3f} rad  follows {corr:+.2f}  lag {lag_ms:3.0f} ms  "
              f"peak torque {peak:4.1f} N.m", ok, "error <0.04, follows >0.9")
    save_rec(w, f"small_motion_{j}", rec)
  return res


def s_home(w):
  res = Result()
  if not need_ids(res):
    return res
  r = w.robot
  q0 = r.enable(range(N), C.CURRENT_LIMIT_BENCH)
  rec = ramp(w, q0, DEFAULT_Q, 3.0, check_tilt=False)
  if rec.stop is None:
    rec2 = hold(w, DEFAULT_Q, 2.0, check_tilt=False)
    rec.rows += rec2.rows
    rec.stop = rec2.stop
  err = np.abs(rec.state["q"] - DEFAULT_Q) if rec.state else np.full(N, np.nan)
  peak = np.nanmax(np.abs(rec.arr("tau")), 0)
  joint_table(res, "error at home pose (rad):", err, fmt="{:.3f}", bad=err > 0.03)
  res.check("reached home pose", f"worst error {np.nanmax(err):.3f} rad",
            rec.stop is None and np.nanmax(err) < 0.03, "< 0.03 rad")
  res.check("torque while holding", f"peak {np.nanmax(peak / RATED_TQ) * 100:.0f}% of rated",
            np.all(peak < RATED_TQ), "< rated")
  save_rec(w, "home_pose", rec)
  if r.enabled.any():
    hold_until_enter(w, DEFAULT_Q, "Look at the robot: slight knee bend, feet flat, like the sim")
  park(w, q0)
  a = ask("  Did it look like the sim home pose?", "yn")
  res.check("pose looks right to you", "yes" if a == "y" else "no", a == "y")
  return res


def s_timing(w):
  res = Result()
  if not need_ids(res):
    return res
  r = w.robot
  q0 = r.enable(range(N), C.CURRENT_LIMIT_BENCH)
  ramp(w, q0, DEFAULT_Q, 3.0, check_tilt=False)
  rec = hold(w, DEFAULT_Q, 10.0, check_tilt=False)
  park(w, q0)
  per = rec.arr("period")[1:] * 1000
  rt = rec.arr("roundtrip") * 1000
  got = rec.arr("got")
  res.check("loop period", f"mean {per.mean():.2f} ms", abs(per.mean() - 20) < 0.5, "20 +/- 0.5 ms")
  res.check("loop jitter", f"99% under {np.percentile(per, 99):.1f} ms, worst {per.max():.1f}",
            np.percentile(per, 99) < 23, "99% < 23 ms")
  res.check("bus round trip", f"99% under {np.percentile(rt, 99):.1f} ms",
            np.percentile(rt, 99) < 6, "< 6 ms")
  lost = 100 * (1 - got.mean())
  res.check("lost feedback frames", f"{lost:.2f}%", lost < 0.5, "< 0.5%")
  save_rec(w, "timing", rec)
  return res


def s_safety(w):
  res = Result()
  if not need_ids(res):
    return res
  r = w.robot
  # 1. motor watchdog
  i = JOINTS.index("left_hip_yaw")
  timeout_s = C.CAN_TIMEOUT_RAW / 20000.0
  print(f"  1) Watchdog: {JOINTS[i]} holds, then the script goes silent for {timeout_s * 1.5:.1f} s.")
  q0 = r.enable([i], C.CURRENT_LIMIT_BENCH)
  hold(w, q0, 0.5, check_tilt=False)
  r.b.silence(timeout_s * 1.5)
  if r.b.is_sim:
    fb = r.b.command(r.to_raw(q0))
  else:
    r.b.drain()
    r.b.write_float(i, LOC_REF, r.to_raw(q0)[i])      # any frame makes the motor report its mode
    fb = r.b.feedback({i}, timeout=0.05)
  mode = int(fb["mode"][i])
  r.disable([i])
  res.check("motor stopped itself when commands stopped", MODE_NAMES.get(mode, "no reply"),
            mode == 0, "mode off")
  if mode == 2:
    res.info("motor kept running: CAN_TIMEOUT is not active - check CAN_TIMEOUT_RAW")
  # 2. tilt detection
  if C.IMU_TYPE != "none" or ARGS.sim:
    print("  2) Tilt detector: motors are OFF.  Slowly tilt the whole robot past 60 deg, then back.")
    wait_enter("  Ready? Start tilting after Enter (8 s).")
    gy, g = record_imu(w, 8.0 if not r.b.is_sim else 3.0,
                       sim=("tilt", dict(axis=[0, 1, 0], angle=1.2, duration=2.0, hold=False)))
    tilt = np.degrees(np.arccos(np.clip(-g[:, 2], -1, 1)))
    soft = np.nonzero(tilt > C.TILT_SOFT_DEG)[0]
    cut = np.nonzero(tilt > C.TILT_CUT_DEG)[0]
    res.check(f"soft stop at {C.TILT_SOFT_DEG:.0f} deg seen", f"max tilt {tilt.max():.0f} deg",
              len(soft) > 0)
    res.check(f"motor cut at {C.TILT_CUT_DEG:.0f} deg seen", "yes" if len(cut) else "no", len(cut) > 0)
    if r.b.is_sim:
      r.b.simulate("level")
  # 3. E-stop
  print("  3) E-stop: all motors hold the home pose, then YOU press the emergency stop.")
  if ask("  Do the E-stop test now?", "yn") == "y":
    q0 = r.enable(range(N), C.CURRENT_LIMIT_BENCH)
    ramp(w, q0, DEFAULT_Q, 3.0, check_tilt=False)
    print(color("  PRESS THE E-STOP NOW (within 20 s)", "y"))
    if r.b.is_sim:
      r.b.simulate("estop")
    rec = run_loop(w, 20.0, lambda k, t, s: DEFAULT_Q, check_tilt=False, allow_key=False)
    seen = rec.stop is not None and "no feedback" in rec.stop
    res.check("script noticed the E-stop", f"after {len(rec) * STEP_DT:.1f} s" if seen else "NO",
              seen)
    r.b.disable_everything()
    wait_enter("  Release the E-stop and power the motors back on.")
    if r.b.is_sim:
      r.b.simulate("estop_release")
    time.sleep(0.5)
    back = r.b.ping()
    res.check("all motors answer again", f"{int(back.sum())}/10", back.all())
  else:
    res.fail("E-stop test skipped")
  return res


def policy_prep(w):
  """Enable everything, go to home pose.  Returns the pose to park at."""
  r = w.robot
  q0 = r.enable(range(N), C.CURRENT_LIMIT_POLICY)
  ramp(w, q0, DEFAULT_Q, 3.0, check_tilt=False)
  hold(w, DEFAULT_Q, 0.5, check_tilt=False)
  w.actions_seen = []
  return q0


def action_report(res, w, rec):
  acts = np.array(w.actions_seen) if w.actions_seen else np.zeros((1, N))
  tgt = DEFAULT_Q + acts * ACTION_SCALE
  clipped = 100 * np.mean(np.abs(tgt) > C.JOINT_TARGET_LIMIT)
  res.check("actions finite", "yes" if np.isfinite(acts).all() else "NaN/inf",
            np.isfinite(acts).all())
  res.check("targets inside joint limits", f"{clipped:.1f}% clipped", clipped < 5, "< 5%")
  infer = rec.arr("infer") * 1000
  if len(infer):
    res.check("policy time", f"99% under {np.percentile(infer, 99):.2f} ms",
              np.percentile(infer, 99) < 5, "< 5 ms")
  jitter = np.abs(np.diff(acts, axis=0)).mean() if len(acts) > 1 else 0
  res.info(f"mean action change per step {jitter:.3f}  (big = shaky legs)")
  res.info("mean action R leg " + " ".join(f"{x:+.2f}" for x in acts[:, :5].mean(0))
           + "   L leg " + " ".join(f"{x:+.2f}" for x in acts[:, 5:].mean(0)))


def torque_report(res, rec):
  tau = rec.arr("tau")
  if not len(tau):
    return
  rms = np.sqrt(np.nanmean(tau ** 2, 0))
  joint_table(res, "torque RMS (% of rated):", 100 * rms / RATED_TQ, fmt="{:5.0f}%",
              bad=rms > RATED_TQ)
  res.check("torque RMS below rated", f"worst {np.nanmax(rms / RATED_TQ) * 100:.0f}%",
            np.all(rms <= RATED_TQ), "< 100%")


def s_dry_run(w):
  res = Result()
  if not (need_ids(res) and need_imu(w, res) and need_policy(w, res)):
    return res
  q0 = policy_prep(w)
  print("  Policy runs on the real sensor data but its output is NOT sent (motors hold home).")
  rec = run_loop(w, 5.0, policy_fn(w, lambda t: [0, 0, 0], send=False), check_tilt=False)
  park(w, q0)
  res.check("ran without stop", rec.stop or "yes", rec.stop is None)
  action_report(res, w, rec)
  save_rec(w, "policy_dry_run", rec)
  return res


def s_air_still(w):
  res = Result()
  if not (need_ids(res) and need_imu(w, res) and need_policy(w, res)):
    return res
  print("  Policy in control, command zero, feet in the air.  Some leg movement is normal")
  print("  (the policy expects the floor).  Press Enter to stop if it looks violent.")
  q0 = policy_prep(w)
  rec = run_loop(w, 5.0, policy_fn(w, lambda t: [0, 0, 0]), policy_mode=True)
  park(w, q0)
  res.check("ran 5 s without stop", rec.stop or "yes", rec.stop is None)
  action_report(res, w, rec)
  torque_report(res, rec)
  save_rec(w, "policy_air_still", rec)
  a = ask("  Did the legs stay calm (no shaking, no hitting anything)?", "yn")
  res.check("looked calm to you", "yes" if a == "y" else "no", a == "y")
  return res


def s_air_step(w):
  res = Result()
  if not (need_ids(res) and need_imu(w, res) and need_policy(w, res)):
    return res
  vx = 0.3
  print(f"  Policy in control, command vx = {vx} m/s, feet in the air: legs should step")
  print(f"  in the air, left and right taking turns, about every {GAIT_PERIOD:.1f} s.")
  q0 = policy_prep(w)
  rec = run_loop(w, 6.0, policy_fn(w, lambda t: [vx, 0, 0]), policy_mode=True)
  park(w, q0)
  res.check("ran 6 s without stop", rec.stop or "yes", rec.stop is None)
  q = rec.arr("q")
  if len(q) > 100:
    kr, kl = JOINTS.index("right_knee_pitch"), JOINTS.index("left_knee_pitch")
    hr = JOINTS.index("right_hip_pitch")
    x = q[50:, hr] - np.nanmean(q[50:, hr])
    spec = np.abs(np.fft.rfft(np.nan_to_num(x)))
    freqs = np.fft.rfftfreq(len(x), STEP_DT)
    f = freqs[1 + np.argmax(spec[1:])]
    res.check("stepping rhythm (hip swing)", f"{f:.2f} Hz", 1.2 < f < 2.2, f"~{1 / GAIT_PERIOD:.2f} Hz")
    hl = JOINTS.index("left_hip_pitch")
    c = np.corrcoef(np.nan_to_num(q[50:, hr]), np.nan_to_num(-q[50:, hl]))[0, 1]
    res.check("left/right take turns", f"hip correlation {c:+.2f}", c < 0, "< 0 (opposite)")
    res.info(f"knee swing R {np.ptp(q[50:, kr]):.2f} rad, L {np.ptp(q[50:, kl]):.2f} rad")
  action_report(res, w, rec)
  torque_report(res, rec)
  save_rec(w, "policy_air_step", rec)
  a = ask("  Did it look like stepping (smooth, left/right alternating)?", "yn")
  res.check("looked right to you", "yes" if a == "y" else "no", a == "y")
  return res


def gantry_run(w, name, commands):
  """Lower onto the floor, run the policy with the given (seconds, vx) list, raise again."""
  res = Result()
  if not (need_ids(res) and need_imu(w, res) and need_policy(w, res)):
    return res, None
  r = w.robot
  q0 = policy_prep(w)
  hold_until_enter(w, DEFAULT_Q, "Lower the gantry until both feet are FLAT on the floor "
                   "(rope still catches a fall)", sim_action="lower")
  total = sum(s for s, _ in commands)
  edges = np.cumsum([0] + [s for s, _ in commands])

  def cmd(t):
    k = min(np.searchsorted(edges, t, side="right") - 1, len(commands) - 1)
    return [commands[k][1], 0, 0]
  rec = run_loop(w, total, policy_fn(w, cmd), policy_mode=True, tilt_soft=C.TILT_SOFT_DEG)
  if r.enabled.any():
    ramp(w, r.target, DEFAULT_Q, 0.5, check_tilt=False, allow_key=False)
    hold_until_enter(w, DEFAULT_Q, "Raise the gantry so the feet are off the floor",
                     sim_action="raise")
  park(w, q0)
  tilt = rec.arr("tilt")
  res.check(f"ran {total:.0f} s without stop", rec.stop or "yes", rec.stop is None)
  res.check("body tilt", f"max {np.max(tilt) if len(tilt) else 0:.1f} deg",
            len(tilt) and np.max(tilt) < 10, "< 10 deg")
  torque_report(res, rec)
  action_report(res, w, rec)
  save_rec(w, name, rec)
  return res, rec


def s_gantry_stand(w):
  res, _ = gantry_run(w, "gantry_stand", [(20.0, 0.0)])
  if res.lines and not ARGS.yes:
    a = ask("  Did it stand quietly (no stomping, no leaning into the rope)?", "yn")
    res.check("looked right to you", "yes" if a == "y" else "no", a == "y")
  return res


def s_gantry_walk(w):
  cmds = [(3.0, 0.0)] + [(10.0, v) for v in C.WALK_TEST_SPEEDS] + [(3.0, 0.0)]
  print("  Walk on the spot/forward in the gantry: " +
        ", ".join(f"{v} m/s" for v in C.WALK_TEST_SPEEDS) + " for 10 s each.")
  print("  Walk the gantry along with the robot.  Enter = stop.")
  res, rec = gantry_run(w, "gantry_walk", cmds)
  if not ARGS.yes and rec is not None:
    a = ask("  Did it walk without catching the rope?", "yn")
    res.check("looked right to you", "yes" if a == "y" else "no", a == "y")
  return res


STEPS = [
  ("Safety checklist", "Make sure nothing can get hurt before any motor is powered.",
   "Hang the robot in the gantry, feet off the ground.", s_checklist),
  ("Config and policy", "Check hw_config.py and that the policy file has 41 inputs / 10 outputs.",
   "Nothing.", s_config),
  ("CAN ports", "Check every CAN channel in the config is up and can be opened.",
   "Motors powered.  Channels up: sudo ip link set canX up type can bitrate 1000000", s_ports),
  ("Find the motors", "Ask every motor for its position.  All 10 must answer.",
   "Motors powered.", s_find),
  ("Read positions (motors off)", "Read all joint angles without moving anything.",
   "Nothing.", s_positions),
  ("Joint order and direction", "Make sure each motor id belongs to the right joint and that "
   "+ means the same thing as in the sim.  THE most common real-robot bug.",
   "Motors stay OFF.  You move one joint at a time by hand.", s_order_sign),
  ("Zero check", "Legs straight must read 0.  An offset here shifts every pose the policy sees.",
   "A zeroing jig or a helper to hold the legs straight.", s_zero),
  ("IMU check", "Check the IMU's axes match the robot's (x forward, y left, z up).",
   "You will tilt the robot by hand.", s_imu),
  ("Enable and hold, one motor at a time",
   "First power-on of each motor: it must hold still, answer every frame, not heat up.",
   "Low current limit (hw_config CURRENT_LIMIT_BENCH).", s_enable_hold),
  ("Small motion, one motor at a time",
   "Each joint follows a slow +/-6 deg wave: checks direction, lag and smoothness.",
   "Legs free to swing.", s_small_motion),
  ("All motors to home pose", "All 10 motors ramp over 3 s to the policy's home pose.",
   "Legs free to swing.", s_home),
  ("50 Hz loop timing", "The policy was trained at exactly 50 Hz.  Checks period, jitter "
   "and lost frames while holding home.", "Nothing.", s_timing),
  ("Safety stops", "Motor watchdog, tilt detector and E-stop must all work before the policy runs.",
   "Be ready to press the E-stop.", s_safety),
  ("Policy dry run", "Policy reads the real sensors; outputs are checked but NOT sent.",
   "IMU working (step 8).", s_dry_run),
  ("Policy in the air, standing", "First time the policy moves the motors (feet in the air).",
   "Feet off the ground.  Hand on the E-stop.", s_air_still),
  ("Policy in the air, stepping", "vx = 0.3 m/s in the air: legs should alternate.",
   "Feet off the ground.  Hand on the E-stop.", s_air_step),
  ("Gantry: stand on the floor", "20 s standing with the gantry catching a fall.",
   "Gantry rope just slack.  Hand on the E-stop.", s_gantry_stand),
  ("Gantry: slow walk", "Walking at the speeds in WALK_TEST_SPEEDS, gantry catching.",
   "Clear floor, someone moving the gantry along.  Hand on the E-stop.", s_gantry_walk),
]


# ================================================================== wizard

class Wizard:
  def __init__(self, robot, run_dir):
    self.robot, self.run_dir = robot, run_dir
    self.policy = None
    self.actions_seen = []
    self.files = []
    self.results = []


def shutdown(w):
  if w is None:
    return
  try:
    b = w.robot.b
    if b.enabled.any():
      print(color("\nStopping: holding for 0.3 s, then disabling all motors ...", "y"))
      try:
        tgt = w.robot.to_raw(w.robot.target)
        for _ in range(15):
          b.command(tgt)
          time.sleep(STEP_DT)
      except Exception:  # noqa: BLE001
        pass
    b.disable_everything()
  except Exception as exc:  # noqa: BLE001
    print(f"(shutdown: {exc})")


def write_summary(w, done):
  lines = [f"OC1 bring-up  {datetime.now():%Y-%m-%d %H:%M}   backend: "
           f"{'sim' if ARGS.sim else C.CAN_BACKEND}   mode: {C.CONTROL_MODE}", ""]
  for num, title, status in w.results:
    lines.append(f"  step {num:2d}  {status:9s}  {title}")
  lines += ["", f"finished: {'yes' if done else 'no (stopped early)'}",
            f"files: {', '.join(w.files) or 'none'}"]
  if w.robot.saved:
    lines.append(f"saved fixes: {json.dumps(w.robot.saved)}")
  text = "\n".join(lines)
  (w.run_dir / "summary.txt").write_text(text + "\n")
  return text


def main():
  global ARGS, TWIN, REALTIME
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--start", type=int, default=1, help="step number to start from")
  p.add_argument("--only", type=int, default=None, help="run just this step")
  p.add_argument("--list", action="store_true", help="list the steps and exit")
  p.add_argument("--policy", default=None, help="override POLICY in hw_config.py")
  p.add_argument("--sim", action="store_true", help="no hardware: use a MuJoCo copy of the robot")
  p.add_argument("--yes", action="store_true", help="answer every question with the first choice "
                 "(only allowed with --sim)")
  p.add_argument("--sim-flip", nargs="*", default=[], help="sim test: these joints wired backwards")
  p.add_argument("--sim-swap", nargs=2, default=None, help="sim test: these two joints' ids swapped")
  p.add_argument("--sim-bad-zero", type=float, default=0.0, help="sim test: random zero error (rad)")
  p.add_argument("--twin", default=None, metavar="IP[:PORT]",
                 help=f"send live state to twin_viewer.py on this computer (port {TWIN_PORT})")
  p.add_argument("--fast", action="store_true", help="sim: do not wait in real time")
  ARGS = p.parse_args()
  if ARGS.list:
    for n, (title, why, *_r) in enumerate(STEPS, 1):
      print(f"{n:2d}. {title}\n    {why}")
    return
  if ARGS.yes and not ARGS.sim:
    sys.exit("--yes is only allowed with --sim (on the real robot you must confirm every step)")

  backend = SimBackend(ARGS.sim_flip, ARGS.sim_swap, ARGS.sim_bad_zero) if ARGS.sim else HardwareBackend()
  if ARGS.twin:
    TWIN = TwinSender(ARGS.twin)
  REALTIME = not (ARGS.sim and ARGS.fast)
  robot = Robot(backend)
  run_dir = ROOT / "runs" / "hardware" / (f"{datetime.now():%Y-%m-%d_%H-%M-%S}" + ("_sim" if ARGS.sim else ""))
  run_dir.mkdir(parents=True, exist_ok=True)
  w = Wizard(robot, run_dir)

  atexit.register(shutdown, w)

  def on_signal(*_):
    raise KeyboardInterrupt
  signal.signal(signal.SIGTERM, on_signal)

  print(color("\nOC1 hardware bring-up", "b") + f"   ({'SIM' if ARGS.sim else C.CAN_BACKEND}, "
        f"{C.CONTROL_MODE} mode, log -> {run_dir.relative_to(ROOT)})")
  print("At every step: y = go, s = skip, q = quit.  While motors move: Enter = stop.\n")

  todo = [ARGS.only] if ARGS.only else range(ARGS.start, len(STEPS) + 1)
  if ARGS.only or ARGS.start > 2:
    try:                                   # later steps need the policy loaded
      path = Path(ARGS.policy or C.POLICY)
      w.policy = Policy(path if path.is_absolute() else ROOT / path)
    except Exception as exc:  # noqa: BLE001
      print(color(f"(policy not loaded: {exc})", "y"))
    if not backend.is_sim:
      backend.open_ports()
  done = False
  try:
    for num in todo:
      title, why, before, fn = STEPS[num - 1]
      print(color(f"\n{'=' * 70}\nSTEP {num}/{len(STEPS)}: {title}\n{'=' * 70}", "b"))
      TWIN.update(force=True, step=f"STEP {num}/{len(STEPS)}: {title}", msg=before, stop="",
                  target=None, hl=-1)
      print(f"  Why:    {why}")
      print(f"  Before: {before}")
      a = ask("Start this step?", "ysq")
      if a == "q":
        break
      if a == "s":
        w.results.append((num, title, "skipped"))
        continue
      while True:
        try:
          res = fn(w)
        except StopRun as exc:
          res = Result()
          res.fail(str(exc))
        if w.robot.enabled.any():
          w.robot.b.disable_everything()
        print("\n".join(res.lines))
        TWIN.update(force=True, msg=f"STEP {num} {'PASSED' if res.ok else 'FAILED'}")
        if res.ok:
          print(color(f"  STEP {num} PASSED", "g"))
          a = ask("Everything looked right?  y = next step, r = repeat, q = quit", "yrq")
          status = "PASS"
        else:
          print(color(f"  STEP {num} FAILED - fix the problem above, then repeat", "r"))
          a = ask("r = repeat, c = continue anyway (not advised), q = quit", "rcq", auto="c")
          status = "FAIL"
        if a != "r":
          w.results.append((num, title, status if a != "c" else "FAIL-cont"))
          break
      if a == "q":
        break
    else:
      done = True
  except KeyboardInterrupt:
    print(color("\nInterrupted.", "y"))
  finally:
    shutdown(w)
    print("\n" + write_summary(w, done))
    print(f"\nLog folder: {run_dir}")
    backend.close()


ARGS = None

if __name__ == "__main__":
  main()
