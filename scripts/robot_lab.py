#!/usr/bin/env python3
"""
robot_lab.py - run an OC1 policy, push it, change the test conditions, watch every joint
torque, and run the whole walking-policy evaluation from the same window.

  python scripts/robot_lab.py                                          # latest run's policy.onnx
  python scripts/robot_lab.py --policy pretrained/policy.onnx
  python scripts/robot_lab.py --policy pretrained/policy.onnx --eval full    # no window: report only

Run it with plain `python`, not `mjpython`: the 3D view is rendered off-screen into the window.
Needs PySide6 and matplotlib on top of the training environment:  pip install PySide6 matplotlib

Window
  left    3D view
  middle  policy, velocity command, test conditions (foot friction, payload, motor strength,
          sensor noise), push, push-recovery ramp test
  right   tabs: Joint torque | Gait (live) | Evaluation

Keys (click the 3D view first): Up/Down vx, ,/. vy, Left/Right yaw rate, 0 stop,
Space apply push, Backspace reset, P pause.  Mouse on the 3D view: left-drag orbit,
right-drag or wheel zoom, double-click reset the camera.

Evaluation tab: the tests of the walking-policy report (speed tracking, straightness, response
time, gait quality, joint torque, push while walking, floor friction, payload, motor strength)
run on many robots in parallel with mujoco.rollout, off the GUI thread. It writes
eval_reports/<policy>_<time>.png + .json, lists what failed and how to fix it, and
"Watch in lab" replays a failed case in the 3D view under the same conditions.

The policy sees exactly the observations it was trained on: they come from
OC1VelocityEnv.observations() (noise, domain randomization and random pushes off, as in
play.py, unless you tick "sensor noise"). The physics is stepped here so that external forces
can be applied (MjData.xfrc_applied, at the chosen body's centre of mass).
"""
import argparse
import copy
import csv
import json
import math
import os
import sys
import textwrap
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
from mujoco import rollout
from PySide6 import QtCore, QtGui, QtWidgets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import oc1_rl.env as envmod  # noqa: E402
from oc1_rl import robot  # noqa: E402
from oc1_rl.env import (DECIMATION, NUM_ACTOR_OBS, STEP_DT, EnvCfg, OC1VelocityEnv,  # noqa: E402
                        wrap_to_pi, yaw_of)

GAIT_PERIOD = getattr(envmod, "GAIT_PERIOD", 0.6)
Qt = QtCore.Qt

# RobStride datasheets, N*m: (rated / continuous, peak)
MOTORS = {"RS04": (40.0, 120.0), "RS03": (21.0, 60.0)}
JOINT_MOTOR = {"hip_pitch": "RS04", "hip_roll": "RS04", "knee_pitch": "RS04",
               "hip_yaw": "RS03", "ankle_pitch": "RS03"}
JOINT_TYPES = ["hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch"]
CMD_LIMITS = np.array([[-1.0, 2.0], [-1.0, 1.0], [-1.0, 1.0]])   # same as play.py
FALL_TILT_DEG = 70.0          # training termination angle
HISTORY_S = 5.0
LIVE_WINDOW_S = 4.0           # gait (live) panel window
VIEW_W, VIEW_H = 960, 720

C = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#8a8984", grid="#e4e3df",
         track="#f0efec", left="#2a78d6", right="#eb6834", warn="#fab219", crit="#d03b3b",
         push="#4a3aa7", good="#0ca30c", warn_text="#9a6a00", header="#ebeae6")

# ---------------------------------------------------------------- evaluation settings
EVAL_SPEEDS = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)          # commanded vx, m/s
EVAL_FRICTIONS = (0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6)     # foot-floor mu
EVAL_PAYLOADS = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0)               # kg on the torso
EVAL_STRENGTHS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)              # fraction of motor peak torque
PUSH_DIRS_DEG = (0, 45, 90, 135, 180, 225, 270, 315)         # force direction, 0 = forward, CCW
PUSH_DIR_NAMES = ("forward", "forward-left", "left", "back-left", "backward", "back-right",
                  "right", "forward-right")
PRESETS = {
    # trials per cell. "full" matches the report: 8 per speed, 16 per push cell, 4 per condition
    "quick": dict(nominal=4, push_forces=(50, 100, 150, 200, 250), push_repeats=1, cond=2),
    "full": dict(nominal=8, push_forces=tuple(range(25, 251, 25)), push_repeats=2, cond=4),
}
OPTIONAL_TESTS = ("push", "friction", "payload", "motor")
WALK_S = 10.0           # nominal walk trial (command switches on at ~1 s)
COND_S = 8.0            # friction / payload / motor trial
CMD_ONSET = (0.8, 1.4)  # command switches on at a random time -> random gait phase at the start
STEADY_AFTER_S = 3.0    # speed error and gait are measured from this long after the command onset
PUSH_AFTER_S = 2.5      # push lands this long after the onset, plus a random part of a gait cycle
PUSH_WATCH_S = 3.0      # survived = still up this long after the push ended
INIT_JOINT_JITTER = 0.01
MAX_BATCH = 768
SURVIVE_OK = 0.9        # a cell passes at >= 90 % survival
TRACK_OK = 0.15         # m/s, a command counts as followed

# What each test is for. Shown in the Evaluation tab next to its findings.
TESTS = {
    "walk": ("Speed tracking, straightness, response, gait, torque",
             "The basic contract of a velocity policy: when you ask for a speed, does the robot "
             "walk at that speed, in a straight line, soon after you ask, with a gait and joint "
             "torques the hardware can sustain? Every other test is judged against this one: a "
             "speed where it already falls undisturbed tells you nothing about pushes."),
    "push": ("Push while walking",
             "Real robots get bumped, catch a foot, land off balance and react to their own "
             "modelling errors. A push of known force and duration from 8 directions measures how "
             "much disturbance the policy rejects, and which direction it is weakest in."),
    "friction": ("Floor friction",
                 "Sim floors are perfectly uniform; real floors are tiles, wood, dust and wet spots. "
                 "Low friction exposes a gait that relies on the feet not sliding; high friction "
                 "exposes toe stubbing. Training randomizes friction, this shows if that was enough."),
    "payload": ("Payload",
                "Batteries, a computer, cables and the real robot's CAD-vs-actual mass error all "
                "add mass to the torso. This shows how much extra mass the gait tolerates before "
                "it falls or slows down."),
    "motor": ("Motor strength",
              "Real actuators deliver less than the datasheet peak: hot windings, low battery "
              "voltage, gearbox friction, current limits in the driver. Lowering the torque limit "
              "shows how much margin the policy keeps."),
    "model": ("Model check",
              "Not a policy test: properties of the URDF/MJCF that limit what any policy can do."),
}

GAIT_ROWS = (
    # key, label, format, what's good, (low, high) range for the warning flag or None
    ("cot", "Cost of transport", "{:.2f}", "lower = more efficient", None),
    ("power", "Mechanical power (W)", "{:.0f}", "lower = less effort", None),
    ("slip", "Foot slip in contact (m/s)", "{:.3f}", "< 0.05 is good", (None, 0.05)),
    ("clearance", "Foot clearance (cm)", "{:.1f}", "≈ 5–10 cm; too low trips", (5.0, 10.0)),
    ("steps", "Steps per second", "{:.2f}", "gait rate", None),
    ("asym", "Left/right asymmetry (%)", "{:.1f}", "< 5% = no limp", (None, 5.0)),
    ("jump", "Action jumps per step", "{:.3f}", "lower = smoother motors", None),
    ("roll", "Torso roll wobble (° RMS)", "{:.1f}", "a few degrees", (None, 4.0)),
    ("pitch", "Torso pitch wobble (° RMS)", "{:.1f}", "a few degrees", (None, 4.0)),
    ("height", "Base height wobble (cm)", "{:.1f}", "small = no bouncing", (None, 1.5)),
    ("at_limit", "Time at torque limit (%)", "{:.1f}", "near 0 is safe", (None, 0.5)),
)
REC_KEYS = ("pos", "quat", "vel", "jvel", "tau", "fz", "fv", "contact")


def motor_of(joint):
    for k, m in JOINT_MOTOR.items():
        if joint.endswith(k):
            return m
    raise KeyError(joint)


def short_joint(name):
    side = "R" if name.startswith("right") else "L"
    return f"{side} " + name.split("_", 1)[1].replace("_pitch", "").replace("_", " ")


# ====================================================================== shared math
def rot_inv(q, v):
    """World vectors v (..., 3) into the frames of unit quaternions q (..., 4) (w, x, y, z)."""
    w, u = q[..., :1], q[..., 1:]
    return v * (2 * w**2 - 1) - 2 * w * np.cross(u, v) + 2 * u * np.sum(u * v, -1, keepdims=True)


def roll_pitch(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def tilt_of(q):
    """Angle between the base z axis and world z (rad), same as the training termination."""
    x, y = q[..., 1], q[..., 2]
    return np.arccos(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0))


def sensor_slices(model):
    return {model.sensor(i).name: slice(model.sensor_adr[i], model.sensor_adr[i] + model.sensor_dim[i])
            for i in range(model.nsensor)}


def _runs(mask):
    """(starts, ends) of the True runs of a 1-D bool array, ends exclusive."""
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return e[::2], e[1::2]


def _mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def _std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.std()) if x.size else float("nan")


def gait_metrics(r, da, mass, limit, dt):
    """Gait quality of ONE robot over a window (same formulas for the live panel and the eval).

    r: per-physics-step arrays: pos (T,3), quat (T,4), vel (T,3) world linear velocity,
       jvel (T,nj), tau (T,nj), fz (T,2) foot-sole height, fv (T,2) foot horizontal speed,
       contact (T,2), feet ordered left, right.
    da: (P,) RMS action change per policy step.  mass: kg.  limit: (nj,) torque limits, N*m.
    """
    T = len(r["tau"])
    quat = np.asarray(r["quat"], dtype=float)
    v_b = rot_inv(quat, np.asarray(r["vel"], dtype=float))
    speed = float(np.mean(np.linalg.norm(v_b[:, :2], axis=1)))
    tau, jvel = np.asarray(r["tau"], dtype=float), np.asarray(r["jvel"], dtype=float)
    power = float(np.mean(np.sum(np.abs(tau * jvel), axis=1)))
    c = np.asarray(r["contact"]).astype(bool)
    fv = np.asarray(r["fv"], dtype=float)
    fz = np.asarray(r["fz"], dtype=float)
    slip = float(np.mean(fv[c])) if c.any() else 0.0
    min_swing = max(2, int(round(0.03 / dt)))
    clear, steps = [], 0
    for f in range(2):
        z, cf = fz[:, f], c[:, f]
        stance_z = float(np.median(z[cf])) if cf.any() else float(z.min())
        for a, b in zip(*_runs(~cf)):
            if a > 0 and b < T and b - a >= min_swing:   # a whole swing: lift-off and touchdown seen
                clear.append(z[a:b].max() - stance_z)
                steps += 1
    duty = c.mean(axis=0)
    roll, pitch = roll_pitch(quat)
    return dict(
        speed=speed,
        power=power,
        cot=power / (mass * 9.81 * speed) if speed > 0.05 else float("nan"),
        slip=slip,
        clearance=100.0 * float(np.mean(clear)) if clear else 0.0,
        steps=steps / (T * dt),
        asym=100.0 * abs(duty[0] - duty[1]) / max(float(duty.mean()), 1e-6),
        jump=float(np.mean(da)) if len(da) else 0.0,
        roll=math.degrees(float(np.std(roll))),
        pitch=math.degrees(float(np.std(pitch))),
        height=100.0 * float(np.std(np.asarray(r["pos"], dtype=float)[:, 2])),
        at_limit=100.0 * float(np.mean(np.abs(tau) >= 0.98 * np.asarray(limit))),
    )


# ====================================================================== test conditions
def foot_geom_ids(model):
    return np.array([i for i in range(model.ngeom) if robot.FOOT_GEOM_RE.match(model.geom(i).name or "")])


@dataclass
class Conditions:
    """The levers the evaluation turns, applied to a model by apply_conditions()."""
    friction: float = 0.6     # foot-floor sliding friction (the foot spheres have contact priority)
    payload: float = 0.0      # kg added to the torso at its centre of mass
    strength: float = 1.0     # fraction of each motor's peak torque that is available

    def label(self, nominal_friction):
        parts = []
        if abs(self.friction - nominal_friction) > 1e-6:
            parts.append(f"μ {self.friction:.2f}")
        if self.payload > 0:
            parts.append(f"+{self.payload:g} kg")
        if self.strength < 1.0 - 1e-6:
            parts.append(f"motors {self.strength:.0%}")
        return ", ".join(parts) or "nominal"


class Nominal:
    """The model values that Conditions change, as built by robot.make_model()."""

    def __init__(self, model):
        self.feet = foot_geom_ids(model)
        self.friction = float(model.geom_friction[self.feet, 0].mean())
        self.base = model.body(robot.BASE_BODY).id
        self.mass = float(model.body_mass[self.base])
        self.inertia = model.body_inertia[self.base].copy()
        self.forcerange = model.actuator_forcerange.copy()
        self.total_mass = float(model.body_mass.sum())

    def conditions(self, **kw):
        c = Conditions(friction=self.friction)
        for k, v in kw.items():
            setattr(c, k, v)
        return c


def apply_conditions(model, nom, cond):
    model.geom_friction[nom.feet, 0] = cond.friction
    model.body_mass[nom.base] = nom.mass + cond.payload
    # payload spread like the torso itself: its inertia scales with the mass
    model.body_inertia[nom.base] = nom.inertia * (nom.mass + cond.payload) / nom.mass
    mujoco.mj_setConst(model, mujoco.MjData(model))   # subtree masses; scratch data, never the live one
    model.actuator_forcerange[:] = nom.forcerange * cond.strength


def model_check(model):
    """URDF properties that cap what any policy can do: leg mass balance and joint ranges."""
    legs = {}
    for side in ("left", "right"):
        root = model.jnt_bodyid[model.joint(f"{side}_hip_pitch").id]
        total = 0.0
        for b in range(1, model.nbody):
            a = b
            while a > 0 and a != root:
                a = model.body_parentid[a]
            if a == root:
                total += float(model.body_mass[b])
        legs[side] = total
    d = mujoco.MjData(model)
    d.qpos[3] = 1.0
    d.qpos[7:] = robot.DEFAULT_JOINT_POS
    mujoco.mj_forward(model, d)
    out = dict(leg_mass_left=legs["left"], leg_mass_right=legs["right"],
               com_y=float(d.subtree_com[0][1] - d.xpos[model.body(robot.BASE_BODY).id][1]))
    for side in ("left", "right"):
        j = model.joint(f"{side}_knee_pitch").id
        q0 = robot.DEFAULT_JOINT_POS[robot.JOINT_NAMES.index(f"{side}_knee_pitch")]
        lo, hi = model.jnt_range[j]
        out[f"knee_flex_max_{side}"] = float(hi if q0 >= 0 else -lo)   # bend direction = home-pose sign
    out["knee_flex_max"] = min(out["knee_flex_max_left"], out["knee_flex_max_right"])
    out["knee_home"] = float(abs(robot.DEFAULT_JOINT_POS[robot.JOINT_NAMES.index("left_knee_pitch")]))
    hp = model.jnt_range[model.joint("left_hip_pitch").id]
    out["hip_pitch_range"] = [float(hp[0]), float(hp[1])]
    return out


# ====================================================================== live simulation
class LiveGait:
    """Rolling window of the live robot's physics samples for the Gait (live) tab."""

    def __init__(self, seconds, dt):
        self.dt = dt
        self.n = int(round(seconds / dt))
        self.buf = {k: deque(maxlen=self.n) for k in REC_KEYS}
        self.da = deque(maxlen=int(round(seconds / STEP_DT)))

    def clear(self):
        for b in self.buf.values():
            b.clear()
        self.da.clear()

    def add(self, sample):
        for k, v in sample.items():
            self.buf[k].append(v)

    def metrics(self, mass, limit):
        if len(self.buf["tau"]) < self.n // 2:
            return None
        r = {k: np.array(v) for k, v in self.buf.items()}
        return gait_metrics(r, np.array(self.da), mass, limit, self.dt)


class Sim:
    """One robot: OC1VelocityEnv builds the observations, physics is stepped on one MjData."""

    def __init__(self):
        self.model = robot.make_model(visual=True)
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, VIEW_W)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, VIEW_H)
        self.env = OC1VelocityEnv(EnvCfg(num_envs=1, nthread=1, obs_noise=False,
                                         domain_randomization=False, pushes=False,
                                         terminate_on_timeout=False), model=self.model)
        self.data = mujoco.MjData(self.model)
        self.base = self.model.body(robot.BASE_BODY).id
        # actuators were added in JOINT_NAMES order, which is also the env's ctrl order
        self.act_ids = np.array([self.model.actuator(n).id for n in robot.JOINT_NAMES])
        assert list(self.act_ids) == list(range(len(robot.JOINT_NAMES)))
        self.bodies = [self.model.body(i).name for i in range(1, self.model.nbody)
                       if self.model.body_jntnum[i] > 0]
        self.nom = Nominal(self.model)
        self.cond = self.nom.conditions()
        self.sl = sensor_slices(self.model)
        self.live = LiveGait(LIVE_WINDOW_S, self.model.opt.timestep)
        self.sess = None
        self.policy_path = None
        self.cmd = np.zeros(3)
        self.heading_hold = True
        self.push_body = self.base
        self.push_force = np.zeros(3)
        self.push_left = 0.0
        self.reset()

    # ---- policy
    def load_policy(self, path):
        sess = ort.InferenceSession(str(path))
        n = sess.get_inputs()[0].shape[-1]
        if n != NUM_ACTOR_OBS:
            raise ValueError(f"policy expects {n} observations, this env makes {NUM_ACTOR_OBS}")
        self.sess, self.policy_path = sess, Path(path)
        self.input_name = sess.get_inputs()[0].name

    # ---- conditions
    def set_conditions(self, cond, obs_noise=None):
        apply_conditions(self.model, self.nom, cond)
        self.cond = cond
        if obs_noise is not None:
            self.env.cfg.obs_noise = bool(obs_noise)

    def mass(self):
        return float(self.model.body_mass.sum())

    def torque_limits(self):
        return self.model.actuator_forcerange[self.act_ids, 1].copy()

    # ---- state
    def reset(self):
        self.env.reset_envs(np.array([0]))
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, self.env.state[0], self.env.state_spec)
        mujoco.mj_forward(self.model, self.data)
        self.heading_target = self.heading()
        self.push_left = 0.0
        self.push_force[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.live.clear()
        self.obs = self._observe()

    def heading(self):
        return float(yaw_of(self.data.qpos[None, 3:7])[0])

    def tilt_deg(self):
        return math.degrees(math.acos(np.clip(self.data.xmat[self.base][8], -1.0, 1.0)))

    def fallen(self):
        return self.tilt_deg() > FALL_TILT_DEG or not np.all(np.isfinite(self.data.qpos))

    def base_velocity(self):
        R = self.data.xmat[self.base].reshape(3, 3)
        return R.T @ self.data.qvel[0:3]

    def command(self):
        c = self.cmd.copy()
        if self.heading_hold and c[2] == 0.0:   # same heading controller as play.py
            c[2] = np.clip(0.5 * wrap_to_pi(self.heading_target - self.heading()), *CMD_LIMITS[2])
        else:
            self.heading_target = self.heading()
        return c

    def _observe(self):
        env = self.env
        mujoco.mj_getState(self.model, self.data, env.state[0], env.state_spec)
        env.command[0] = self.command()
        return env.observations()[0]

    # ---- push
    def start_push(self, body_id, force_world, duration):
        self.push_body, self.push_force[:] = body_id, force_world
        self.push_left = duration

    # ---- step
    def _sample(self, tau):
        d, sd, sl = self.data, self.data.sensordata, self.sl
        return dict(pos=d.qpos[0:3].copy(), quat=d.qpos[3:7].copy(), vel=d.qvel[0:3].copy(),
                    jvel=d.qvel[6:].copy(), tau=tau,
                    fz=np.array([sd[sl["left_foot_pos"]][2], sd[sl["right_foot_pos"]][2]]),
                    fv=np.array([np.linalg.norm(sd[sl["left_foot_vel"]][:2]),
                                 np.linalg.norm(sd[sl["right_foot_vel"]][:2])]),
                    contact=np.array([sd[sl["left_foot_contact"]][0] > 0,
                                      sd[sl["right_foot_contact"]][0] > 0]))

    def step(self):
        """One policy step (DECIMATION physics steps). Returns torques (DECIMATION, nj) and push flags."""
        env = self.env
        if self.sess is not None:
            action = self.sess.run(None, {self.input_name: self.obs})[0].astype(np.float64).reshape(1, -1)
        else:
            action = np.zeros((1, robot.NUM_JOINTS))
        env.prev_action = env.action
        env.action = action
        self.live.da.append(float(np.sqrt(np.mean((action - env.prev_action) ** 2))))
        self.data.ctrl[:] = env.default_q + action[0] * env.action_scale
        taus, flags = [], []
        dt = self.model.opt.timestep
        for _ in range(DECIMATION):
            self.data.xfrc_applied[:] = 0.0
            on = self.push_left > 1e-9
            if on:
                self.data.xfrc_applied[self.push_body, :3] = self.push_force
                self.push_left -= dt
            mujoco.mj_step(self.model, self.data)
            tau = self.data.actuator_force[self.act_ids].copy()
            taus.append(tau)
            flags.append(on)
            self.live.add(self._sample(tau))
        self.data.xfrc_applied[:] = 0.0
        env.episode_len += 1
        self.obs = self._observe()
        return np.array(taus), np.array(flags)


# ====================================================================== batched evaluation
class Stopped(Exception):
    pass


class PolicyRunner:
    """ONNX policy run on a whole batch of observations (relaxes a fixed batch size of 1)."""

    def __init__(self, path):
        sess = ort.InferenceSession(str(path))
        inp = sess.get_inputs()[0]
        if inp.shape[-1] != NUM_ACTOR_OBS:
            raise ValueError(f"policy expects {inp.shape[-1]} observations, this env makes {NUM_ACTOR_OBS}")
        self.batched = not (isinstance(inp.shape[0], int) and inp.shape[0] == 1)
        if not self.batched:
            try:
                import onnx
                m = onnx.load(str(path))
                for v in list(m.graph.input) + list(m.graph.output):
                    v.type.tensor_type.shape.dim[0].dim_param = "batch"
                sess = ort.InferenceSession(m.SerializeToString())
                self.batched = True
            except Exception:  # noqa: BLE001  - fall back to one row at a time
                pass
        self.sess, self.name = sess, sess.get_inputs()[0].name

    def __call__(self, obs):
        if self.batched:
            return self.sess.run(None, {self.name: obs})[0].astype(np.float64)
        return np.concatenate([self.sess.run(None, {self.name: o[None]})[0] for o in obs]).astype(np.float64)


ROW_DEFAULTS = dict(model=0, vx=0.0, vy=0.0, wz=0.0, t_cmd=1.0, end=WALK_S, push_t=np.inf,
                    push_dur=0.0, push_force=0.0, push_angle=0.0)


def run_batch(model, models, policy, rows, *, settle, record=False, heading_hold=False,
              obs_noise=False, nthread=None, seed=0, progress=None, stop=None):
    """Run one robot per row, all in parallel, and return per-robot results.

    model:  base MjModel (observations are built against it)
    models: MjModel copies with the test conditions; rows[i]["model"] indexes this list
    rows:   dicts with vx vy wz (command after t_cmd, zero before), t_cmd, end (s),
            push_t, push_dur, push_force (N, on the torso centre of mass),
            push_angle (rad, relative to the heading when the push starts)
    settle: speed error / mean velocity are averaged from t_cmd + settle to end
    record: also keep every physics step (for gait metrics, torque, response time)
    """
    n = len(rows)
    T = {k: np.array([r.get(k, dflt) for r in rows], dtype=int if k == "model" else float)
         for k, dflt in ROW_DEFAULTS.items()}
    nthread = max(1, nthread or os.cpu_count() or 1)
    rng = np.random.default_rng(seed + 7919)
    env = OC1VelocityEnv(EnvCfg(num_envs=n, nthread=1, obs_noise=obs_noise, domain_randomization=False,
                                pushes=False, terminate_on_timeout=False, seed=seed), model=model)
    pool = rollout.Rollout(nthread=nthread)
    datas = [mujoco.MjData(model) for _ in range(nthread)]
    try:
        m = model
        nu, nj, dt = m.nu, robot.NUM_JOINTS, m.opt.timestep
        S = mujoco.mj_stateSize(m, env.state_spec)
        assert env.state.shape[1] == S
        qs, vs = slice(1, 1 + m.nq), slice(1 + m.nq, 1 + m.nq + m.nv)   # FULLPHYSICS = [time, qpos, qvel, ...]
        mlist = [models[i] for i in T["model"]]
        par = lambda f: np.stack([f(mm) for mm in models])[T["model"]]   # noqa: E731
        gain = par(lambda mm: mm.actuator_gainprm[:, 0])
        b1, b2 = par(lambda mm: mm.actuator_biasprm[:, 1]), par(lambda mm: mm.actuator_biasprm[:, 2])
        lo, hi = par(lambda mm: mm.actuator_forcerange[:, 0]), par(lambda mm: mm.actuator_forcerange[:, 1])
        mass = par(lambda mm: np.array([mm.body_mass.sum()]))[:, 0]

        # start: home pose, random heading, small joint jitter (feet start 3 mm above the floor)
        d = mujoco.MjData(m)
        d.qpos[2] = robot.init_base_height(m) + 0.003
        d.qpos[3] = 1.0
        d.qpos[7:] = env.default_q
        init = np.zeros(S)
        mujoco.mj_getState(m, d, init, env.state_spec)
        state = np.tile(init, (n, 1))
        yaw0 = rng.uniform(-np.pi, np.pi, n)
        state[:, 1 + 3] = np.cos(yaw0 / 2)            # qpos[3:7] = (cos, 0, 0, sin) of yaw / 2
        state[:, 1 + 6] = np.sin(yaw0 / 2)
        state[:, 1 + 7:1 + m.nq] += rng.uniform(-INIT_JOINT_JITTER, INIT_JOINT_JITTER, (n, nj))

        P = int(math.ceil(T["end"].max() / STEP_DT - 1e-9))
        spec = int(mujoco.mjtState.mjSTATE_CTRL | mujoco.mjtState.mjSTATE_XFRC_APPLIED)
        ctrl = np.zeros((n, DECIMATION, nu + 6 * m.nbody))
        fcol = nu + 6 * m.body(robot.BASE_BODY).id
        so = np.zeros((n, DECIMATION, S))
        se = np.zeros((n, DECIMATION, m.nsensordata))
        sl = sensor_slices(m)
        target = np.stack([T["vx"], T["vy"], T["wz"]], 1)
        head_target = yaw0.copy()
        action = np.zeros((n, nj))
        fall_t = np.full(n, np.inf)
        f_world = np.zeros((n, 3))
        started = np.zeros(n, bool)
        err_sum, v_sum, cnt = np.zeros(n), np.zeros((n, 2)), np.zeros(n)
        if record:
            width = dict(pos=3, quat=4, vel=3, jvel=nj, tau=nj, fz=2, fv=2, contact=2)
            rec = {k: np.zeros((n, P * DECIMATION, w), np.float32) for k, w in width.items()}
            da = np.zeros((n, P), np.float32)

        for k in range(P):
            t = k * STEP_DT
            if stop is not None and stop():
                raise Stopped()
            if progress is not None and k % 10 == 0:
                progress(k / P)
            q = state[:, qs]
            heading = yaw_of(q[:, 3:7])
            cmd = np.where((t >= T["t_cmd"] - 1e-9)[:, None], target, 0.0)
            if heading_hold:                                 # same heading controller as play.py
                hold = cmd[:, 2] == 0.0
                cmd[hold, 2] = np.clip(0.5 * wrap_to_pi(head_target[hold] - heading[hold]), *CMD_LIMITS[2])
                head_target[~hold] = heading[~hold]
            env.state = state
            env.command = cmd
            env.episode_len = np.full(n, k, dtype=np.int64)
            env.action = action
            new = policy(env.observations()[0])
            if record:
                da[:, k] = np.sqrt(np.mean((new - action) ** 2, axis=1))
            action = new
            ctrl[:, :, :nu] = (env.default_q + action * env.action_scale)[:, None, :]
            ctrl[:, :, nu:] = 0.0
            begin = ~started & (T["push_t"] < t + STEP_DT - 1e-9)
            if begin.any():
                a = heading[begin] + T["push_angle"][begin]
                f_world[begin] = T["push_force"][begin, None] * np.stack([np.cos(a), np.sin(a), 0 * a], 1)
                started |= begin
            if started.any():
                for j in range(DECIMATION):
                    ts = t + j * dt
                    on = started & (ts >= T["push_t"] - 1e-9) & (ts < T["push_t"] + T["push_dur"] - 1e-9)
                    ctrl[on, j, fcol:fcol + 3] = f_world[on]
            prev = state
            pool.rollout(mlist, datas, state, ctrl, control_spec=spec, nstep=DECIMATION,
                         skip_checks=True, state=so, sensordata=se)
            state = so[:, -1].copy()
            env.sensordata = se[:, -1].copy()

            bad = ~np.isfinite(state).all(axis=1)
            if bad.any():
                state[bad] = init
            qn = state[:, qs]
            down = bad | (tilt_of(qn[:, 3:7]) > math.radians(FALL_TILT_DEG))
            fall_t[down & np.isinf(fall_t)] = t + STEP_DT
            win = (np.isinf(fall_t) & (t + STEP_DT >= T["t_cmd"] + settle - 1e-9)
                   & (t + STEP_DT <= T["end"] + 1e-9))
            if win.any():
                vb = rot_inv(qn[win, 3:7], state[win, vs][:, 0:3])
                err_sum[win] += np.linalg.norm(vb[:, :2] - cmd[win, :2], axis=1)
                v_sum[win] += vb[:, :2]
                cnt[win] += 1
            if record:
                w = slice(k * DECIMATION, (k + 1) * DECIMATION)
                before = np.concatenate([prev[:, None], so[:, :-1]], axis=1)
                tau = np.clip(gain[:, None] * ctrl[:, :, :nu] + b1[:, None] * before[..., qs][..., 7:]
                              + b2[:, None] * before[..., vs][..., 6:], lo[:, None], hi[:, None])
                rec["pos"][:, w] = so[..., qs][..., 0:3]
                rec["quat"][:, w] = so[..., qs][..., 3:7]
                rec["vel"][:, w] = so[..., vs][..., 0:3]
                rec["jvel"][:, w] = so[..., vs][..., 6:]
                rec["tau"][:, w] = tau
                rec["fz"][:, w] = np.stack([se[..., sl["left_foot_pos"]][..., 2],
                                            se[..., sl["right_foot_pos"]][..., 2]], -1)
                rec["fv"][:, w] = np.stack([np.linalg.norm(se[..., sl["left_foot_vel"]][..., :2], axis=-1),
                                            np.linalg.norm(se[..., sl["right_foot_vel"]][..., :2], axis=-1)], -1)
                rec["contact"][:, w] = np.stack([se[..., sl["left_foot_contact"]][..., 0] > 0,
                                                 se[..., sl["right_foot_contact"]][..., 0] > 0], -1)
        if progress is not None:
            progress(1.0)
        c = np.maximum(cnt, 1)
        out = dict(fall_t=fall_t, survived=fall_t > T["end"] + 1e-9,
                   err=np.where(cnt > 0, err_sum / c, np.nan),
                   vx=np.where(cnt > 0, v_sum[:, 0] / c, np.nan),
                   vy=np.where(cnt > 0, v_sum[:, 1] / c, np.nan),
                   t_cmd=T["t_cmd"], end=T["end"], limit=hi, mass=mass)
        if record:
            out["rec"], out["da"] = rec, da
        return out
    finally:
        pool.close()
        env.close()


def straightness(rec, j, t_cmd, dt):
    """Sideways drift (cm per m walked) and heading change (deg per 10 m) from 1 s after the onset."""
    a = int((t_cmd + 1.0) / dt)
    p = rec["pos"][j, a:, :2].astype(float)
    yaw = np.unwrap(yaw_of(rec["quat"][j, a:].astype(float)))
    h = yaw[0]
    d = p[-1] - p[0]
    fwd = d @ np.array([math.cos(h), math.sin(h)])
    lat = d @ np.array([-math.sin(h), math.cos(h)])
    path = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    drift = 100.0 * abs(lat) / abs(fwd) if abs(fwd) > 0.3 else float("nan")
    head = abs(math.degrees(yaw[-1] - yaw[0])) * 10.0 / path if path > 0.3 else float("nan")
    return drift, head


def response_time(rec, j, t_cmd, dt):
    """Time from the command step until forward speed first reaches 90 % of its steady value."""
    c0 = int(round(t_cmd / dt))
    vb = rot_inv(rec["quat"][j, c0:].astype(float), rec["vel"][j, c0:].astype(float))[:, 0]
    k = max(1, int(round(0.2 / dt)))
    smooth = np.convolve(vb, np.ones(k) / k, mode="same")
    steady = float(vb[-int(round(3.0 / dt)):].mean())
    if abs(steady) < 0.05:
        return float("nan")
    hit = np.flatnonzero(np.sign(steady) * smooth >= 0.9 * abs(steady))
    return float(hit[0] * dt) if len(hit) else float("nan")


class EvalSuite:
    """All the tests of the walking-policy report. run() returns a JSON-able results dict."""

    def __init__(self, policy_path, preset="full", tests=OPTIONAL_TESTS, push_duration=0.2,
                 heading_hold=False, obs_noise=False, nthread=None, seed=0, progress=None, stop=None):
        self.policy_path = Path(policy_path)
        self.preset, self.p = preset, PRESETS[preset]
        self.tests = [t for t in OPTIONAL_TESTS if t in tests]
        self.push_duration = float(push_duration)
        self.heading_hold, self.obs_noise = bool(heading_hold), bool(obs_noise)
        self.nthread = nthread or os.cpu_count() or 1
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._progress, self._stop = progress, stop
        self.total, self.done = 1.0, 0.0
        self._say("building the model")
        self.model = robot.make_model(visual=False)
        self.nom = Nominal(self.model)
        self.policy = PolicyRunner(self.policy_path)
        nS = len(EVAL_SPEEDS)
        cost = {"walk": nS * self.p["nominal"] * WALK_S,
                "push": nS * len(self.p["push_forces"]) * len(PUSH_DIRS_DEG) * self.p["push_repeats"]
                * (PUSH_AFTER_S + 1.1 + GAIT_PERIOD / 2 + self.push_duration + PUSH_WATCH_S),
                "friction": nS * len(EVAL_FRICTIONS) * self.p["cond"] * COND_S,
                "payload": nS * len(EVAL_PAYLOADS) * self.p["cond"] * COND_S,
                "motor": nS * len(EVAL_STRENGTHS) * self.p["cond"] * COND_S}
        self.cost = {k: v for k, v in cost.items() if k == "walk" or k in self.tests}
        self.total = sum(self.cost.values())
        self.done = 0.0

    def _say(self, msg, frac=None):
        if self._progress is not None:
            self._progress(self.done / self.total if frac is None else frac, msg)

    def _run(self, test, models, rows, settle, record=False, label=""):
        chunks = [rows[i:i + MAX_BATCH] for i in range(0, len(rows), MAX_BATCH)]
        outs = []
        for ci, ch in enumerate(chunks):
            share = self.cost[test] * len(ch) / len(rows)
            base = self.done
            msg = f"{label}: {len(rows)} robots" + (f" (batch {ci + 1}/{len(chunks)})" if len(chunks) > 1 else "")

            def prog(f, base=base, share=share, msg=msg):
                if self._progress is not None:
                    self._progress((base + f * share) / self.total, msg)

            outs.append(run_batch(self.model, models, self.policy, ch, settle=settle, record=record,
                                  heading_hold=self.heading_hold, obs_noise=self.obs_noise,
                                  nthread=self.nthread,
                                  seed=self.seed + 101 * len(outs) + 1000 * ("walk", *OPTIONAL_TESTS).index(test),
                                  progress=prog, stop=self._stop))
            self.done = base + share
        out = {}
        for k in outs[0]:
            if k == "rec":
                out[k] = {c: np.concatenate([o[k][c] for o in outs]) for c in outs[0][k]}
            else:
                out[k] = np.concatenate([o[k] for o in outs])
        return out

    def run(self):
        R = dict(policy=str(self.policy_path), created=datetime.now().strftime("%Y-%m-%d %H:%M"),
                 preset=self.preset, speeds=list(EVAL_SPEEDS),
                 settings=dict(heading_hold=self.heading_hold, obs_noise=self.obs_noise,
                               push_duration=self.push_duration, seed=self.seed, tests=self.tests,
                               trials=self.p),
                 model=model_check(self.model), mass=self.nom.total_mass, train=training_setup(),
                 nominal_friction=self.nom.friction)
        R["walk"] = self.walk()
        R["walkable"] = [i for i, s in enumerate(R["walk"]["survival"]) if s >= SURVIVE_OK]
        if "push" in self.tests:
            R["push"] = self.push(R["walkable"])
        for kind in ("friction", "payload", "motor"):
            if kind in self.tests:
                R[kind] = self.condition(kind)
        self._say("analysing", 1.0)
        R["headline"] = headline(R)
        R["findings"] = diagnose(R)
        R["watch"] = failed_cases(R)
        return R

    # ---------------------------------------------------------------- tests
    def walk(self):
        S, N = EVAL_SPEEDS, self.p["nominal"]
        rows = [dict(vx=s, t_cmd=self.rng.uniform(*CMD_ONSET), end=WALK_S, speed=i)
                for i, s in enumerate(S) for _ in range(N)]
        o = self._run("walk", [self.model], rows, STEADY_AFTER_S, record=True, label="walking at each speed")
        dt = self.model.opt.timestep
        spd = np.array([r["speed"] for r in rows])
        rec, da = o["rec"], o["da"]
        limit = self.model.actuator_forcerange[:, 1].copy()
        W = {k: [] for k in ("survival", "vx_mean", "vx_std", "vy_mean", "err", "drift", "heading", "response")}
        gait = {k[0]: [] for k in GAIT_ROWS}
        taus = {i: [] for i in range(len(S))}
        for i, s in enumerate(S):
            idx = np.flatnonzero(spd == i)
            alive = idx[o["survived"][idx]]
            W["survival"].append(float(o["survived"][idx].mean()))
            W["vx_mean"].append(_mean(o["vx"][alive]))
            W["vx_std"].append(_std(o["vx"][alive]))
            W["vy_mean"].append(_mean(o["vy"][alive]))
            W["err"].append(_mean(o["err"][alive]))
            per, drifts, heads, resp = [], [], [], []
            for j in alive:
                tc = o["t_cmd"][j]
                w0 = int(math.ceil((tc + STEADY_AFTER_S) / dt))
                p0 = int(math.ceil((tc + STEADY_AFTER_S) / STEP_DT))
                r = {k: rec[k][j, w0:] for k in REC_KEYS}
                per.append(gait_metrics(r, da[j, p0:], self.nom.total_mass, limit, dt))
                taus[i].append(np.abs(r["tau"]))
                if s != 0.0:
                    dr, hd = straightness(rec, j, tc, dt)
                    drifts.append(dr)
                    heads.append(hd)
                    resp.append(response_time(rec, j, tc, dt))
            for k in gait:
                gait[k].append(_mean([g[k] for g in per]))
            W["drift"].append(_mean(drifts))
            W["heading"].append(_mean(heads))
            W["response"].append(_mean(resp))
        W["gait"] = gait
        W["mean_err"] = _mean(W["err"])
        W["falls_pct"] = 100.0 * (1.0 - float(o["survived"].mean()))
        # torque over the speeds it can walk at: a robot staggering at a speed where most fall
        # saturates its motors, which says nothing about the gait
        use = [i for i in range(len(S)) if W["survival"][i] >= SURVIVE_OK and taus[i]] or \
              [i for i in range(len(S)) if taus[i]]
        tq = np.concatenate([x for i in use for x in taus[i]]) if use else np.zeros((1, robot.NUM_JOINTS))
        peak = np.array([MOTORS[motor_of(n)][1] for n in robot.JOINT_NAMES])
        rated = np.array([MOTORS[motor_of(n)][0] for n in robot.JOINT_NAMES])
        W["torque"] = dict(p95=np.percentile(tq, 95, axis=0).tolist(), peak=tq.max(axis=0).tolist(),
                           p95_pct=(100 * np.percentile(tq, 95, axis=0) / peak).tolist(),
                           peak_pct=(100 * tq.max(axis=0) / peak).tolist(),
                           rated_pct=(100 * rated / peak).tolist(),
                           at_limit_pct=100.0 * float(np.mean(tq >= 0.98 * limit)))
        return W

    def push(self, walkable):
        S, F, D, rep = EVAL_SPEEDS, self.p["push_forces"], PUSH_DIRS_DEG, self.p["push_repeats"]
        dur = self.push_duration
        rows = []
        for i, s in enumerate(S):
            for fi, f in enumerate(F):
                for di, a in enumerate(D):
                    for _ in range(rep):
                        tc = self.rng.uniform(*CMD_ONSET)
                        pt = tc + PUSH_AFTER_S + self.rng.uniform(0.0, GAIT_PERIOD)
                        rows.append(dict(vx=s, t_cmd=tc, push_t=pt, push_dur=dur, push_force=float(f),
                                         push_angle=math.radians(a), end=pt + dur + PUSH_WATCH_S,
                                         cell=(i, fi, di)))
        o = self._run("push", [self.model], rows, STEADY_AFTER_S, label="pushes while walking")
        tot = np.zeros((len(S), len(F), len(D)))
        ok = np.zeros_like(tot)
        for r, sv in zip(rows, o["survived"]):
            tot[r["cell"]] += 1
            ok[r["cell"]] += sv
        cell = ok / np.maximum(tot, 1)
        wk = list(walkable)
        return dict(forces=list(F), duration=dur, dirs=list(D),
                    survival=cell.mean(axis=2).T.tolist(),                       # [force][speed]
                    cells=cell.tolist(),                                          # [speed][force][dir]
                    dir_survival=(cell[wk].mean(axis=0).T.tolist() if wk else None))   # [dir][force]

    def condition(self, kind):
        levels = {"friction": EVAL_FRICTIONS, "payload": EVAL_PAYLOADS, "motor": EVAL_STRENGTHS}[kind]
        attr = {"friction": "friction", "payload": "payload", "motor": "strength"}[kind]
        models = []
        for lv in levels:
            mm = copy.copy(self.model)
            apply_conditions(mm, self.nom, self.nom.conditions(**{attr: lv}))
            models.append(mm)
        S, N = EVAL_SPEEDS, self.p["cond"]
        rows = [dict(model=li, vx=s, t_cmd=self.rng.uniform(*CMD_ONSET), end=COND_S, cell=(li, i))
                for li in range(len(levels)) for i, s in enumerate(S) for _ in range(N)]
        o = self._run(kind, models, rows, STEADY_AFTER_S, label=f"{TESTS[kind][0].lower()}")
        surv = np.zeros((len(levels), len(S)))
        err = np.full((len(levels), len(S)), np.nan)
        for li in range(len(levels)):
            for i in range(len(S)):
                idx = [k for k, r in enumerate(rows) if r["cell"] == (li, i)]
                sv = o["survived"][idx]
                surv[li, i] = sv.mean()
                err[li, i] = _mean(o["err"][idx][sv])
        return dict(levels=list(levels), survival=surv.tolist(), err=err.tolist())


def training_setup():
    """The training settings the diagnosis refers to, read from oc1_rl.env where possible."""
    cfg = EnvCfg()
    stages = [dict(s) for s in getattr(cfg, "velocity_stages", [])]
    vx = [list(s["lin_vel_x"]) for s in stages if "lin_vel_x" in s]
    return dict(stages=[{k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()} for s in stages],
                vx_first=vx[0] if vx else None, vx_last=vx[-1] if vx else None,
                wide_step=stages[1]["step"] if len(stages) > 1 else None,
                push_vel=[float(v) for v in getattr(envmod, "PUSH_VEL", [0.5, 0.5])[:2]],
                push_interval=list(getattr(envmod, "PUSH_INTERVAL_S", (5.0, 6.0))),
                rel_heading_envs=getattr(cfg, "rel_heading_envs", None),
                rel_standing_envs=getattr(cfg, "rel_standing_envs", None),
                resampling=list(getattr(cfg, "resampling_time_range", (3.0, 8.0))),
                weights=dict(getattr(envmod, "REWARD_WEIGHTS", {})))


# ====================================================================== diagnosis
def _walkable_label(R):
    S = R["speeds"]
    wk = R["walkable"]
    return f"speeds {S[wk[0]]:g}…{S[wk[-1]]:g}" if wk else "no speed"


def _threshold(levels, surv, walkable, order):
    """Last level, going from mildest to harshest in `order`, where every walkable speed passes."""
    best = None
    for li in order:
        if walkable and all(surv[li][i] >= SURVIVE_OK for i in walkable):
            best = levels[li]
        else:
            break
    return best


def push_threshold(R):
    P = R.get("push")
    if not P or not R["walkable"]:
        return None
    return _threshold(P["forces"], P["survival"], R["walkable"], range(len(P["forces"])))


def cond_threshold(R, kind):
    X = R.get(kind)
    if not X or not R["walkable"]:
        return None
    lv = X["levels"]
    if kind == "friction":
        nominal = R.get("nominal_friction", 0.6)
        order = sorted([i for i, v in enumerate(lv) if v <= nominal + 1e-9], key=lambda i: -lv[i])
    elif kind == "payload":
        order = sorted(range(len(lv)), key=lambda i: lv[i])
    else:
        order = sorted(range(len(lv)), key=lambda i: -lv[i])
    return _threshold(lv, X["survival"], R["walkable"], order)


def headline(R):
    S, W = R["speeds"], R["walk"]
    out = []
    ok = [s for s, sv, e in zip(S, W["survival"], W["err"]) if sv >= 1.0 and np.isfinite(e) and e <= TRACK_OK]
    moving = [s for s in ok if s != 0]
    out.append((f"Commands followed within {TRACK_OK} m/s (no falls)",
                ", ".join(f"{s:g}" for s in moving) + " m/s" if moving
                else ("none while moving (only standing)" if 0.0 in ok else "none")))
    fw = [v for s, v in zip(S, W["vx_mean"]) if s > 0 and np.isfinite(v)]
    bw = [v for s, v in zip(S, W["vx_mean"]) if s < 0 and np.isfinite(v)]
    out.append(("Fastest speed actually reached",
                f"forward {max(fw) if fw else 0:.2f} m/s, backward {-min(bw) if bw else 0:.2f} m/s"))
    out.append(("Falls in the tracking test", f"{W['falls_pct']:.1f}%"))
    out.append(("Sideways drift (moving speeds)", f"{_mean(W['drift']):.1f} cm per m"))
    out.append(("Median response time", f"{np.nanmedian(W['response']) if np.isfinite(W['response']).any() else float('nan'):.2f} s"))
    tq = W["torque"]
    j = int(np.argmax(tq["peak_pct"]))
    out.append(("Highest joint torque", f"{short_joint(robot.JOINT_NAMES[j])}: peak {tq['peak_pct'][j]:.0f}% of limit"))
    wl = _walkable_label(R)
    if "push" in R:
        f = push_threshold(R)
        d = R["push"]["duration"]
        out.append((f"Largest push survived ≥90% at every speed ({wl})",
                    f"{f:.0f} N for {d:g} s ({f * d:.0f} N·s)" if f else "none of the tested pushes"))
    if "friction" in R:
        f = cond_threshold(R, "friction")
        out.append((f"Lowest friction with ≥90% survival ({wl})", f"{f:g}" if f else "none tested"))
    if "payload" in R:
        f = cond_threshold(R, "payload")
        out.append((f"Largest payload with ≥90% survival ({wl})", f"{f:g} kg" if f is not None else "none"))
    if "motor" in R:
        f = cond_threshold(R, "motor")
        out.append((f"Weakest motors with ≥90% survival ({wl})", f"{f:.0%}" if f else "none tested"))
    return out


def diagnose(R):
    """Turn the numbers into findings: what failed, why it probably failed, what to change."""
    S, W, M, TR = np.array(R["speeds"]), R["walk"], R["model"], R["train"]
    wk = R["walkable"]
    surv = np.array(W["survival"])
    F = []
    mass = R["mass"]
    wts = TR.get("weights", {})

    def add(sev, test, title, detail, fixes, watch=None):
        F.append(dict(severity=sev, test=test, title=title, detail=detail, fixes=fixes, watch=watch))

    def watch(vx, **kw):
        w = dict(vx=float(vx), friction=R.get("nominal_friction", 0.6), payload=0.0, strength=1.0,
                 heading_hold=R["settings"]["heading_hold"], push=None)
        w.update(kw)
        return w

    knee_deg = math.degrees(M["knee_flex_max"])
    knee_small = M["knee_flex_max"] < 0.9
    knee_fix = (f"The knees can only bend {knee_deg:.0f}° (joint limit {M['knee_flex_max']:.2f} rad, and the home "
                f"pose already uses {math.degrees(M['knee_home']):.0f}°). Walking normally uses 50–65° of knee bend "
                "in swing: set the URDF joint limits to the real mechanical range and retrain.") if knee_small else None
    leg_l, leg_r = M["leg_mass_left"], M["leg_mass_right"]
    asym_mass = abs(leg_l - leg_r) / max(leg_l, leg_r) > 0.05
    mass_fix = (f"The model is not left/right symmetric: left leg {leg_l:.2f} kg, right leg {leg_r:.2f} kg, "
                f"whole-robot CoM {100 * M['com_y']:+.1f} cm sideways. Compare the URDF link masses (shank lc vs rc "
                "especially) with the real robot; a real imbalance is fine, a CAD-export error is not.") if asym_mass else None

    # ---- model
    if knee_small:
        add("warn", "model", f"Knee range is only {knee_deg:.0f}°", knee_fix,
            ["This caps step length (top speed) and foot clearance for any policy, trained or not.",
             "Also check hip pitch: the range is "
             f"{M['hip_pitch_range'][0]:+.2f}…{M['hip_pitch_range'][1]:+.2f} rad."])
    if asym_mass:
        add("warn", "model", f"Legs differ by {abs(leg_l - leg_r):.2f} kg", mass_fix,
            ["Asymmetric mass shows up as heading drift, left/right torque differences and limping.",
             "If the real robot really is like this, train with it; a mirror (symmetry) loss in PPO "
             "then must not be used."])

    # ---- standing and falls without disturbance
    i0 = int(np.argmin(np.abs(S)))
    if surv[i0] < 1.0:
        add("fail", "walk", "Falls while standing still",
            f"{(1 - surv[i0]):.0%} of the robots fell with a zero command.",
            ["Fix standing before anything else: check the home pose, base height and the PD gains.",
             "Raise rel_standing_envs (now "
             f"{TR.get('rel_standing_envs')}) so more training robots practise standing."], watch(0.0))
    bad = [i for i in range(len(S)) if surv[i] < SURVIVE_OK and S[i] != 0]
    if bad:
        worst = min(bad, key=lambda i: surv[i])
        lo1, hi1 = TR.get("vx_first") or (None, None)
        lo2, hi2 = TR.get("vx_last") or (None, None)
        step = TR.get("wide_step")
        outside = [S[i] for i in bad if lo1 is not None and not (lo1 <= S[i] <= hi1)]
        fixes = []
        if outside and step:
            fixes.append(f"Check that training reached the wide command stage: EnvCfg.velocity_stages starts at "
                         f"vx {lo1:g}…{hi1:g} m/s and only widens to {lo2:g}…{hi2:g} once common_step > {step} "
                         f"(= {step / 24:.0f} PPO iterations of 24 steps). train.py defaults to 3000 iterations, so "
                         f"a default run has never been asked for {', '.join(f'{s:g}' for s in outside)} m/s. "
                         "Train longer (--resume latest --iterations 8000) or move the stage earlier.")
        fixes.append("Until then clip commands to what the policy was trained on (CMD_LIMITS here, in play.py "
                     "and on the robot).")
        if knee_small:
            fixes.append(f"Stride is capped by the model: with the fixed {GAIT_PERIOD:g} s gait clock, speed = step "
                         f"length × {2 / GAIT_PERIOD:.1f} steps/s, and the {knee_deg:.0f}° knee limits step length.")
        add("fail", "walk", "Falls when walking at " + ", ".join(f"{S[i]:g}" for i in bad) + " m/s",
            "Undisturbed, on the nominal floor: " + ", ".join(f"{S[i]:g} m/s {surv[i]:.0%} survive" for i in bad)
            + ". Every other test at these speeds fails for this reason, not because of the disturbance.",
            fixes, watch(S[worst]))

    # ---- tracking
    err = np.array(W["err"], dtype=float)
    vxm = np.array(W["vx_mean"], dtype=float)
    poor = [i for i in wk if S[i] != 0 and np.isfinite(err[i]) and err[i] > TRACK_OK]
    if poor:
        wi = max(poor, key=lambda i: err[i])
        sev = "fail" if err[wi] > 0.3 else "warn"
        fixes = [f"Make tracking worth more: track_linear_velocity (weight {wts.get('track_linear_velocity', 1.0)}) "
                 "uses exp(-err²/0.25), which still pays 37% for a 0.5 m/s error, so walking slower than asked "
                 "costs little next to pose, foot_gait and action_rate. Raise the weight to ~2 or narrow the "
                 "kernel (0.25 → 0.1) once it walks without falling.",
                 f"Loosen what competes with speed: pose (weight {wts.get('pose', 1.0)}) pulls joints back to the "
                 "home pose; action_rate_l2 and joint_acc_l2 punish the large fast swings that speed needs.",
                 "Train longer: the tracking error usually keeps falling well after the falls stop."]
        if knee_small:
            fixes.append(knee_fix)
        add(sev, "walk", f"Speed tracking off by up to {err[wi]:.2f} m/s",
            "Commanded → actual: " + ", ".join(f"{S[i]:g} → {vxm[i]:.2f}" for i in poor) + " m/s.",
            fixes, watch(S[wi]))

    # ---- straightness
    drift = np.array(W["drift"], dtype=float)
    mv = [i for i in wk if S[i] != 0 and np.isfinite(drift[i])]
    if mv and max(drift[i] for i in mv) > 10.0:
        wi = max(mv, key=lambda i: drift[i])
        head = np.array(W["heading"], dtype=float)
        vym = np.array(W.get("vy_mean", [np.nan] * len(S)), dtype=float)
        crab = np.where(np.abs(vxm) > 0.05, 100 * np.abs(vym) / np.maximum(np.abs(vxm), 1e-6), np.nan)
        side = "right" if _mean(vym[mv]) < 0 else "left"
        fixes = []
        if not R["settings"]["heading_hold"]:
            fixes.append(f"Turning part: deploy with heading hold (play.py and this lab do it: yaw-rate from the IMU "
                         f"yaw error). Training used rel_heading_envs = {TR.get('rel_heading_envs')}, i.e. always "
                         "with a heading controller closing the loop, so the policy never had to hold yaw by "
                         "itself. Tick 'heading hold' and re-run to see what is left.")
        fixes.append("Crabbing part (sideways body velocity with a zero vy command): heading hold cannot remove it. "
                     "Track vy harder (the same exp(-err²/0.25) kernel covers vx and vy) and, on the real robot, "
                     "close an outer position/odometry loop on vy if it must walk straight.")
        if mass_fix:
            fixes.append(mass_fix)
        fixes.append("To make the policy itself hold heading: train with rel_heading_envs ≈ 0.5 so it also follows "
                     "pure yaw-rate commands, and tighten track_angular_velocity (exp(-err²/0.5) → /0.25).")
        add("fail" if drift[wi] > 30 else "warn", "walk", f"Walks in a curve: up to {drift[wi]:.0f} cm sideways per m",
            "Sideways drift per metre walked: " + ", ".join(f"{S[i]:g} m/s {drift[i]:.0f} cm" for i in mv)
            + ". Two causes are mixed in this: turning (heading drift "
            + ", ".join(f"{head[i]:.0f}°" for i in mv if np.isfinite(head[i])) + " per 10 m) and crabbing to its "
            + f"{side} (body vy alone ≈ " + ", ".join(f"{crab[i]:.0f}" for i in mv if np.isfinite(crab[i]))
            + " cm per m)." + (" No heading hold in this run." if not R["settings"]["heading_hold"] else ""),
            fixes, watch(S[wi]))

    # ---- response
    rs = np.array(W["response"], dtype=float)
    if np.isfinite(rs).any() and np.nanmedian(rs) > 0.8:
        add("warn", "walk", f"Slow to start walking: median {np.nanmedian(rs):.2f} s to 90% speed",
            "Time from the command step until 90% of the steady speed: "
            + ", ".join(f"{S[i]:g} m/s {rs[i]:.2f} s" for i in range(len(S)) if np.isfinite(rs[i])) + ".",
            [f"Commands change only every {TR['resampling'][0]:g}–{TR['resampling'][1]:g} s and mostly smoothly, "
             "so a fast reaction is rarely rewarded: shorten resampling_time_range to (1, 4) and include "
             "standing ↔ walking steps.",
             "The gait clock is zeroed while standing and resumes from episode time, so the first step can wait "
             f"up to {GAIT_PERIOD / 2:g} s for the clock: reset the clock phase when the command switches on.",
             "action_rate_l2 / joint_acc_l2 slow the start: reduce them once the gait is stable."])

    # ---- torque
    tq = W["torque"]
    j = int(np.argmax(tq["peak_pct"]))
    hot = [k for k in range(robot.NUM_JOINTS) if tq["p95_pct"][k] > tq["rated_pct"][k]]
    if tq["peak_pct"][j] > 75 or tq["at_limit_pct"] > 0.5 or hot:
        sev = "fail" if (tq["peak_pct"][j] > 95 or tq["at_limit_pct"] > 1.0) else "warn"
        n = robot.JOINT_NAMES[j]
        detail = (f"{short_joint(n)} peaks at {tq['peak'][j]:.0f} N·m = {tq['peak_pct'][j]:.0f}% of the "
                  f"{motor_of(n)} peak; time at the limit {tq['at_limit_pct']:.1f}%.")
        if hot:
            detail += (" Typical torque (95th pct) is above the rated, continuous value on: "
                       + ", ".join(f"{short_joint(robot.JOINT_NAMES[k])} {tq['p95'][k]:.0f} N·m "
                                   f"(rated {MOTORS[motor_of(robot.JOINT_NAMES[k])][0]:g})" for k in hot) + ".")
        add(sev, "walk", f"Joint torque: {short_joint(n)} reaches {tq['peak_pct'][j]:.0f}% of its motor", detail,
            ["Above the rated torque the motor heats up: in sustained walking it will hit its thermal limit. Add a "
             "torque penalty, e.g. -1e-4·Σ(τ/τ_rated)², or penalize only τ above τ_rated.",
             "Train with the torque limit at ~80% of the real peak (robot._ACTUATOR_GROUPS effort) so the policy "
             "keeps a reserve for pushes and model error.",
             "Hardware option: the ankle pitch carries most of the balance load; a bigger motor or gear ratio there "
             "buys margin."])

    # ---- gait quality
    g = W["gait"]
    mvw = [i for i in wk if S[i] != 0]
    if mvw:
        cl = [g["clearance"][i] for i in mvw]
        if np.nanmin(cl) < 5.0:
            fixes = []
            if knee_small:
                fixes.append(knee_fix)
            fixes += [f"foot_clearance (weight {wts.get('foot_clearance', -1.0)}, target 0.10 m) is weighted by foot "
                      "speed and is easy to ignore: raise it (−2…−5) or use a swing-height reward at mid-swing.",
                      "Low clearance trips on carpet edges, cables and small steps on the real robot."]
            add("warn", "walk", f"Low foot clearance: {np.nanmin(cl):.1f}–{np.nanmax(cl):.1f} cm (want 5–10)",
                "Mean peak swing height above the stance height, per speed: "
                + ", ".join(f"{S[i]:g} m/s {g['clearance'][i]:.1f} cm" for i in mvw) + ".", fixes)
        sl_ = [g["slip"][i] for i in mvw]
        if np.nanmax(sl_) > 0.05:
            add("warn", "walk", f"Feet slide while loaded: up to {np.nanmax(sl_):.3f} m/s",
                "Horizontal foot speed while in contact: "
                + ", ".join(f"{S[i]:g} m/s {g['slip'][i]:.3f}" for i in mvw) + ".",
                [f"Raise foot_slip (weight {wts.get('foot_slip', -0.25)}) once walking is stable.",
                 "Slip on the sim floor means far worse slip on a real one (see the friction test)."])
        asy = [g["asym"][i] for i in mvw]
        if np.nanmax(asy) > 5.0:
            add("warn", "walk", f"Left/right asymmetry up to {np.nanmax(asy):.0f}%",
                "Difference in stance time between the legs: "
                + ", ".join(f"{S[i]:g} m/s {g['asym'][i]:.1f}%" for i in mvw) + ".",
                [x for x in (mass_fix,) if x] + ["If the robot is symmetric, add a mirror-symmetry loss or "
                                                 "mirrored data augmentation to PPO."])
    if 0.0 in R["speeds"]:
        st = g["steps"][R["speeds"].index(0.0)]
        if np.isfinite(st) and st > 0.2:
            add("warn", "walk", f"Shuffles while standing: {st:.2f} steps/s",
                "With a zero command the feet should stay down.",
                [f"stand_still (weight {wts.get('stand_still', -1.0)}) only penalizes joint deviation; add a reward "
                 "for both feet in contact when |cmd| < 0.1.",
                 f"Raise rel_standing_envs ({TR.get('rel_standing_envs')}) to 0.1–0.2."])

    # ---- push
    if "push" in R and wk:
        P = R["push"]
        fstar = push_threshold(R)
        dur = P["duration"]
        dv = max(TR["push_vel"])
        train_imp = mass * dv
        imp = (fstar or 0.0) * dur
        forces = P["forces"]
        first_fail = next((fi for fi in range(len(forces))
                           if any(P["survival"][fi][i] < SURVIVE_OK for i in wk)), None)
        if first_fail is not None:
            sev = "fail" if imp < train_imp else "warn"
            # direction weakness averaged over every tested force (one force level alone is noisy)
            ds = np.array(P["dir_survival"], dtype=float).mean(axis=1) if P["dir_survival"] else None
            detail = ((f"At every walkable speed ≥90% survive only up to {fstar:.0f} N for {dur:g} s ({imp:.0f} N·s)."
                       if fstar else f"Even the smallest push ({forces[0]:.0f} N for {dur:g} s) drops some walkable "
                                     "speed below 90% survival.")
                      + f" Training kicks are |Δv| ≤ {dv:g} m/s per axis ≈ {train_imp:.0f} N·s on this "
                      f"{mass:.1f} kg robot (≈{train_imp / dur:.0f} N for {dur:g} s), so larger pushes are outside "
                      "what it practised.")
            fixes = [f"Push curriculum: raise PUSH_VEL from {dv:g} m/s towards 1.0–1.5 m/s as the training fall "
                     "rate drops below a few %.",
                     f"Train with force pushes too (xfrc_applied for 0.1–0.3 s, exactly what this test does), not only "
                     f"instant velocity kicks; push every 2–4 s instead of {TR['push_interval'][0]:g}–"
                     f"{TR['push_interval'][1]:g} s while learning recovery."]
            if ds is not None:
                order = np.argsort(ds)
                weak = [PUSH_DIR_NAMES[k] for k in order[:3]]
                detail += (f" Averaged over all forces it is weakest when pushed "
                           f"{', '.join(f'{PUSH_DIR_NAMES[k]} ({ds[k]:.0%} survive)' for k in order[:3])}, and "
                           f"strongest when pushed {PUSH_DIR_NAMES[order[-1]]} ({ds[order[-1]]:.0%}).")
                if any(d in ("left", "right", "back-left", "back-right") for d in weak):
                    fixes.append("Sideways pushes need a sideways step: check the hip roll range (±0.5 rad) and that "
                                 "the policy is allowed to widen its stance; reward foot placement after a push.")
            tqp = W["torque"]["peak_pct"]
            ank = max(tqp[robot.JOINT_NAMES.index("left_ankle_pitch")], tqp[robot.JOINT_NAMES.index("right_ankle_pitch")])
            if ank > 60:
                fixes.append(f"The ankles already reach {ank:.0f}% of their peak torque while walking: little is left "
                             "for an ankle strategy, so recovery has to come from stepping.")
            # watch: walkable speed that fails first, in its weakest direction
            wi = min(wk, key=lambda i: P["survival"][first_fail][i])
            di = int(np.argmin(P["cells"][wi][first_fail]))
            title = (f"Push recovery safe only up to {fstar:.0f} N ({imp:.0f} N·s)" if fstar
                     else f"Push recovery: under 90% already at the smallest push ({forces[0]:.0f} N)")
            add(sev, "push", title, detail, fixes,
                watch(S[wi], push=dict(force=float(forces[first_fail]), angle=float(P["dirs"][di]), dur=dur)))
        else:
            add("ok", "push", f"Survives every tested push at walkable speeds (up to {forces[-1]:.0f} N)",
                "Raise the force range to find the limit.", [])

    # ---- friction / payload / motor
    for kind in ("friction", "payload", "motor"):
        if kind not in R or not wk:
            continue
        X = R[kind]
        lv = X["levels"]
        thr = cond_threshold(R, kind)
        fails = [(li, i) for li in range(len(lv)) for i in wk if X["survival"][li][i] < SURVIVE_OK]
        if kind == "friction":
            nominal = R.get("nominal_friction", 0.6)
            high = [(li, i) for li, i in fails if lv[li] > nominal]
            limit_bad = thr is None or thr > 0.3
            if limit_bad:
                li, i = max(((li, i) for li, i in fails if lv[li] <= nominal), key=lambda x: (lv[x[0]], -x[1]),
                            default=(0, wk[0]))
                add("fail" if (thr is None or thr > 0.4) else "warn", "friction",
                    f"Slips and falls below μ = {thr if thr else nominal:g}",
                    "Survival at walkable speeds drops below 90% for μ < " + f"{thr if thr else nominal:g}.",
                    ["Widen the friction randomization in env._make_models (now 0.3–1.6) down to 0.15–0.2.",
                     f"Raise foot_slip (weight {wts.get('foot_slip', -0.25)}); slippery floors need shorter, "
                     "flatter steps.",
                     "Typical real floors: rubber on tiles or concrete μ ≈ 0.4–0.7, dusty or wet 0.2–0.3."],
                    watch(S[i], friction=float(lv[li])))
            elif high:
                li, i = high[0]
                add("warn", "friction", f"Falls on high-friction floors (μ {lv[li]:g})",
                    "Sticky floors catch the swing foot: usually a foot-clearance problem.",
                    [knee_fix or "Raise foot clearance (see the gait findings)."], watch(S[i], friction=float(lv[li])))
            else:
                add("ok", "friction", f"Walks on every tested floor down to μ = {thr:g}", "", [])
        elif kind == "payload":
            if thr is None or thr < 3.0:
                li, i = min(fails, key=lambda x: (lv[x[0]], x[1])) if fails else (0, wk[0])
                add("fail" if (thr is None or thr < 1.0) else "warn", "payload",
                    f"Tolerates only {thr or 0:g} kg on the torso ({100 * (thr or 0) / mass:.0f}% of its mass)",
                    "A battery or computer upgrade, or a CAD mass error, is enough to make it fall.",
                    ["Randomize the torso mass in env._make_models (e.g. +0…5 kg or ±20%) next to the CoM offset it "
                     "already randomizes.",
                     "More mass means more knee and ankle torque: set a payload under Test conditions and watch the "
                     "torque bars."], watch(S[i], payload=float(lv[li])))
            else:
                add("ok", "payload", f"Carries {thr:g} kg at every walkable speed", "", [])
        else:
            if thr is None or thr > 0.7:
                li, i = max(fails, key=lambda x: (lv[x[0]], -x[1])) if fails else (0, wk[0])
                add("fail" if (thr is None or thr > 0.8) else "warn", "motor",
                    f"Needs {thr or 1:.0%} of the motor peak torque",
                    "Real actuators lose torque when hot, at low battery voltage or through gearbox friction.",
                    ["Randomize actuator strength per env: scale actuator_forcerange 0.7–1.0 and Kp/Kd ±20% in "
                     "env._make_models.",
                     "Add a torque penalty so the gait keeps a margin (see the torque finding)."],
                    watch(S[i], strength=float(lv[li])))
            else:
                add("ok", "motor", f"Still walks with {thr:.0%} of the motor torque", "", [])

    rank = {"fail": 0, "warn": 1, "ok": 2}
    F.sort(key=lambda f: rank[f["severity"]])
    return F


def failed_cases(R):
    """Mildest failing case per test and speed, for 'Watch in lab'."""
    S, W, wk = R["speeds"], R["walk"], R["walkable"]
    nf = R.get("nominal_friction", 0.6)
    hh = R["settings"]["heading_hold"]
    base = dict(friction=nf, payload=0.0, strength=1.0, heading_hold=hh, push=None)
    out = []
    for i, sv in enumerate(W["survival"]):
        if sv < SURVIVE_OK:
            out.append((f"walk {S[i]:+g} m/s: {sv:.0%} survive", dict(base, vx=S[i])))
    if "push" in R:
        P = R["push"]
        for i in wk:
            fi = next((fi for fi in range(len(P["forces"])) if P["survival"][fi][i] < SURVIVE_OK), None)
            if fi is None:
                continue
            di = int(np.argmin(P["cells"][i][fi]))
            out.append((f"push {P['forces'][fi]:.0f} N, pushed {PUSH_DIR_NAMES[di]}, at {S[i]:+g} m/s: "
                        f"{P['cells'][i][fi][di]:.0%} survive ({P['survival'][fi][i]:.0%} all directions)",
                        dict(base, vx=S[i], push=dict(force=float(P["forces"][fi]), angle=float(P["dirs"][di]),
                                                      dur=P["duration"]))))
    for kind, attr, unit in (("friction", "friction", "μ "), ("payload", "payload", "+"), ("motor", "strength", "")):
        if kind not in R:
            continue
        X = R[kind]
        lv = X["levels"]
        if kind == "friction":
            order = sorted(range(len(lv)), key=lambda li: abs(lv[li] - nf))
        elif kind == "payload":
            order = sorted(range(len(lv)), key=lambda li: lv[li])
        else:
            order = sorted(range(len(lv)), key=lambda li: -lv[li])
        for i in wk:
            li = next((li for li in order if X["survival"][li][i] < SURVIVE_OK), None)
            if li is None:
                continue
            v = lv[li]
            txt = f"μ {v:g}" if kind == "friction" else (f"+{v:g} kg" if kind == "payload" else f"motors {v:.0%}")
            out.append((f"{txt} at {S[i]:+g} m/s: {X['survival'][li][i]:.0%} survive", dict(base, vx=S[i], **{attr: v})))
    return out


# ====================================================================== report
def _clean(x):
    """JSON-able copy: numpy -> python, NaN/inf -> None."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def save_results(R, out_dir=None):
    out_dir = Path(out_dir or ROOT / "eval_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = Path(R["policy"])
    stem = f"{p.parent.name}_{p.stem}" if p.parent.name else p.stem
    name = f"{stem}_{datetime.now():%Y%m%d_%H%M%S}"
    js = out_dir / f"{name}.json"
    js.write_text(json.dumps(_clean(R), indent=1))
    png = out_dir / f"{name}.png"
    try:
        save_report(R, png)
    except ImportError:
        png = None
    return png, js


def save_report(R, path):
    from matplotlib.figure import Figure

    S = R["speeds"]
    xs = np.arange(len(S))
    W = R["walk"]
    wk = set(R["walkable"])
    sl = [f"{s:g}" for s in S]
    fig = Figure(figsize=(20, 28), dpi=90, facecolor=C["surface"])
    gs = fig.add_gridspec(5, 3, height_ratios=[1.0, 1.15, 1.15, 1.25, 0.85], hspace=0.38, wspace=0.27,
                          left=0.05, right=0.985, top=0.955, bottom=0.015)
    p = Path(R["policy"])
    st = R["settings"]
    fig.text(0.012, 0.985, f"Walking policy report — {p.parent.name or p.stem}", fontsize=21, fontweight="bold",
             color=C["ink"])
    fig.text(0.012, 0.973, "  ·  ".join([
        str(p), f"{R['preset']} preset", "sensor noise" if st["obs_noise"] else "clean conditions (no noise)",
        "heading hold" if st["heading_hold"] else "no heading hold", R["created"]]), fontsize=11, color=C["muted"])

    def style(ax, title):
        ax.set_facecolor(C["surface"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(C["grid"])
        ax.tick_params(colors=C["ink2"], labelsize=10)
        ax.grid(True, color=C["grid"], lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=C["ink"], pad=10)

    def speed_ticks(ax, axis="x", positions=None):
        positions = xs if positions is None else positions
        getattr(ax, f"set_{axis}ticks")(positions, [sl[int(i)] for i in range(len(S))][:len(positions)])
        for lab, i in zip(getattr(ax, f"get_{axis}ticklabels")(), range(len(S))):
            if i not in wk:
                lab.set_color(C["crit"])

    def mark_ticks(ax, idx):
        for lab, i in zip(ax.get_xticklabels(), idx):
            if i not in wk:
                lab.set_color(C["crit"])

    def not_run(ax, title):
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=C["muted"])
        ax.text(0.5, 0.5, "not run", ha="center", va="center", color=C["muted"], fontsize=12, transform=ax.transAxes)

    def heat(ax, grid, ylabels, title, ylabel, err=None, xl="forward speed command vx (m/s)", xlabels=None):
        g = np.array(grid, dtype=float)
        ax.imshow(g, cmap="Blues", vmin=0, vmax=1, origin="lower", aspect="auto")
        for (r, c), v in np.ndenumerate(g):
            txt = f"{v * 100:.0f}%" if np.isfinite(v) else "–"
            dark = np.isfinite(v) and v > 0.55
            ax.text(c, r + (0.13 if err is not None else 0), txt, ha="center", va="center", fontsize=9.5,
                    fontweight="bold", color="white" if dark else C["ink"])
            if err is not None and err[r][c] is not None and np.isfinite(err[r][c]):
                ax.text(c, r - 0.22, f"err {err[r][c]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if dark else C["ink2"])
        ax.set_xticks(np.arange(-0.5, g.shape[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, g.shape[0]), minor=True)
        ax.grid(which="minor", color=C["surface"], lw=2.5)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(colors=C["ink2"], labelsize=10, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_yticks(range(len(ylabels)), ylabels)
        if xlabels is None:
            speed_ticks(ax)
        else:
            ax.set_xticks(range(len(xlabels)), xlabels)
        ax.set_xlabel(xl, color=C["ink2"], fontsize=10.5)
        ax.set_ylabel(ylabel, color=C["ink2"], fontsize=10.5)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=C["ink"], pad=10)

    # row 0 ------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    style(ax, f"Tracking vx: mean error {W['mean_err']:.2f} m/s")
    ax.plot(S, S, "--", color=C["muted"], lw=1.3, label="ideal (actual = command)")
    vm = np.array(W["vx_mean"], dtype=float)
    ax.errorbar(S, vm, yerr=np.nan_to_num(np.array(W["vx_std"], dtype=float)), color=C["left"], marker="o",
                ms=8, lw=2, capsize=3, label="actual (mean ± spread)")
    for s, v, sv in zip(S, vm, W["survival"]):
        if sv < 1.0:
            ax.annotate(f"× {1 - sv:.0%} fell", (s, v if np.isfinite(v) else 0), textcoords="offset points",
                        xytext=(0, -18), ha="center", color=C["crit"], fontsize=10)
    ax.set_xlabel("command vx (m/s)", color=C["ink2"])
    ax.set_ylabel("actual vx (m/s)", color=C["ink2"])
    ax.legend(frameon=False, fontsize=10, loc="upper left")

    mv = [i for i, s in enumerate(S) if s != 0]
    ax = fig.add_subplot(gs[0, 1])
    style(ax, "Straightness (sideways drift; label = heading drift)")
    dr = [W["drift"][i] for i in mv]
    ax.bar(range(len(mv)), np.nan_to_num(np.array(dr, dtype=float)), width=0.6, color=C["left"])
    for k, i in enumerate(mv):
        if np.isfinite(W["drift"][i]):
            ax.text(k, W["drift"][i], f"{W['heading'][i]:.0f}°/10 m", ha="center", va="bottom", fontsize=9,
                    color=C["ink2"])
    ax.set_xticks(range(len(mv)), [sl[i] for i in mv])
    mark_ticks(ax, mv)
    ax.set_xlabel("command vx (m/s)", color=C["ink2"])
    ax.set_ylabel("sideways drift (cm per m walked)", color=C["ink2"])

    ax = fig.add_subplot(gs[0, 2])
    style(ax, "Response time (standing → walking)")
    rs = [W["response"][i] for i in mv]
    ax.bar(range(len(mv)), np.nan_to_num(np.array(rs, dtype=float)), width=0.6, color=C["left"])
    for k, v in enumerate(rs):
        if np.isfinite(v):
            ax.text(k, v, f"{v:.2f} s", ha="center", va="bottom", fontsize=9, color=C["ink2"])
    ax.set_xticks(range(len(mv)), [sl[i] for i in mv])
    mark_ticks(ax, mv)
    ax.set_xlabel("command vx (m/s)", color=C["ink2"])
    ax.set_ylabel("time to 90% of its steady speed (s)", color=C["ink2"])

    # row 1 ------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    if "push" in R:
        P = R["push"]
        heat(ax, P["survival"], [f"{f:g}" for f in P["forces"]],
             f"Push while walking ({P['duration']:g} s, 8 dirs) — % survived", "push force on torso (N)")
    else:
        not_run(ax, "Push while walking")
    for col, kind, title, ylab, fmt in ((1, "friction", "Floor friction — % survived", "foot friction μ", "{:g}"),
                                        (2, "payload", "Payload — % survived", "extra mass on torso (kg)", "{:g}")):
        ax = fig.add_subplot(gs[1, col])
        if kind in R:
            X = R[kind]
            order = np.argsort(X["levels"])
            heat(ax, [X["survival"][i] for i in order], [fmt.format(X["levels"][i]) for i in order], title, ylab,
                 err=[X["err"][i] for i in order])
        else:
            not_run(ax, title)

    # row 2 ------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    if "motor" in R:
        X = R["motor"]
        order = np.argsort(X["levels"])
        heat(ax, [X["survival"][i] for i in order], [f"{X['levels'][i]:.0%}" for i in order],
             "Motor strength — % survived", "motor torque limit (% of real)", err=[X["err"][i] for i in order])
    else:
        not_run(ax, "Motor strength")

    ax = fig.add_subplot(gs[2, 1])
    tq = W["torque"]
    style(ax, f"Joint torque, walkable speeds (at limit {tq['at_limit_pct']:.1f}% of the time)")
    names = list(robot.JOINT_NAMES)
    y = np.arange(len(names))[::-1]
    ax.barh(y, tq["p95_pct"], height=0.55, color=C["left"], label="typical (95th pct)")
    ax.scatter(tq["peak_pct"], y, s=55, color=C["right"], zorder=3, label="peak")
    for yy, rp in zip(y, tq["rated_pct"]):
        ax.plot([rp, rp], [yy - 0.38, yy + 0.38], color=C["ink2"], lw=1.6, ls=(0, (2, 1.5)))
    ax.axvline(100, color=C["crit"], ls="--", lw=1.5)
    ax.text(100, len(names) - 0.4, " limit", color=C["crit"], fontsize=9, va="bottom")
    ax.plot([], [], color=C["ink2"], lw=1.6, ls=(0, (2, 1.5)), label="rated (continuous)")
    ax.set_yticks(y, [short_joint(n) for n in names])
    ax.set_xlim(0, 110)
    ax.set_xlabel("joint torque, % of motor peak", color=C["ink2"])
    ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=13, fontweight="bold", color=C["ink"], pad=30)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3,
              borderaxespad=0.2)
    ax.grid(axis="y", visible=False)

    ax = fig.add_subplot(gs[2, 2])
    if "push" in R and R["push"].get("dir_survival"):
        P = R["push"]
        heat(ax, P["dir_survival"], [f"pushed {d}" for d in PUSH_DIR_NAMES],
             "Push survival by direction (walkable speeds)", "", xl="push force (N)",
             xlabels=[f"{f:g}" for f in P["forces"]])
    else:
        not_run(ax, "Push survival by direction")

    # row 3: headline + findings ----------------------------------------
    ax = fig.add_subplot(gs[3, 0])
    ax.axis("off")
    ax.set_title("Headline results", loc="left", fontsize=13, fontweight="bold", color=C["ink"], pad=10)
    yy = 0.98
    for lab, val in R["headline"]:
        ax.text(0.0, yy, lab, fontsize=10, color=C["ink2"], transform=ax.transAxes, va="top")
        ax.text(0.0, yy - 0.032, val, fontsize=13, fontweight="bold", color=C["ink"], transform=ax.transAxes, va="top")
        yy -= 0.088
    ax.text(0.0, max(yy, 0.0), "red speed labels = it falls there even without a disturbance",
            fontsize=9, color=C["crit"], transform=ax.transAxes, va="top")

    ax = fig.add_subplot(gs[3, 1:])
    ax.axis("off")
    ax.set_title("What failed and what to change first", loc="left", fontsize=13, fontweight="bold",
                 color=C["ink"], pad=10)
    yy = 0.98
    tag_col = {"fail": C["crit"], "warn": C["warn_text"], "ok": C["good"]}
    for f in R["findings"]:
        if yy < 0.06:
            ax.text(0.0, yy, "… more in the Evaluation tab / JSON", fontsize=10, color=C["muted"],
                    transform=ax.transAxes, va="top")
            break
        ax.text(0.0, yy, f["severity"].upper(), fontsize=10.5, fontweight="bold", color=tag_col[f["severity"]],
                transform=ax.transAxes, va="top")
        ax.text(0.055, yy, f"[{f['test']}] {f['title']}", fontsize=11, fontweight="bold", color=C["ink"],
                transform=ax.transAxes, va="top")
        yy -= 0.042
        if f["fixes"]:
            lines = textwrap.wrap("→ " + f["fixes"][0], 160)
            if len(lines) > 2:
                lines = lines[:2]
                lines[1] = lines[1].rsplit(" ", 1)[0] + " …"
            for line in lines:
                ax.text(0.055, yy, line, fontsize=9.5, color=C["ink2"], transform=ax.transAxes, va="top")
                yy -= 0.034
        yy -= 0.012

    # row 4: gait table ------------------------------------------------
    ax = fig.add_subplot(gs[4, :])
    ax.axis("off")
    ax.set_title("Gait quality & effort at each commanded speed (robots that did not fall)", loc="left",
                 fontsize=13, fontweight="bold", color=C["ink"], pad=6)
    g = W["gait"]
    cells = []
    for key, label, fmt, good, _ in GAIT_ROWS:
        row = [label]
        for i, s in enumerate(S):
            v = g[key][i]
            row.append("–" if (v is None or not np.isfinite(v) or (key == "cot" and s == 0)) else fmt.format(v))
        cells.append(row + [good])
    tbl = ax.table(cellText=cells, colLabels=["metric"] + [f"{s:g} m/s" for s in S] + ["what's good"],
                   loc="upper center", cellLoc="center", colWidths=[0.17] + [0.085] * len(S) + [0.2])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(C["grid"])
        if r == 0:
            cell.set_facecolor(C["header"])
            cell.set_text_props(fontweight="bold", color=C["ink"])
        else:
            cell.set_facecolor(C["surface"] if r % 2 else "#f5f4f1")
            if c == 0:
                cell.set_text_props(ha="left", color=C["ink"])
            elif c == len(S) + 1:
                cell.set_text_props(color=C["muted"])
    fig.savefig(path, facecolor=C["surface"])


# ====================================================================== 3D view
class View(QtWidgets.QLabel):
    def __init__(self, sim, on_key):
        super().__init__()
        self.sim, self.on_key = sim, on_key
        self.setMinimumSize(480, 360)
        self.setAlignment(Qt.AlignCenter)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet(f"background:{C['surface']}; color:{C['ink2']};")
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.reset_camera()
        self._drag = None
        try:
            self.renderer = mujoco.Renderer(sim.model, VIEW_H, VIEW_W)
        except Exception as e:  # noqa: BLE001
            self.renderer = None
            self.setText(f"3D view unavailable ({e}).\nThe simulation and torque panels still work.")

    def reset_camera(self):
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = self.sim.base
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.6, 135.0, -15.0

    def render(self):
        if self.renderer is None:
            return
        r = self.renderer
        r.update_scene(self.sim.data, camera=self.cam, scene_option=self.opt)
        s = self.sim
        if s.push_left > 0 and r.scene.ngeom < r.scene.maxgeom:
            com = s.data.xipos[s.push_body].copy()
            f = s.push_force
            length = float(np.clip(np.linalg.norm(f) * 0.004, 0.15, 0.8))
            start = com - f / (np.linalg.norm(f) + 1e-9) * length
            g = r.scene.geoms[r.scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                                np.eye(3).ravel(), np.array([0.29, 0.23, 0.65, 1.0], np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.025, start, com)
            r.scene.ngeom += 1
        px = np.ascontiguousarray(r.render())
        img = QtGui.QImage(px.data, px.shape[1], px.shape[0], 3 * px.shape[1],
                           QtGui.QImage.Format_RGB888).copy()
        self.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def mousePressEvent(self, e):
        self.setFocus()
        self._drag = (e.position(), e.button())

    def mouseMoveEvent(self, e):
        if not self._drag:
            return
        p0, btn = self._drag
        d = e.position() - p0
        if btn == Qt.LeftButton:
            self.cam.azimuth -= d.x() * 0.3
            self.cam.elevation = float(np.clip(self.cam.elevation - d.y() * 0.3, -89, 89))
        elif btn == Qt.RightButton:
            self.cam.distance = float(np.clip(self.cam.distance * (1 + d.y() * 0.005), 0.5, 15))
        self._drag = (e.position(), btn)

    def mouseReleaseEvent(self, e):
        self._drag = None

    def wheelEvent(self, e):
        self.cam.distance = float(np.clip(self.cam.distance * 0.9 ** (e.angleDelta().y() / 120),
                                          0.5, 15))

    def mouseDoubleClickEvent(self, e):
        self.reset_camera()

    def keyPressEvent(self, e):
        if not self.on_key(e.key()):
            super().keyPressEvent(e)


# ====================================================================== torque widgets
def _pen(color, width=1.0, style=Qt.SolidLine):
    p = QtGui.QPen(QtGui.QColor(color), width)
    p.setStyle(style)
    return p


class TorqueBars(QtWidgets.QWidget):
    """Live |torque| per joint on its own motor's scale (0 .. motor peak)."""

    def __init__(self):
        super().__init__()
        self.cur = np.zeros(robot.NUM_JOINTS)
        self.peak = np.zeros(robot.NUM_JOINTS)
        self.limit_frac = 1.0          # motor strength from Test conditions
        self.setMinimumHeight(28 * robot.NUM_JOINTS + 46)

    def set_values(self, cur, peak):
        self.cur, self.peak = cur, peak
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(C["surface"]))
        f = p.font()
        f.setPointSizeF(8.5)
        p.setFont(f)
        names = robot.JOINT_NAMES
        label_w, value_w = 150, 128
        x0, x1 = label_w, self.width() - value_w
        row_h = (self.height() - 40) / len(names)
        for i, n in enumerate(names):
            rated, pk = MOTORS[motor_of(n)]
            lim = pk * self.limit_frac
            y = 4 + i * row_h
            cy = y + row_h / 2
            side = C["left"] if n.startswith("left") else C["right"]
            p.setPen(_pen(C["ink"]))
            p.drawText(QtCore.QRectF(4, y, label_w - 8, row_h), Qt.AlignVCenter | Qt.AlignLeft,
                       f"{n}  ({motor_of(n)})")
            bh = min(12.0, row_h * 0.5)
            p.fillRect(QtCore.QRectF(x0, cy - bh / 2, x1 - x0, bh), QtGui.QColor(C["track"]))
            frac = min(self.cur[i] / pk, 1.0)
            color = C["crit"] if self.cur[i] >= 0.98 * lim else C["warn"] if self.cur[i] > rated else side
            p.setPen(Qt.NoPen)
            p.setBrush(QtGui.QColor(color))
            p.drawRoundedRect(QtCore.QRectF(x0, cy - bh / 2, max(frac * (x1 - x0), 1.0), bh), 2, 2)
            xr = x0 + rated / pk * (x1 - x0)
            p.setPen(_pen(C["ink2"], 1.5, Qt.DashLine))
            p.drawLine(QtCore.QPointF(xr, cy - bh), QtCore.QPointF(xr, cy + bh))
            if self.limit_frac < 1.0:
                xl = x0 + self.limit_frac * (x1 - x0)
                p.setPen(_pen(C["crit"], 1.5, Qt.DashLine))
                p.drawLine(QtCore.QPointF(xl, cy - bh), QtCore.QPointF(xl, cy + bh))
            xp = x0 + min(self.peak[i] / pk, 1.0) * (x1 - x0)
            p.setPen(_pen(C["ink"], 2.0))
            p.drawLine(QtCore.QPointF(xp, cy - bh * 0.9), QtCore.QPointF(xp, cy + bh * 0.9))
            tag = "  AT LIMIT" if self.peak[i] >= 0.98 * lim else ("  > rated" if self.peak[i] > rated else "")
            p.setPen(_pen(C["crit"] if "LIMIT" in tag else C["ink"] if not tag else C["warn_text"]))
            p.drawText(QtCore.QRectF(x1 + 6, y, value_w - 6, row_h), Qt.AlignVCenter | Qt.AlignLeft,
                       f"{self.cur[i]:5.1f} | {self.peak[i]:5.1f}{tag}")
        p.setPen(_pen(C["ink2"]))
        p.drawText(QtCore.QRectF(4, self.height() - 36, self.width() - 8, 16),
                   Qt.AlignVCenter | Qt.AlignLeft,
                   "bar = |torque| now (blue left, orange right, yellow > rated, red at limit)")
        extra = f"   ┆ red = {self.limit_frac:.0%} strength limit" if self.limit_frac < 1.0 else ""
        p.drawText(QtCore.QRectF(4, self.height() - 19, self.width() - 8, 16),
                   Qt.AlignVCenter | Qt.AlignLeft,
                   "┃ peak since last push   ┆ rated   bar end = motor peak   numbers: now | peak N·m" + extra)


class TorquePlot(QtWidgets.QWidget):
    """Rolling torque of one joint type, left vs right, with the motor's rated / peak lines."""

    def __init__(self, n_hist):
        super().__init__()
        self.t = deque(maxlen=n_hist)
        self.tau = deque(maxlen=n_hist)
        self.push = deque(maxlen=n_hist)
        self.kind = "knee_pitch"
        self.setMinimumHeight(220)

    def add(self, t, tau, push):
        self.t.append(t)
        self.tau.append(tau)
        self.push.append(push)

    def clear(self):
        self.t.clear()
        self.tau.clear()
        self.push.clear()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(C["surface"]))
        f = p.font()
        f.setPointSizeF(8.5)
        p.setFont(f)
        L, R, T, B = 46, 10, 40, 24
        w, h = self.width() - L - R, self.height() - T - B
        rated, pk = MOTORS[JOINT_MOTOR[self.kind]]
        jl = robot.JOINT_NAMES.index("left_" + self.kind)
        jr = robot.JOINT_NAMES.index("right_" + self.kind)
        t = np.array(self.t)
        tau = np.array(self.tau) if self.tau else np.zeros((0, robot.NUM_JOINTS))
        top = rated * 1.25
        if len(tau):
            top = min(max(top, np.abs(tau[:, [jl, jr]]).max() * 1.15), pk * 1.05)
        t1 = t[-1] if len(t) else HISTORY_S
        t0 = t1 - HISTORY_S

        def X(tt):
            return L + (tt - t0) / HISTORY_S * w

        def Y(v):
            return T + h / 2 - v / top * h / 2

        if len(t):  # push intervals
            pu = np.array(self.push)
            edges = np.flatnonzero(np.diff(np.r_[0, pu.astype(int), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                p.fillRect(QtCore.QRectF(X(t[a]), T, max(X(t[b - 1]) - X(t[a]), 2), h),
                           QtGui.QColor(74, 58, 167, 40))
        p.setPen(_pen(C["grid"]))
        p.drawLine(QtCore.QPointF(L, Y(0)), QtCore.QPointF(L + w, Y(0)))
        for v, style, lab in ((rated, Qt.DashLine, "rated"), (pk, Qt.SolidLine, "peak")):
            if v <= top:
                for s in (1, -1):
                    p.setPen(_pen(C["ink2"] if lab == "rated" else C["ink"], 1.2, style))
                    p.drawLine(QtCore.QPointF(L, Y(s * v)), QtCore.QPointF(L + w, Y(s * v)))
                p.setPen(_pen(C["ink2"]))
                p.drawText(QtCore.QRectF(L + w - 120, Y(v) - 15, 118, 14), Qt.AlignRight,
                           f"{lab} ±{v:.0f}")
        p.setPen(_pen(C["ink2"]))
        for v in (top, top / 2, 0, -top / 2, -top):
            p.drawText(QtCore.QRectF(0, Y(v) - 7, L - 6, 14), Qt.AlignRight | Qt.AlignVCenter,
                       f"{v:.0f}")
        p.drawText(QtCore.QRectF(L, T + h + 4, w, 16), Qt.AlignCenter,
                   f"last {HISTORY_S:.0f} s   (shaded = push applied)")
        if len(t) > 1:
            step = max(1, len(t) // 600)
            for j, col in ((jl, C["left"]), (jr, C["right"])):
                path = QtGui.QPainterPath(QtCore.QPointF(X(t[0]), Y(tau[0, j])))
                for k in range(step, len(t), step):
                    path.lineTo(X(t[k]), Y(tau[k, j]))
                p.setPen(_pen(col, 2.0))
                p.setBrush(Qt.NoBrush)
                p.drawPath(path)
        p.setPen(_pen(C["ink"]))
        p.drawText(QtCore.QRectF(L, 2, w, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   f"{self.kind} torque (N·m, {JOINT_MOTOR[self.kind]})")
        for k, (lab, col) in enumerate((("left leg", C["left"]), ("right leg", C["right"]))):
            x = L + k * 90
            p.fillRect(QtCore.QRectF(x, 26, 14, 3), QtGui.QColor(col))
            p.drawText(QtCore.QRectF(x + 18, 19, 70, 16), Qt.AlignLeft | Qt.AlignVCenter, lab)


# ====================================================================== gait + evaluation widgets
class GaitPanel(QtWidgets.QWidget):
    """Live gait quality over the last few seconds, same formulas as the evaluation's table."""

    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        for c, h in enumerate(("metric", f"last {LIVE_WINDOW_S:.0f} s", "what's good")):
            lab = QtWidgets.QLabel(f"<b>{h}</b>")
            grid.addWidget(lab, 0, c)
        self.rows = [("speed", "Speed actual / command (m/s)", "", None),
                     ("err", "Speed error (m/s)", f"< {TRACK_OK}", None)] + \
                    [(k, lab, good, rng) for k, lab, _, good, rng in GAIT_ROWS]
        self.vals = {}
        for r, (key, lab, good, _) in enumerate(self.rows, start=1):
            grid.addWidget(QtWidgets.QLabel(lab), r, 0)
            v = QtWidgets.QLabel("–")
            v.setMinimumWidth(110)
            grid.addWidget(v, r, 1)
            g = QtWidgets.QLabel(good)
            g.setStyleSheet(f"color:{C['muted']}")
            grid.addWidget(g, r, 2)
            self.vals[key] = v
        lay.addLayout(grid)
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color:{C['ink2']}")
        lay.addWidget(self.note)
        lay.addStretch(1)

    def show_metrics(self, m, actual, cmd, conditions):
        moving = np.linalg.norm(cmd[:2]) > 0.1
        self.note.setText(f"Conditions: {conditions}.  A value marked “!” is outside the good range. "
                          "Pushes inside the window are included.")
        if m is None:
            for v in self.vals.values():
                v.setText("–")
            return
        self.vals["speed"].setText(f"{actual[0]:+.2f} / {cmd[0]:+.2f}")
        e = float(np.linalg.norm(actual[:2] - cmd[:2]))
        self._set("err", f"{e:.2f}", e > TRACK_OK)
        fmts = {k: f for k, _, f, _, _ in GAIT_ROWS}
        for key, _, _, rng in self.rows[2:]:
            v = m[key]
            if not np.isfinite(v):
                self._set(key, "–", False)
                continue
            out = False
            if rng is not None and (moving or key != "clearance"):
                lo, hi = rng
                out = (lo is not None and v < lo) or (hi is not None and v > hi)
            self._set(key, fmts[key].format(v), out)

    def _set(self, key, text, flag):
        lab = self.vals[key]
        lab.setText(text + ("  !" if flag else ""))
        lab.setStyleSheet(f"color:{C['warn_text'] if flag else C['ink']}; font-weight:{'bold' if flag else 'normal'}")


class EvalWorker(QtCore.QObject):
    progress = QtCore.Signal(float, str)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self._stop = False

    def stop(self):
        self._stop = True

    @QtCore.Slot()
    def run(self):
        try:
            suite = EvalSuite(progress=lambda f, m: self.progress.emit(float(f), m),
                              stop=lambda: self._stop, **self.kwargs)
            R = suite.run()
            R["nominal_friction"] = suite.nom.friction
            png, js = save_results(R)
            R["files"] = dict(png=str(png) if png else None, json=str(js))
            self.finished.emit(R)
        except Stopped:
            self.failed.emit("stopped")
        except Exception:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())


class ReportWindow(QtWidgets.QMainWindow):
    def __init__(self, png, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Evaluation report: {Path(png).name}")
        self.pix = QtGui.QPixmap(str(png))
        self.label = QtWidgets.QLabel()
        self.label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidget(self.label)
        self.scroll.setWidgetResizable(True)
        self.fit = QtWidgets.QCheckBox("fit to width")
        self.fit.setChecked(True)
        self.fit.toggled.connect(self._show)
        bar = self.addToolBar("report")
        bar.addWidget(self.fit)
        self.setCentralWidget(self.scroll)
        self.resize(1300, 950)
        self._show()

    def _show(self):
        if self.fit.isChecked():
            w = max(400, self.scroll.viewport().width() - 4)
            self.label.setPixmap(self.pix.scaledToWidth(w, Qt.SmoothTransformation))
        else:
            self.label.setPixmap(self.pix)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._show()


# ====================================================================== main window
class Lab(QtWidgets.QMainWindow):
    def __init__(self, policy):
        super().__init__()
        self.setWindowTitle("OC1 robot lab: policy, pushes, conditions, joint torque, evaluation")
        self.sim = Sim()
        self.running = True
        self.speed = 1.0
        self.last = time.perf_counter()
        self.acc = 0.0
        self.fallen_since = None
        self.peak_push = np.zeros(robot.NUM_JOINTS)
        self.cur = np.zeros(robot.NUM_JOINTS)
        self.record = None
        self.ramp = None
        self.pending_push = None
        self.eval_thread = None
        self.eval_worker = None
        self.last_eval = None
        self.report_windows = []
        self._gait_t = 0.0
        self._build_ui()
        if policy:
            self._load(policy)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(int(STEP_DT * 1000))

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        self.view = View(self.sim, self.handle_key)
        n_hist = int(HISTORY_S / self.sim.model.opt.timestep)
        self.bars = TorqueBars()
        self.plot = TorquePlot(n_hist)

        ctl = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(ctl)

        # policy
        g = QtWidgets.QGroupBox("Policy")
        gl = QtWidgets.QGridLayout(g)
        self.policy_label = QtWidgets.QLabel("no policy (holding the home pose)")
        self.policy_label.setWordWrap(True)
        load = QtWidgets.QPushButton("Load .onnx…")
        load.clicked.connect(self._choose_policy)
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        reset = QtWidgets.QPushButton("Reset robot")
        reset.clicked.connect(self.reset)
        self.speed_box = QtWidgets.QComboBox()
        self.speed_box.addItems(["1× real time", "0.5×", "0.25×"])
        self.speed_box.currentIndexChanged.connect(
            lambda i: setattr(self, "speed", [1.0, 0.5, 0.25][i]))
        self.auto_reset = QtWidgets.QCheckBox("auto-reset 1 s after a fall")
        gl.addWidget(self.policy_label, 0, 0, 1, 2)
        gl.addWidget(load, 1, 0)
        gl.addWidget(self.speed_box, 1, 1)
        gl.addWidget(self.pause_btn, 2, 0)
        gl.addWidget(reset, 2, 1)
        gl.addWidget(self.auto_reset, 3, 0, 1, 2)
        cl.addWidget(g)

        # command
        g = QtWidgets.QGroupBox("Velocity command")
        gl = QtWidgets.QGridLayout(g)
        self.sliders, self.slider_labels = [], []
        for i, name in enumerate(["vx forward (m/s)", "vy left (m/s)", "yaw rate (rad/s)"]):
            s = QtWidgets.QSlider(Qt.Horizontal)
            s.setRange(int(CMD_LIMITS[i, 0] * 100), int(CMD_LIMITS[i, 1] * 100))
            s.setSingleStep(5)
            s.valueChanged.connect(lambda v, i=i: self._set_cmd(i, v / 100))
            lab = QtWidgets.QLabel("0.00")
            lab.setMinimumWidth(40)
            gl.addWidget(QtWidgets.QLabel(name), i, 0)
            gl.addWidget(s, i, 1)
            gl.addWidget(lab, i, 2)
            self.sliders.append(s)
            self.slider_labels.append(lab)
        self.hold = QtWidgets.QCheckBox("heading hold when yaw rate = 0 (as play.py)")
        self.hold.setChecked(True)
        self.hold.toggled.connect(lambda v: setattr(self.sim, "heading_hold", v))
        stop = QtWidgets.QPushButton("Stop (all 0)")
        stop.clicked.connect(lambda: [s.setValue(0) for s in self.sliders])
        gl.addWidget(self.hold, 3, 0, 1, 3)
        gl.addWidget(stop, 4, 0, 1, 3)
        cl.addWidget(g)

        # test conditions
        g = QtWidgets.QGroupBox("Test conditions (same levers as the evaluation)")
        gl = QtWidgets.QGridLayout(g)
        self.c_fric = QtWidgets.QDoubleSpinBox()
        self.c_fric.setRange(0.05, 2.0)
        self.c_fric.setSingleStep(0.05)
        self.c_fric.setValue(self.sim.nom.friction)
        self.c_payload = QtWidgets.QDoubleSpinBox()
        self.c_payload.setRange(0.0, 20.0)
        self.c_payload.setSingleStep(0.5)
        self.c_payload.setSuffix(" kg")
        self.c_strength = QtWidgets.QSpinBox()
        self.c_strength.setRange(20, 100)
        self.c_strength.setSingleStep(5)
        self.c_strength.setValue(100)
        self.c_strength.setSuffix(" %")
        self.c_noise = QtWidgets.QCheckBox("sensor noise (training levels)")
        nominal = QtWidgets.QPushButton("Nominal")
        nominal.clicked.connect(self.set_nominal_conditions)
        rows = [(f"foot friction μ (training {0.3:g}–{1.6:g})", self.c_fric),
                ("payload on torso", self.c_payload), ("motor strength", self.c_strength)]
        for r, (lab, wdg) in enumerate(rows):
            gl.addWidget(QtWidgets.QLabel(lab), r, 0)
            gl.addWidget(wdg, r, 1)
            wdg.valueChanged.connect(self._conditions_changed)
        self.c_noise.toggled.connect(self._conditions_changed)
        gl.addWidget(self.c_noise, 3, 0)
        gl.addWidget(nominal, 3, 1)
        cl.addWidget(g)

        # push
        g = QtWidgets.QGroupBox("Disturbance (push)")
        gl = QtWidgets.QGridLayout(g)
        self.body_box = QtWidgets.QComboBox()
        self.body_box.addItems(self.sim.bodies)
        self.body_box.setCurrentText(robot.BASE_BODY)
        self.dir_box = QtWidgets.QComboBox()
        self.dir_box.addItems(["front → pushes forward", "back → pushes backward",
                               "left → pushes to its left", "right → pushes to its right",
                               "custom angle"])
        self.angle = QtWidgets.QDoubleSpinBox()
        self.angle.setRange(0, 359)
        self.angle.setSuffix(" °")
        self.angle.setEnabled(False)
        self.dir_box.currentIndexChanged.connect(lambda i: self.angle.setEnabled(i == 4))
        self.force = QtWidgets.QDoubleSpinBox()
        self.force.setRange(1, 2000)
        self.force.setValue(100)
        self.force.setSuffix(" N")
        self.duration = QtWidgets.QDoubleSpinBox()
        self.duration.setRange(0.01, 5)
        self.duration.setSingleStep(0.05)
        self.duration.setValue(0.2)
        self.duration.setSuffix(" s")
        self.impulse = QtWidgets.QLabel()
        for wdg in (self.force, self.duration):
            wdg.valueChanged.connect(self._update_impulse)
        self._update_impulse()
        push = QtWidgets.QPushButton("Apply push  (Space)")
        push.clicked.connect(self.apply_push)
        rows = [("body", self.body_box), ("direction", self.dir_box), ("angle (0 = front, CCW)", self.angle),
                ("force", self.force), ("duration", self.duration)]
        for r, (lab, wdg) in enumerate(rows):
            gl.addWidget(QtWidgets.QLabel(lab), r, 0)
            gl.addWidget(wdg, r, 1)
        gl.addWidget(self.impulse, 5, 0, 1, 2)
        gl.addWidget(push, 6, 0, 1, 2)
        cl.addWidget(g)

        # ramp test
        g = QtWidgets.QGroupBox("Push-recovery test (raises the force until it falls)")
        gl = QtWidgets.QGridLayout(g)
        self.r_start, self.r_step, self.r_max = (QtWidgets.QDoubleSpinBox() for _ in range(3))
        for wdg, v in ((self.r_start, 20), (self.r_step, 10), (self.r_max, 400)):
            wdg.setRange(1, 3000)
            wdg.setValue(v)
            wdg.setSuffix(" N")
        self.ramp_btn = QtWidgets.QPushButton("Run test")
        self.ramp_btn.clicked.connect(self.toggle_ramp)
        self.ramp_out = QtWidgets.QPlainTextEdit()
        self.ramp_out.setReadOnly(True)
        self.ramp_out.setMaximumHeight(120)
        for c, (lab, wdg) in enumerate((("start", self.r_start), ("step", self.r_step),
                                        ("max", self.r_max))):
            gl.addWidget(QtWidgets.QLabel(lab), 0, 2 * c)
            gl.addWidget(wdg, 0, 2 * c + 1)
        gl.addWidget(QtWidgets.QLabel("uses the direction, body and duration above"), 1, 0, 1, 6)
        gl.addWidget(self.ramp_btn, 2, 0, 1, 6)
        gl.addWidget(self.ramp_out, 3, 0, 1, 6)
        cl.addWidget(g)
        cl.addStretch(1)

        # torque tab
        tq = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(tq)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(QtWidgets.QLabel("<b>Joint torque</b>"))
        head.addStretch(1)
        rp = QtWidgets.QPushButton("Reset peaks")
        rp.clicked.connect(self.reset_peaks)
        self.rec_btn = QtWidgets.QPushButton("Record CSV")
        self.rec_btn.setCheckable(True)
        self.rec_btn.toggled.connect(self.toggle_record)
        head.addWidget(rp)
        head.addWidget(self.rec_btn)
        tl.addLayout(head)
        tl.addWidget(self.bars, 3)
        ph = QtWidgets.QHBoxLayout()
        ph.addWidget(QtWidgets.QLabel("plot joint:"))
        self.kind_box = QtWidgets.QComboBox()
        self.kind_box.addItems(JOINT_TYPES)
        self.kind_box.setCurrentText(self.plot.kind)
        self.kind_box.currentTextChanged.connect(lambda k: setattr(self.plot, "kind", k))
        ph.addWidget(self.kind_box)
        ph.addStretch(1)
        tl.addLayout(ph)
        tl.addWidget(self.plot, 2)

        self.gait = GaitPanel()
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(tq, "Joint torque")
        self.tabs.addTab(self.gait, "Gait (live)")
        self.tabs.addTab(self._build_eval_tab(), "Evaluation")

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(ctl)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)
        split = QtWidgets.QSplitter()
        split.addWidget(self.view)
        split.addWidget(scroll)
        self.tabs.setMinimumWidth(580)
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 0)
        split.setStretchFactor(2, 2)
        self.setCentralWidget(split)
        self.status = QtWidgets.QLabel()
        self.statusBar().addWidget(self.status, 1)
        self.resize(1780, 980)

    def _build_eval_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        g = QtWidgets.QGroupBox("Run the walking-policy evaluation")
        gv = QtWidgets.QVBoxLayout(g)
        gl = QtWidgets.QGridLayout()
        gv.addLayout(gl)
        self.e_preset = QtWidgets.QComboBox()
        self.e_preset.addItems(["full (as the report)", "quick (fewer trials)"])
        self.e_threads = QtWidgets.QSpinBox()
        self.e_threads.setRange(1, 128)
        self.e_threads.setValue(os.cpu_count() or 4)
        self.e_tests = {}
        tests_box = QtWidgets.QHBoxLayout()
        for t in OPTIONAL_TESTS:
            cb = QtWidgets.QCheckBox(TESTS[t][0])
            cb.setChecked(True)
            cb.setToolTip(TESTS[t][1])
            self.e_tests[t] = cb
            tests_box.addWidget(cb)
        self.e_push_dur = QtWidgets.QDoubleSpinBox()
        self.e_push_dur.setRange(0.02, 2.0)
        self.e_push_dur.setSingleStep(0.05)
        self.e_push_dur.setValue(0.2)
        self.e_push_dur.setSuffix(" s")
        self.e_hold = QtWidgets.QCheckBox("heading hold (off = the policy's own drift, as in the report)")
        self.e_noise = QtWidgets.QCheckBox("sensor noise")
        gl.addWidget(QtWidgets.QLabel("trials"), 0, 0)
        gl.addWidget(self.e_preset, 0, 1)
        gl.addWidget(QtWidgets.QLabel("threads"), 0, 2)
        gl.addWidget(self.e_threads, 0, 3)
        gl.addLayout(tests_box, 1, 0, 1, 4)
        gl.addWidget(QtWidgets.QLabel("push duration"), 2, 0)
        gl.addWidget(self.e_push_dur, 2, 1)
        gl.addWidget(self.e_noise, 2, 2, 1, 2)
        gl.addWidget(self.e_hold, 3, 0, 1, 4)
        note = QtWidgets.QLabel("Speed tracking, straightness, response time, gait and torque always run: "
                                "the other tests are judged against them. The live robot pauses while it runs.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{C['ink2']}")
        gv.addWidget(note)
        row = QtWidgets.QHBoxLayout()
        self.e_run = QtWidgets.QPushButton("Run evaluation")
        self.e_run.clicked.connect(self.toggle_eval)
        self.e_bar = QtWidgets.QProgressBar()
        self.e_bar.setRange(0, 1000)
        row.addWidget(self.e_run)
        row.addWidget(self.e_bar, 1)
        gv.addLayout(row)
        self.e_status = QtWidgets.QLabel("")
        self.e_status.setWordWrap(True)
        self.e_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        gv.addWidget(self.e_status)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("Results")
        gl = QtWidgets.QVBoxLayout(g)
        row = QtWidgets.QHBoxLayout()
        self.e_show = QtWidgets.QPushButton("Show report")
        self.e_show.clicked.connect(self.show_report)
        self.e_folder = QtWidgets.QPushButton("Open folder")
        self.e_folder.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(ROOT / "eval_reports"))))
        self.e_load = QtWidgets.QPushButton("Load results .json…")
        self.e_load.clicked.connect(self.load_results)
        for b in (self.e_show, self.e_folder, self.e_load):
            row.addWidget(b)
        row.addStretch(1)
        self.e_show.setEnabled(False)
        gl.addLayout(row)
        # headline | findings | details, each scrolls by itself and the dividers can be dragged
        split = QtWidgets.QSplitter(Qt.Vertical)
        self.e_headline = QtWidgets.QTextBrowser()
        self.e_headline.setHtml(f"<p style='color:{C['ink2']}'>No evaluation yet.</p>")
        self.e_headline.setMinimumHeight(90)
        box = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(box)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(QtWidgets.QLabel("<b>Findings</b>: click one for why it matters and what to change; "
                                      "double-click to watch it"))
        self.e_find = QtWidgets.QListWidget()
        self.e_find.setMinimumHeight(90)
        self.e_find.currentRowChanged.connect(self._show_finding)
        self.e_find.itemDoubleClicked.connect(lambda _: self._watch_finding())
        bl.addWidget(self.e_find)
        self.e_detail = QtWidgets.QTextBrowser()
        self.e_detail.setMinimumHeight(110)
        for wdg in (self.e_headline, box, self.e_detail):
            split.addWidget(wdg)
        split.setSizes([230, 200, 300])
        gl.addWidget(split, 1)
        row = QtWidgets.QHBoxLayout()
        self.e_watch_find = QtWidgets.QPushButton("Watch this in the lab")
        self.e_watch_find.clicked.connect(self._watch_finding)
        self.e_watch_find.setEnabled(False)
        row.addWidget(self.e_watch_find)
        row.addStretch(1)
        gl.addLayout(row)
        gl.addWidget(QtWidgets.QLabel("<b>Failed cases</b> (mildest failing condition per speed)"))
        row = QtWidgets.QHBoxLayout()
        self.e_cases = QtWidgets.QComboBox()
        self.e_cases.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.e_watch_case = QtWidgets.QPushButton("Watch")
        self.e_watch_case.clicked.connect(lambda: self.watch_case(self.e_cases.currentData()))
        self.e_watch_case.setEnabled(False)
        row.addWidget(self.e_cases, 1)
        row.addWidget(self.e_watch_case)
        gl.addLayout(row)
        lay.addWidget(g, 1)
        return w

    # ---------------------------------------------------------------- actions
    def _load(self, spec):
        path = Path(spec)
        if not path.is_file():
            try:
                from oc1_rl.paths import resolve_policy
                path = resolve_policy(spec)
            except Exception as e:  # noqa: BLE001
                self.policy_label.setText(f"could not find policy '{spec}': {e}")
                return
        try:
            self.sim.load_policy(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Policy", str(e))
            return
        try:
            shown = path.resolve().relative_to(ROOT)
        except ValueError:
            shown = path
        self.policy_label.setText(f"policy: {shown}")
        self.reset()

    def _choose_policy(self):
        start = ROOT / "runs" if (ROOT / "runs").exists() else ROOT
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load policy", str(start), "ONNX (*.onnx)")
        if f:
            self._load(f)

    def _set_cmd(self, i, v):
        self.sim.cmd[i] = v
        self.slider_labels[i].setText(f"{v:+.2f}")

    def _update_impulse(self):
        self.impulse.setText(f"impulse = {self.force.value() * self.duration.value():.1f} N·s")

    def conditions(self):
        return Conditions(friction=self.c_fric.value(), payload=self.c_payload.value(),
                          strength=self.c_strength.value() / 100.0)

    def _conditions_changed(self, *_):
        cond = self.conditions()
        self.sim.set_conditions(cond, obs_noise=self.c_noise.isChecked())
        self.bars.limit_frac = cond.strength
        self.bars.update()

    def set_conditions(self, friction, payload, strength, noise=None):
        for wdg in (self.c_fric, self.c_payload, self.c_strength, self.c_noise):
            wdg.blockSignals(True)
        self.c_fric.setValue(friction)
        self.c_payload.setValue(payload)
        self.c_strength.setValue(int(round(strength * 100)))
        if noise is not None:
            self.c_noise.setChecked(noise)
        for wdg in (self.c_fric, self.c_payload, self.c_strength, self.c_noise):
            wdg.blockSignals(False)
        self._conditions_changed()

    def set_nominal_conditions(self):
        self.set_conditions(self.sim.nom.friction, 0.0, 1.0, noise=False)

    def toggle_pause(self):
        self.running = not self.running
        self.pause_btn.setText("Pause" if self.running else "Resume")

    def reset(self):
        self.sim.reset()
        self.plot.clear()
        self.fallen_since = None
        self.pending_push = None
        self.reset_peaks()

    def reset_peaks(self):
        self.peak_push[:] = 0.0

    def push_vector(self, magnitude):
        angle = [0.0, 180.0, 90.0, 270.0, self.angle.value()][self.dir_box.currentIndex()]
        a = self.sim.heading() + math.radians(angle)
        return magnitude * np.array([math.cos(a), math.sin(a), 0.0])

    def apply_push(self, magnitude=None):
        mag = self.force.value() if not magnitude else magnitude
        body = self.sim.model.body(self.body_box.currentText()).id
        self.sim.start_push(body, self.push_vector(mag), self.duration.value())
        self.reset_peaks()

    def handle_key(self, k):
        steps = {Qt.Key_Up: (0, 0.1), Qt.Key_Down: (0, -0.1), Qt.Key_Comma: (1, 0.1),
                 Qt.Key_Period: (1, -0.1), Qt.Key_Left: (2, 0.2), Qt.Key_Right: (2, -0.2)}
        if k in steps:
            i, dv = steps[k]
            self.sliders[i].setValue(int(round((self.sim.cmd[i] + dv) * 100)))
        elif k == Qt.Key_0:
            for s in self.sliders:
                s.setValue(0)
        elif k == Qt.Key_Space:
            self.apply_push()
        elif k == Qt.Key_Backspace:
            self.reset()
        elif k == Qt.Key_P:
            self.toggle_pause()
        else:
            return False
        return True

    def keyPressEvent(self, e):
        if not self.handle_key(e.key()):
            super().keyPressEvent(e)

    def toggle_record(self, on):
        if on:
            self.record = []
            self.rec_btn.setText("Stop && save CSV")
            return
        rows, self.record = self.record, None
        self.rec_btn.setText("Record CSV")
        if not rows:
            return
        out = ROOT / "torque_logs"
        out.mkdir(exist_ok=True)
        path = out / f"lab_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "cmd_vx", "cmd_vy", "cmd_wz", "push_fx", "push_fy", "push_fz", "fallen"]
                       + [f"tau_{n}" for n in robot.JOINT_NAMES] + ["friction", "payload_kg", "strength"])
            w.writerows(rows)
        self.statusBar().showMessage(f"saved {len(rows)} samples to {path}", 8000)

    # ---------------------------------------------------------------- push-recovery test
    def toggle_ramp(self):
        if self.ramp:
            self._ramp_finish("stopped by user")
            return
        self.ramp = {"F": self.r_start.value(), "state": "settle", "t": 0.0, "passed": None}
        self.ramp_out.setPlainText(f"push test: {self.dir_box.currentText().split(' ')[0]}, "
                                   f"{self.duration.value():.2f} s pushes on '{self.body_box.currentText()}' "
                                   f"({self.sim.cond.label(self.sim.nom.friction)})")
        self.ramp_btn.setText("Stop test")
        self.reset()

    def _ramp_step(self):
        r = self.ramp
        r["t"] += STEP_DT
        settle, watch = 2.0, self.duration.value() + 3.0
        if r["state"] == "settle":
            if self.sim.fallen():
                self._ramp_finish("the robot falls without any push: fix standing first")
            elif r["t"] >= settle:
                self.apply_push(r["F"])
                r["state"], r["t"] = "watch", 0.0
        elif r["state"] == "watch":
            j = int(np.argmax(self.peak_push / [MOTORS[motor_of(n)][1] for n in robot.JOINT_NAMES]))
            if self.sim.fallen():
                self._log(f"  {r['F']:6.0f} N  FELL      (peak {self.peak_push[j]:.1f} N·m at "
                          f"{robot.JOINT_NAMES[j]})")
                best = r["passed"]
                self._ramp_finish(f"survives {best:.0f} N, falls at {r['F']:.0f} N "
                                  f"(impulse {best * self.duration.value():.1f} N·s survived)"
                                  if best else f"falls already at {r['F']:.0f} N")
            elif r["t"] >= watch:
                self._log(f"  {r['F']:6.0f} N  recovered (peak {self.peak_push[j]:.1f} N·m at "
                          f"{robot.JOINT_NAMES[j]})")
                r["passed"] = r["F"]
                r["F"] += self.r_step.value()
                if r["F"] > self.r_max.value():
                    self._ramp_finish(f"recovered from every push up to {r['passed']:.0f} N")
                else:
                    self.reset()
                    r["state"], r["t"] = "settle", 0.0

    def _log(self, line):
        self.ramp_out.appendPlainText(line)

    def _ramp_finish(self, msg):
        self._log("RESULT: " + msg)
        self.ramp = None
        self.ramp_btn.setText("Run test")

    # ---------------------------------------------------------------- evaluation
    def toggle_eval(self):
        if self.eval_worker is not None:
            self.eval_worker.stop()
            self.e_status.setText("stopping…")
            return
        if self.sim.policy_path is None:
            QtWidgets.QMessageBox.information(self, "Evaluation", "Load a policy first.")
            return
        kw = dict(policy_path=self.sim.policy_path,
                  preset="full" if self.e_preset.currentIndex() == 0 else "quick",
                  tests=[t for t, cb in self.e_tests.items() if cb.isChecked()],
                  push_duration=self.e_push_dur.value(), heading_hold=self.e_hold.isChecked(),
                  obs_noise=self.e_noise.isChecked(), nthread=self.e_threads.value())
        self._was_running = self.running
        self.running = False
        self.pause_btn.setText("Resume")
        self.e_run.setText("Stop")
        self.e_bar.setValue(0)
        self.e_status.setText("starting…")
        self._eval_t0 = time.perf_counter()
        self.eval_thread = QtCore.QThread(self)
        self.eval_worker = EvalWorker(kw)
        self.eval_worker.moveToThread(self.eval_thread)
        self.eval_thread.started.connect(self.eval_worker.run)
        self.eval_worker.progress.connect(self._eval_progress)
        self.eval_worker.finished.connect(self._eval_done)
        self.eval_worker.failed.connect(self._eval_failed)
        self.eval_worker.finished.connect(self.eval_thread.quit)
        self.eval_worker.failed.connect(self.eval_thread.quit)
        self.eval_thread.finished.connect(self._eval_cleanup)
        self.eval_thread.start()

    def _eval_progress(self, frac, msg):
        self.e_bar.setValue(int(frac * 1000))
        el = time.perf_counter() - self._eval_t0
        eta = f", about {el / frac - el:.0f} s left" if frac > 0.03 else ""
        self.e_status.setText(f"{msg}  ({100 * frac:.0f}%{eta})")

    def _eval_cleanup(self):
        self.eval_worker.deleteLater()
        self.eval_thread.deleteLater()
        self.eval_worker = self.eval_thread = None
        self.e_run.setText("Run evaluation")
        self.running = getattr(self, "_was_running", True)
        self.pause_btn.setText("Pause" if self.running else "Resume")
        self.last = time.perf_counter()
        self.acc = 0.0

    def _eval_failed(self, msg):
        self.e_status.setText("stopped" if msg == "stopped" else "evaluation failed (details printed)")
        if msg != "stopped":
            print(msg, file=sys.stderr)
            QtWidgets.QMessageBox.warning(self, "Evaluation failed", msg[-2000:])

    def _eval_done(self, R):
        self.e_bar.setValue(1000)
        files = R.get("files", {})
        self.e_status.setText(f"done in {time.perf_counter() - self._eval_t0:.0f} s, saved {files.get('json')}")
        self.show_results(R)
        if files.get("png"):
            self.show_report()

    def show_results(self, R):
        self.last_eval = R
        hl = "".join(f"<tr><td style='color:{C['ink2']}; padding-right:12px'>{lab}</td>"
                     f"<td><b>{val}</b></td></tr>" for lab, val in R["headline"])
        self.e_headline.setHtml(f"<table cellspacing='0' cellpadding='2'>{hl}</table>")
        self.e_show.setEnabled(bool(R.get("files", {}).get("png")))
        self.e_find.clear()
        col = {"fail": C["crit"], "warn": C["warn_text"], "ok": C["good"]}
        for f in R["findings"]:
            it = QtWidgets.QListWidgetItem(f"{f['severity'].upper():5s} [{f['test']}] {f['title']}")
            it.setForeground(QtGui.QColor(col[f["severity"]]))
            self.e_find.addItem(it)
        self.e_cases.clear()
        for label, rep in R.get("watch", []):
            self.e_cases.addItem(label, rep)
        self.e_watch_case.setEnabled(self.e_cases.count() > 0)
        if R["findings"]:
            self.e_find.setCurrentRow(0)

    def _show_finding(self, row):
        if self.last_eval is None or row < 0 or row >= len(self.last_eval["findings"]):
            self.e_detail.clear()
            self.e_watch_find.setEnabled(False)
            return
        f = self.last_eval["findings"][row]
        title, why = TESTS.get(f["test"], ("", ""))
        fixes = "".join(f"<li>{x}</li>" for x in f["fixes"])
        self.e_detail.setHtml(
            f"<h3>{f['title']}</h3><p>{f['detail']}</p>"
            + (f"<p><b>What to change</b></p><ul>{fixes}</ul>" if fixes else "")
            + f"<p style='color:{C['ink2']}'><b>Why this test exists ({title}):</b> {why}</p>")
        self.e_watch_find.setEnabled(bool(f.get("watch")))

    def _watch_finding(self):
        row = self.e_find.currentRow()
        if self.last_eval and 0 <= row < len(self.last_eval["findings"]):
            w = self.last_eval["findings"][row].get("watch")
            if w:
                self.watch_case(w)

    def watch_case(self, rep):
        """Replay an evaluation case in the 3D view: same conditions, command and push."""
        if not rep:
            return
        if self.eval_worker is not None:
            self.statusBar().showMessage("wait for the evaluation to finish", 5000)
            return
        self.set_conditions(rep.get("friction", self.sim.nom.friction), rep.get("payload", 0.0),
                            rep.get("strength", 1.0))
        self.hold.setChecked(bool(rep.get("heading_hold", False)))
        for s in self.sliders:
            s.setValue(0)
        self.reset()
        self.sliders[0].setValue(int(round(rep.get("vx", 0.0) * 100)))
        msg = f"watching: vx {rep.get('vx', 0):+g} m/s, {self.sim.cond.label(self.sim.nom.friction)}"
        push = rep.get("push")
        if push:
            self.dir_box.setCurrentIndex(4)
            self.angle.setValue(push["angle"])
            self.force.setValue(push["force"])
            self.duration.setValue(push["dur"])
            self.pending_push = PUSH_AFTER_S + 0.3
            msg += f", push {push['force']:.0f} N at {push['angle']:.0f}° after {self.pending_push:.1f} s"
        self.running = True
        self.pause_btn.setText("Pause")
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage(msg, 10000)

    def show_report(self):
        png = (self.last_eval or {}).get("files", {}).get("png")
        if png and Path(png).exists():
            win = ReportWindow(png, self)
            win.show()
            self.report_windows.append(win)

    def load_results(self):
        start = ROOT / "eval_reports"
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load results", str(start if start.exists() else ROOT),
                                                     "JSON (*.json)")
        if not f:
            return
        R = json.loads(Path(f).read_text())
        png = Path(f).with_suffix(".png")
        R["files"] = dict(json=f, png=str(png) if png.exists() else None)
        R["watch"] = [tuple(x) for x in R.get("watch", [])]
        self.show_results(R)

    # ---------------------------------------------------------------- loop
    def tick(self):
        now = time.perf_counter()
        dt, self.last = now - self.last, now
        if self.running:
            self.acc += dt * self.speed
            n = 0
            while self.acc >= STEP_DT and n < 4:
                self.acc -= STEP_DT
                n += 1
                self._policy_step()
            if n == 4:
                self.acc = 0.0
        self.view.render()
        self.bars.set_values(self.cur, self.peak_push)
        self.plot.update()
        if now - self._gait_t > 0.5:
            self._gait_t = now
            s = self.sim
            self.gait.show_metrics(s.live.metrics(s.mass(), s.torque_limits()), s.base_velocity(), s.cmd,
                                   s.cond.label(s.nom.friction) + (", sensor noise" if s.env.cfg.obs_noise else ""))
        self._status()

    def _policy_step(self):
        if self.pending_push is not None and self.sim.data.time >= self.pending_push:
            self.pending_push = None
            self.apply_push()
        taus, flags = self.sim.step()
        dt = self.sim.model.opt.timestep
        t_end = self.sim.data.time
        fallen = self.sim.fallen()
        c = self.sim.cond
        for k in range(len(taus)):
            t = t_end - (len(taus) - 1 - k) * dt
            self.plot.add(t, taus[k], bool(flags[k]))
            if self.record is not None:
                f = self.sim.push_force if flags[k] else np.zeros(3)
                self.record.append([round(t, 4), *self.sim.cmd, *f, int(fallen), *taus[k],
                                    c.friction, c.payload, c.strength])
        self.cur = np.abs(taus[-1])
        self.peak_push = np.maximum(self.peak_push, np.abs(taus).max(axis=0))
        if self.ramp:
            self._ramp_step()
        elif fallen:
            self.fallen_since = self.fallen_since or t_end
            if self.auto_reset.isChecked() and t_end - self.fallen_since > 1.0:
                self.reset()
        else:
            self.fallen_since = None

    def _status(self):
        s = self.sim
        v = s.base_velocity()
        state = ("EVALUATING (live robot paused)" if self.eval_worker is not None
                 else "FELL: press Reset (Backspace)" if s.fallen() and not self.ramp
                 else "PUSHING" if s.push_left > 0 else "running" if self.running else "paused")
        self.status.setText(
            f"t {s.data.time:6.2f} s   |   {state}   |   base height {s.data.qpos[2]:.3f} m   tilt "
            f"{s.tilt_deg():4.1f}°   |   cmd vx {s.cmd[0]:+.2f}  vy {s.cmd[1]:+.2f}  wz {s.cmd[2]:+.2f}"
            f"   |   actual vx {v[0]:+.2f}  vy {v[1]:+.2f}   |   {s.cond.label(s.nom.friction)}"
            + ("   |   ● REC" if self.record is not None else ""))

    def closeEvent(self, e):
        if self.eval_worker is not None:
            self.eval_worker.stop()
            self.eval_thread.quit()
            self.eval_thread.wait(5000)
        super().closeEvent(e)


# ====================================================================== headless evaluation
def run_headless(args):
    from oc1_rl.paths import resolve_policy
    path = Path(args.policy) if Path(args.policy).is_file() else resolve_policy(args.policy)
    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    last = [0.0, ""]

    def prog(f, msg):
        if f - last[0] >= 0.02 or msg != last[1] or f >= 1.0:
            last[:] = [f, msg]
            print(f"\r[{100 * f:5.1f}%] {msg:60s}", end="", file=sys.stderr, flush=True)

    t0 = time.perf_counter()
    suite = EvalSuite(path, preset=args.eval, tests=tests, push_duration=args.push_duration,
                      heading_hold=args.heading_hold, obs_noise=args.obs_noise,
                      nthread=args.threads or None, seed=args.seed, progress=prog)
    R = suite.run()
    R["nominal_friction"] = suite.nom.friction
    png, js = save_results(R, args.out)
    print(f"\n\npolicy {path}  ({args.eval} preset, {time.perf_counter() - t0:.0f} s)\n")
    for lab, val in R["headline"]:
        print(f"  {lab:58s} {val}")
    print("\nfindings:")
    for f in R["findings"]:
        print(f"  {f['severity'].upper():5s} [{f['test']}] {f['title']}")
        for x in f["fixes"][:1]:
            print(textwrap.indent(textwrap.fill("→ " + x, 100), "        "))
    print(f"\nreport: {png}\nresults: {js}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--policy", default="latest", help="ONNX path, run dir or 'latest'")
    ap.add_argument("--eval", choices=sorted(PRESETS), help="run the evaluation without a window, then exit")
    ap.add_argument("--tests", default=",".join(OPTIONAL_TESTS),
                    help="optional tests for --eval (walking always runs): " + ",".join(OPTIONAL_TESTS))
    ap.add_argument("--push-duration", type=float, default=0.2)
    ap.add_argument("--heading-hold", action="store_true", help="evaluate with heading hold (as deployed)")
    ap.add_argument("--obs-noise", action="store_true", help="evaluate with training-level sensor noise")
    ap.add_argument("--threads", type=int, default=0, help="physics threads (default: all cores)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="report folder (default eval_reports/)")
    args = ap.parse_args()
    if args.eval:
        run_headless(args)
        return
    app = QtWidgets.QApplication(sys.argv)
    win = Lab(args.policy)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()