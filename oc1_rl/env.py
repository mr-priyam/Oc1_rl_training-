"""Batched OC1 velocity-tracking environment on CPU MuJoCo.

Same task as g1_rl/env.py (unitree_rl_mjlab's `Unitree-G1-Flat`: observations, rewards,
commands, events, terminations) with numpy, stepping all environments in C
threads via `mujoco.rollout`.
"""

import copy
from dataclasses import dataclass, field

import mujoco
import numpy as np
from mujoco import rollout

from . import robot

STEP_DT = 0.02  # Policy period (timestep 0.005 x decimation 4).
DECIMATION = 4
EPISODE_LENGTH_S = 20.0
GAIT_PERIOD = 0.6


@dataclass
class EnvCfg:
  num_envs: int = 2048
  nthread: int = 8
  seed: int = 0
  # Training-time features; turned off for evaluation/playback.
  obs_noise: bool = True
  domain_randomization: bool = True
  pushes: bool = True
  terminate_on_timeout: bool = True
  # Command sampling (UniformVelocityCommandCfg + commands_vel curriculum).
  resampling_time_range: tuple = (3.0, 8.0)
  rel_standing_envs: float = 0.05
  rel_heading_envs: float = 1.0
  heading_control_stiffness: float = 0.5
  velocity_stages: list = field(default_factory=lambda: [
    {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
    {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)},
  ])


# Reward weights (velocity_env_cfg.py + G1 overrides). Rewards are scaled by STEP_DT.
REWARD_WEIGHTS = {
  "track_linear_velocity": 1.0,
  "track_angular_velocity": 1.0,
  "body_orientation_l2": -1.0,
  "pose": 1.0,
  "body_ang_vel": -0.05,
  "angular_momentum": -0.025,
  "is_terminated": -200.0,
  "joint_acc_l2": -2.5e-7,
  "joint_pos_limits": -10.0,
  "action_rate_l2": -0.05,
  "foot_gait": 0.5,
  "foot_clearance": -1.0,
  "foot_slip": -0.25,
  "soft_landing": -1e-3,
  "stand_still": -1.0,
  "self_collisions": -1.0,
}

# Actor observation noise (uniform, +/-).
_NOISE = {"ang_vel": 0.2, "gravity": 0.05, "joint_pos": 0.01, "joint_vel": 1.5}

PUSH_INTERVAL_S = (5.0, 6.0)
PUSH_VEL = np.array([0.5, 0.5, 0.4, 0.52, 0.52, 0.78])  # x y z roll pitch yaw (world)

NUM_ACTOR_OBS = 3 + 3 + 3 + 2 + 3 * robot.NUM_JOINTS
NUM_CRITIC_OBS = NUM_ACTOR_OBS + 3 + 2 + 2 + 2 + 6


def quat_rotate_inverse(q, v):
  w, u = q[:, :1], q[:, 1:]
  return v * (2 * w**2 - 1) - 2 * w * np.cross(u, v) + 2 * u * np.sum(u * v, 1, keepdims=True)


def quat_rotate(q, v):
  w, u = q[:, :1], q[:, 1:]
  return v * (2 * w**2 - 1) + 2 * w * np.cross(u, v) + 2 * u * np.sum(u * v, 1, keepdims=True)


def yaw_of(q):
  w, x, y, z = q.T
  return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap_to_pi(a):
  return (a + np.pi) % (2 * np.pi) - np.pi


class OC1VelocityEnv:
  def __init__(self, cfg: EnvCfg, model: mujoco.MjModel | None = None):
    self.cfg = cfg
    n = self.num_envs = cfg.num_envs
    self.rng = np.random.default_rng(cfg.seed)

    self.model = model if model is not None else robot.make_model(visual=False)
    m = self.model
    self.timestep = m.opt.timestep
    assert abs(self.timestep * DECIMATION - STEP_DT) < 1e-9
    self.max_episode_steps = int(round(EPISODE_LENGTH_S / STEP_DT))

    # Full-physics state layout: [time, qpos, qvel, act, ...].
    self.state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    self.state_size = mujoco.mj_stateSize(m, self.state_spec)
    self.qpos_slice = slice(1, 1 + m.nq)
    self.qvel_slice = slice(1 + m.nq, 1 + m.nq + m.nv)
    self._check_state_layout()

    s = {m.sensor(i).name: i for i in range(m.nsensor)}
    self._sens = {k: slice(m.sensor_adr[i], m.sensor_adr[i] + m.sensor_dim[i]) for k, i in s.items()}

    self.default_q = robot.DEFAULT_JOINT_POS.copy()
    self.soft_limits = robot.soft_joint_limits(m)
    self.action_scale = robot.ACTION_SCALE.copy()
    nj = robot.NUM_JOINTS

    # Per-env model copies carry the startup domain randomization.
    self.models = self._make_models() if cfg.domain_randomization else [m] * n
    self.encoder_bias = (
      self.rng.uniform(-0.015, 0.015, (n, nj)) if cfg.domain_randomization else np.zeros((n, nj))
    )

    self.pool = rollout.Rollout(nthread=cfg.nthread)
    self.datas = [mujoco.MjData(m) for _ in range(max(cfg.nthread, 1))]

    # Buffers.
    self.state = np.zeros((n, self.state_size))
    self.sensordata = np.zeros((n, m.nsensordata))
    self._states_out = np.zeros((n, DECIMATION, self.state_size))
    self._sens_out = np.zeros((n, DECIMATION, m.nsensordata))
    self.action = np.zeros((n, nj))
    self.prev_action = np.zeros((n, nj))
    self.episode_len = np.zeros(n, dtype=np.int64)
    self.common_step = 0
    self.air_time = np.zeros((n, 2))
    self.contact_prev = np.zeros((n, 2), dtype=bool)
    self.command = np.zeros((n, 3))
    self.command_override = None  # (3,) array to force a command (playback).
    self.cmd_timer = np.zeros(n)
    self.heading_target = np.zeros(n)
    self.is_heading_env = np.zeros(n, dtype=bool)
    self.is_standing_env = np.zeros(n, dtype=bool)
    self.push_timer = self.rng.uniform(*PUSH_INTERVAL_S, n)
    self.ranges = dict(self.cfg.velocity_stages[0])
    self.episode_sums = {k: np.zeros(n) for k in REWARD_WEIGHTS}

    self._init_state = np.zeros(self.state_size)
    d = mujoco.MjData(m)
    d.qpos[2] = robot.init_base_height(m)
    d.qpos[3] = 1.0
    d.qpos[7:] = self.default_q
    mujoco.mj_getState(m, d, self._init_state, self.state_spec)

    self.reset_envs(np.arange(n))
    self._refresh_sensors(np.arange(n))

  # ------------------------------------------------------------------ setup

  def _check_state_layout(self):
    m = self.model
    d = mujoco.MjData(m)
    d.time = 0.5
    d.qpos[:] = np.arange(m.nq) + 1.0
    d.qvel[:] = -(np.arange(m.nv) + 1.0)
    st = np.zeros(self.state_size)
    mujoco.mj_getState(m, d, st, self.state_spec)
    assert st[0] == 0.5
    assert np.array_equal(st[self.qpos_slice], d.qpos)
    assert np.array_equal(st[self.qvel_slice], d.qvel)

  def _make_models(self):
    m = self.model
    foot_geoms = [
      i for i in range(m.ngeom) if robot.FOOT_GEOM_RE.match(m.geom(i).name or "")
    ]
    torso = m.body(robot.BASE_BODY).id
    models = []
    for _ in range(self.num_envs):
      mi = copy.copy(m)
      mi.geom_friction[foot_geoms, 0] = self.rng.uniform(0.3, 1.6)
      mi.body_ipos[torso] = m.body_ipos[torso] + self.rng.uniform(-0.05, 0.05, 3)
      models.append(mi)
    return models

  # ------------------------------------------------------------------ helpers

  def qpos(self, state=None):
    return (self.state if state is None else state)[..., self.qpos_slice]

  def qvel(self, state=None):
    return (self.state if state is None else state)[..., self.qvel_slice]

  def sensor(self, name, data=None):
    return (self.sensordata if data is None else data)[..., self._sens[name]]

  def _refresh_sensors(self, ids):
    """Compute sensors for freshly reset envs (rollout only returns them after stepping)."""
    d = self.datas[0]
    for i in ids:
      mujoco.mj_setState(self.models[i], d, self.state[i], self.state_spec)
      mujoco.mj_forward(self.models[i], d)
      self.sensordata[i] = d.sensordata

  # ------------------------------------------------------------------ commands

  def _update_ranges(self):
    for stage in self.cfg.velocity_stages:
      if self.common_step > stage["step"]:
        self.ranges.update({k: v for k, v in stage.items() if k != "step"})

  def _resample_commands(self, ids):
    if len(ids) == 0:
      return
    k = len(ids)
    r = self.rng
    cmd = np.stack([
      r.uniform(*self.ranges["lin_vel_x"], k),
      r.uniform(*self.ranges["lin_vel_y"], k),
      r.uniform(*self.ranges["ang_vel_z"], k),
    ], axis=1)
    cmd *= (np.linalg.norm(cmd, axis=1) > 0.1)[:, None]
    self.command[ids] = cmd
    self.heading_target[ids] = r.uniform(-np.pi, np.pi, k)
    self.is_heading_env[ids] = r.uniform(0, 1, k) <= self.cfg.rel_heading_envs
    self.is_standing_env[ids] = r.uniform(0, 1, k) <= self.cfg.rel_standing_envs
    self.cmd_timer[ids] = r.uniform(*self.cfg.resampling_time_range, k)

  def _update_commands(self):
    self.cmd_timer -= STEP_DT
    self._resample_commands(np.nonzero(self.cmd_timer <= 0)[0])
    heading = yaw_of(self.qpos()[:, 3:7])
    err = wrap_to_pi(self.heading_target - heading)
    h = self.is_heading_env
    self.command[h, 2] = np.clip(
      self.cfg.heading_control_stiffness * err[h], *self.ranges["ang_vel_z"]
    )
    self.command[self.is_standing_env] = 0.0
    if self.command_override is not None:
      self.command[:] = self.command_override

  # ------------------------------------------------------------------ reset

  def reset_envs(self, ids):
    if len(ids) == 0:
      return
    k = len(ids)
    st = np.tile(self._init_state, (k, 1))
    qpos = st[:, self.qpos_slice]
    qpos[:, 0:2] = self.rng.uniform(-0.5, 0.5, (k, 2))
    yaw = self.rng.uniform(-3.14, 3.14, k)
    qpos[:, 3] = np.cos(yaw / 2)
    qpos[:, 6] = np.sin(yaw / 2)
    self.state[ids] = st

    self.action[ids] = 0.0
    self.prev_action[ids] = 0.0
    self.episode_len[ids] = 0
    self.air_time[ids] = 0.0
    self.contact_prev[ids] = False
    self._resample_commands(ids)
    self._update_commands()

  # ------------------------------------------------------------------ observations

  def _phase(self):
    phase = (self.episode_len * STEP_DT) % GAIT_PERIOD / GAIT_PERIOD
    out = np.stack([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], axis=1)
    out[np.linalg.norm(self.command, axis=1) < 0.1] = 0.0
    return out

  def observations(self):
    qpos, qvel = self.qpos(), self.qvel()
    quat = qpos[:, 3:7]
    n = self.num_envs
    ang_vel = qvel[:, 3:6].copy()  # Body-frame angular velocity (== base IMU gyro).
    gravity = quat_rotate_inverse(quat, np.tile([0.0, 0.0, -1.0], (n, 1)))
    joint_pos = qpos[:, 7:] - self.default_q
    joint_vel = qvel[:, 6:].copy()
    phase = self._phase()

    clean = [ang_vel, gravity, self.command, phase, joint_pos, joint_vel, self.action]
    if self.cfg.obs_noise:
      actor = [
        ang_vel + self.rng.uniform(-_NOISE["ang_vel"], _NOISE["ang_vel"], (n, 3)),
        gravity + self.rng.uniform(-_NOISE["gravity"], _NOISE["gravity"], (n, 3)),
        self.command, phase,
        joint_pos + self.encoder_bias
        + self.rng.uniform(-_NOISE["joint_pos"], _NOISE["joint_pos"], joint_pos.shape),
        joint_vel + self.rng.uniform(-_NOISE["joint_vel"], _NOISE["joint_vel"], joint_vel.shape),
        self.action,
      ]
    else:
      actor = clean

    lin_vel_b = quat_rotate_inverse(quat, qvel[:, 0:3])
    foot_h = np.stack([self.sensor("left_foot_pos")[:, 2], self.sensor("right_foot_pos")[:, 2]], 1)
    contact = self._foot_contact()
    forces = np.concatenate([self.sensor("left_foot_contact")[:, 1:4],
                             self.sensor("right_foot_contact")[:, 1:4]], axis=1)
    critic = clean + [lin_vel_b, foot_h, self.air_time, contact.astype(np.float64),
                      np.sign(forces) * np.log1p(np.abs(forces))]
    return (np.concatenate(actor, axis=1).astype(np.float32),
            np.concatenate(critic, axis=1).astype(np.float32))

  def _foot_contact(self, sens=None):
    return np.stack([self.sensor("left_foot_contact", sens)[:, 0] > 0,
                     self.sensor("right_foot_contact", sens)[:, 0] > 0], axis=1)

  # ------------------------------------------------------------------ step

  def step(self, actions):
    n = self.num_envs
    self.prev_action = self.action
    self.action = np.asarray(actions, dtype=np.float64).copy()
    target = self.default_q + self.action * self.action_scale
    ctrl = np.repeat(target[:, None, :], DECIMATION, axis=1)

    states, sens = self.pool.rollout(
      self.models, self.datas, self.state, ctrl, nstep=DECIMATION, skip_checks=True,
      state=self._states_out, sensordata=self._sens_out,
    )
    self.state = states[:, -1].copy()
    self.sensordata = sens[:, -1].copy()
    self.episode_len += 1
    self.common_step += 1
    self._update_ranges()

    qpos, qvel = self.qpos(), self.qvel()
    quat = qpos[:, 3:7]
    gravity = quat_rotate_inverse(quat, np.tile([0.0, 0.0, -1.0], (n, 1)))
    jpos, jvel = qpos[:, 7:], qvel[:, 6:]

    # Terminations.
    unstable = ~np.isfinite(self.state).all(axis=1) | (np.abs(qvel) > 1e3).any(axis=1)
    fell = np.arccos(np.clip(-gravity[:, 2], -1, 1)) > np.radians(70.0)
    terminated = fell | unstable
    time_out = self.episode_len >= self.max_episode_steps
    if not self.cfg.terminate_on_timeout:
      time_out[:] = False

    # Foot contact bookkeeping (mjlab ContactSensor air/contact time).
    contact = self._foot_contact()
    first_contact = contact & ~self.contact_prev
    self.air_time = np.where(contact, 0.0, self.air_time + STEP_DT)
    self.contact_prev = contact

    # ---- rewards
    cmd = self.command
    cmd_total = np.linalg.norm(cmd[:, :2], axis=1) + np.abs(cmd[:, 2])
    moving = cmd_total > 0.1
    lin_vel_b = quat_rotate_inverse(quat, qvel[:, 0:3])
    ang_vel_b = qvel[:, 3:6]
    torso_g = quat_rotate_inverse(self.sensor("torso_quat"), np.tile([0.0, 0.0, -1.0], (n, 1)))
    foot_pos = np.stack([self.sensor("left_foot_pos"), self.sensor("right_foot_pos")], 1)
    foot_vel_xy = np.linalg.norm(
      np.stack([self.sensor("left_foot_vel"), self.sensor("right_foot_vel")], 1)[..., :2], axis=-1
    )
    foot_force = np.linalg.norm(np.stack([self.sensor("left_foot_contact")[:, 1:4],
                                          self.sensor("right_foot_contact")[:, 1:4]], 1), axis=-1)

    walking = (cmd_total >= 0.1) & (cmd_total < 1.5)
    running = cmd_total >= 1.5
    std = np.where(running[:, None], robot.POSE_STD_RUNNING,
                   np.where(walking[:, None], robot.POSE_STD_WALKING, robot.POSE_STD_STANDING))

    jvel_prev = self.qvel(states[:, -2])[:, 6:]
    joint_acc = (jvel - jvel_prev) / self.timestep

    leg_phase = ((self.episode_len * STEP_DT / GAIT_PERIOD)[:, None] + np.array([0.0, 0.5])) % 1.0
    self_force = np.linalg.norm(self.sensor("self_contact", sens)[..., 1:4], axis=-1)  # (n, 4)

    lin_err = np.sum((cmd[:, :2] - lin_vel_b[:, :2]) ** 2, 1) + 2 * lin_vel_b[:, 2] ** 2
    ang_err = (cmd[:, 2] - ang_vel_b[:, 2]) ** 2 + 0.05 * np.sum(ang_vel_b[:, :2] ** 2, 1)
    terms = {
      "track_linear_velocity": np.exp(-lin_err / 0.25),
      "track_angular_velocity": np.exp(-ang_err / 0.5),
      "body_orientation_l2": np.sum(torso_g[:, :2] ** 2, 1),
      "pose": np.exp(-np.mean((jpos - self.default_q) ** 2 / std**2, 1)),
      "body_ang_vel": np.sum(self.sensor("torso_angvel")[:, :2] ** 2, 1),
      "angular_momentum": np.sum(self.sensor("root_angmom") ** 2, 1),
      "is_terminated": terminated.astype(np.float64),
      "joint_acc_l2": np.sum(joint_acc**2, 1),
      "joint_pos_limits": np.sum(np.clip(self.soft_limits[:, 0] - jpos, 0, None)
                                 + np.clip(jpos - self.soft_limits[:, 1], 0, None), 1),
      "action_rate_l2": np.sum((self.action - self.prev_action) ** 2, 1),
      "foot_gait": np.mean((leg_phase < 0.56) == contact, 1) * moving,
      "foot_clearance": np.sum(np.abs(foot_pos[..., 2] - 0.10) * foot_vel_xy, 1) * moving,
      "foot_slip": np.sum(foot_vel_xy**2 * contact, 1) * moving,
      "soft_landing": np.sum(foot_force * first_contact, 1) * moving,
      "stand_still": np.sum((jpos - self.default_q) ** 2, 1) * ~moving,
      "self_collisions": np.sum(self_force > 10.0, 1).astype(np.float64),
    }
    reward = np.zeros(n)
    for k, v in terms.items():
      v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
      if k != "is_terminated":
        v = np.where(unstable, 0.0, v)
      r = REWARD_WEIGHTS[k] * v * STEP_DT
      reward += r
      self.episode_sums[k] += r

    # ---- resets
    done = terminated | time_out
    ids = np.nonzero(done)[0]
    extras = {"episode": {}}
    if len(ids):
      for k, v in self.episode_sums.items():
        extras["episode"][k] = float(np.mean(v[ids]) / EPISODE_LENGTH_S)
        v[ids] = 0.0
      extras["episode_length"] = float(np.mean(self.episode_len[ids]))
      extras["fell_frac"] = float(np.mean(terminated[ids]))
      self.reset_envs(ids)
      self._refresh_sensors(ids)

    self._update_commands()

    if self.cfg.pushes:
      self.push_timer -= STEP_DT
      push = np.nonzero(self.push_timer <= 0)[0]
      if len(push):
        self.push_timer[push] = self.rng.uniform(*PUSH_INTERVAL_S, len(push))
        qv = self.qvel()
        dv = self.rng.uniform(-PUSH_VEL, PUSH_VEL, (len(push), 6))
        quat = self.qpos()[push, 3:7]
        ang_w = quat_rotate(quat, qv[push, 3:6]) + dv[:, 3:]
        qv[push, 0:3] += dv[:, :3]
        qv[push, 3:6] = quat_rotate_inverse(quat, ang_w)

    obs, critic_obs = self.observations()
    return obs, critic_obs, reward.astype(np.float32), done, time_out, extras

  def reset(self):
    return self.observations()

  def close(self):
    self.pool.close()
