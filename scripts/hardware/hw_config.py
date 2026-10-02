"""hw_config.py - everything about YOUR robot that bringup.py needs.

Fill in the lines marked  <-- CHECK  before the first run.
bringup.py refuses to move any motor while a motor id is still None.

Units: radians, rad/s, N.m, amps.
"""

import math

# ------------------------------------------------------------------ CAN
# "socketcan": Raspberry Pi CAN hub (can0..can4).  Bring each channel up first:
#     sudo ip link set can0 up type can bitrate 1000000
# "usbcan":    RobStride USB-CAN dongle (one channel).  Needs usbcan.py next to
#              bringup.py (copy it from robstride-can-hub-with-rpi) and pyserial.
CAN_BACKEND = "socketcan"
USBCAN_PORT = None            # None = find the CH340 automatically
HOST_ID = 0xFD

# ---------------------------------------------------------------- motors
# One line per joint, in the POLICY's order (do not reorder).
#   channel : which CAN port the motor is on
#   id      : the motor's CAN id  -> run discover.py and write what it prints   <-- CHECK
#   model   : "RS04" (hip pitch, hip roll, knee) or "RS03" (hip yaw, ankle)
#   sign    : +1 or -1.  Set by step "Joint order and direction" if it is wrong  <-- CHECK
#   zero    : motor reading (rad) when the joint is at its URDF zero.
#             Step "Zero check" measures this and saves it to hw_offsets.json,
#             which overrides the number here.
MOTORS = {
  "right_hip_pitch":   dict(channel="can0", id=None, model="RS04", sign=+1, zero=0.0),
  "right_hip_roll":    dict(channel="can0", id=None, model="RS04", sign=+1, zero=0.0),
  "right_hip_yaw":     dict(channel="can0", id=None, model="RS03", sign=+1, zero=0.0),
  "right_knee_pitch":  dict(channel="can1", id=None, model="RS04", sign=+1, zero=0.0),
  "right_ankle_pitch": dict(channel="can1", id=None, model="RS03", sign=+1, zero=0.0),
  "left_hip_pitch":    dict(channel="can2", id=None, model="RS04", sign=+1, zero=0.0),
  "left_hip_roll":     dict(channel="can2", id=None, model="RS04", sign=+1, zero=0.0),
  "left_hip_yaw":      dict(channel="can2", id=None, model="RS03", sign=+1, zero=0.0),
  "left_knee_pitch":   dict(channel="can3", id=None, model="RS04", sign=+1, zero=0.0),
  "left_ankle_pitch":  dict(channel="can3", id=None, model="RS03", sign=+1, zero=0.0),
}

# Motor data sheet numbers (feedback-frame scaling + ratings).  <-- CHECK against your manuals
MOTOR_SPECS = {
  "RS04": dict(p_max=4 * math.pi, v_max=15.0, t_max=120.0, kp_max=5000.0, kd_max=100.0,
               rated=40.0),
  "RS03": dict(p_max=4 * math.pi, v_max=20.0, t_max=60.0, kp_max=5000.0, kd_max=100.0,
               rated=20.0),
}

# ------------------------------------------------------------ control mode
# "csp": position mode (run_mode 5) + current limit.  This is what your
#        robstride scripts already use and have tested.  The motor's own position
#        loop is NOT the same as the Kp/Kd the policy was trained with, so walking
#        may feel stiffer/softer than in sim.
# "mit": MIT mode (run_mode 0, type-1 frames) with the SAME Kp/Kd as the sim.
#        Closest to training, but not yet tested on your rig - try it only after
#        every "csp" step passes.
CONTROL_MODE = "csp"

# Gains used in "mit" mode (same as oc1_rl/robot.py).
KP = {"RS04": 157.91, "RS03": 78.96}
KD = {"RS04": 10.05, "RS03": 5.03}

# Current limits (A).  Low for the bench steps, higher for the policy steps.
CURRENT_LIMIT_BENCH = {"RS04": 3.0, "RS03": 2.0}
CURRENT_LIMIT_POLICY = {"RS04": 12.0, "RS03": 8.0}      # <-- CHECK: raise slowly
SPEED_LIMIT = 2.0              # rad/s, csp mode (motor enforces it)

# Motor watchdog: motor stops by itself if it hears nothing for this long.
# 20000 is ~1 s according to your jog scripts (not measured yet).  Never 0.
CAN_TIMEOUT_RAW = 20000

# --------------------------------------------------------------- safety
JOINT_LIMIT = 0.5              # URDF limit, rad (all joints)
JOINT_TARGET_LIMIT = 0.45      # targets are clamped to +/- this
MAX_TEMP_C = 70.0
TILT_SOFT_DEG = 45.0           # policy stops, motors hold home
TILT_CUT_DEG = 60.0            # motors disabled
MISSED_FRAMES_STOP = 3         # 3 policy steps (60 ms) without feedback -> stop

# ------------------------------------------------------------------ IMU
# "none":   no IMU yet - the IMU and policy steps are skipped.
# "custom": edit read_imu() below to return your IMU's numbers.
IMU_TYPE = "custom"              # <-- CHECK

# Rotation from the IMU's axes to the robot base axes (x forward, y left, z up).
# Identity = IMU mounted with x forward, y left, z up.
IMU_TO_BASE = [[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]]


def open_imu():
  """Open your IMU here and return any object (passed to read_imu)."""
  raise NotImplementedError("write open_imu() in hw_config.py for your IMU")


def read_imu(handle):
  """Return (gyro, quat) in the IMU's own axes:
       gyro: [wx, wy, wz] rad/s
       quat: [w, x, y, z] orientation of the IMU in the world (z up)
  Example for a serial IMU: parse the newest packet and return the two lists."""
  raise NotImplementedError("write read_imu() in hw_config.py for your IMU")


# ---------------------------------------------------------------- policy
POLICY = "runs/2026-10-01_20-43-17/policy.onnx"
WALK_TEST_SPEEDS = [0.1, 0.2, 0.3]     # m/s, gantry walking step
