"""Check that the base frame in oc1_rl/robot.py is x forward, y left, z up.

Run from the repo root:   python scripts/check_frame.py
"""
import sys
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oc1_rl import robot  # noqa: E402

model = robot.make_model(visual=False)
data = mujoco.MjData(model)
data.qpos[3] = 1.0                      # base upright
data.qpos[7:] = robot.DEFAULT_JOINT_POS  # home pose
mujoco.mj_kinematics(model, data)

left = data.site_xpos[model.site("left_foot").id]
right = data.site_xpos[model.site("right_foot").id]
print(f"left foot  x={left[0]:+.3f}  y={left[1]:+.3f}  z={left[2]:+.3f}")
print(f"right foot x={right[0]:+.3f}  y={right[1]:+.3f}  z={right[2]:+.3f}")

if left[1] > 0.2 and right[1] < -0.2 and abs(left[0]) < 0.1 and abs(right[0]) < 0.1:
    print("OK: base frame is x forward, y left, z up. Safe to train.")
else:
    print("WRONG: base frame is rotated. Check _HIP_CENTRE and _TORSO_RPY in oc1_rl/robot.py.")