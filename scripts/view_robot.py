#!/usr/bin/env python3
"""
view_robot.py - look at the OC1 model in the MuJoCo viewer, exactly as training builds it
(urdf/oc1_bipedal.urdf through oc1_rl/robot.py: foot spheres, actuators, joint limits).

  python scripts/view_robot.py            # hanging in the air, home pose
  python scripts/view_robot.py --zero     # hanging, every joint at 0 (the URDF's zero pose)
  python scripts/view_robot.py --stand    # standing on the floor, joints held at the home pose

Plain `python` works on macOS (this uses mujoco.viewer.launch, not launch_passive).

In the viewer:
  * right panel > Control: one slider per joint = its target angle (rad). Drag to move it.
  * left panel > Rendering > Geom groups: group 3 shows collision meshes + foot spheres.
  * left panel > Rendering > Frame: "Body" shows link frames; "Joint" shows joint axes.
  * Space pauses; Backspace resets; the Key slider in the Simulation section loads the
    "home" / "zero" poses.
"""
import argparse
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oc1_rl import robot  # noqa: E402

HOLD_KP, HOLD_KD = 400.0, 10.0   # stiff hold so the pose stays visible without a policy
HANG_HEIGHT = 1.2


def build(stand):
    spec = robot.make_spec(visual=True)
    if not stand:
        # weld the base to the world so the robot hangs still while you move its joints
        spec.body(robot.BASE_BODY).pos = [0.0, 0.0, HANG_HEIGHT]
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.name1 = robot.BASE_BODY
        eq.solref = [0.004, 1.0]   # stiff weld (default is soft and lets the robot sag)
    model = spec.compile()
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.timestep = 0.002
    model.actuator_gainprm[:, 0] = HOLD_KP
    model.actuator_biasprm[:, 1] = -HOLD_KP
    model.actuator_biasprm[:, 2] = -HOLD_KD
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--stand", action="store_true", help="free base on the floor instead of hanging")
    ap.add_argument("--zero", action="store_true", help="start with every joint at 0, not the home pose")
    args = ap.parse_args()

    model = build(args.stand)
    names = [model.joint(i).name for i in range(1, model.njnt)]
    if tuple(names) != tuple(robot.JOINT_NAMES):
        print("WARNING: the URDF's joint order differs from JOINT_NAMES in oc1_rl/robot.py,\n"
              "         so train.py / play.py will stop at robot.make_model(). Rebuild the URDF\n"
              "         (build_urdf.py, FIRST_LEG) or change JOINT_NAMES.\n")
    home = dict(zip(robot.JOINT_NAMES, robot.DEFAULT_JOINT_POS))   # matched by name
    pose = np.array([0.0 if args.zero else home[n] for n in names])

    data = mujoco.MjData(model)
    data.qpos[3] = 1.0
    data.qpos[2] = robot.init_base_height(model) if args.stand else HANG_HEIGHT
    data.qpos[7:] = pose
    data.ctrl[:] = pose
    mujoco.mj_forward(model, data)

    print("joint            target (rad)")
    for n, q in zip(names, pose):
        print(f"  {n:18s} {q:+.4f}")
    print("\nControl panel sliders set these targets. Close the window to quit.")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()