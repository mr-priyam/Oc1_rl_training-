#!/usr/bin/env python3
"""
show_foot_tilt.py - SEE whether each foot is level when every joint is at 0.

Loads urdf/oc1_bipedal.urdf directly into MuJoCo (none of the other project scripts are
used), holds the torso fixed and puts every joint at 0, then draws on each foot:

  GREEN line + plate : perfectly level (fixed to the world)
  RED   line + plate : along the sole (fixed to the foot, moves with it)

The two lines cross at the middle of the sole and run 0.5 m to each side, so a small tilt
shows up as an X: a level foot = the lines lie on top of each other.

  python show_foot_tilt.py              # writes foot_tilt.png (side views, orthographic)
  python show_foot_tilt.py --view       # also opens the MuJoCo viewer ([ and ] switch cameras)
  python show_foot_tilt.py --offset 0.0296   # also show the left foot with that ankle offset

Works from the repo root or from scripts/ (it reads urdf/oc1_bipedal.urdf and meshes/).
Sole geometry (from oc1_rl/robot.py): in the foot frame the sole is the plane z = -0.105,
centred at x = -0.065, y = -0.032, toe towards -y.
"""
import argparse
import math
import re
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next((d for d in (HERE, HERE.parent, Path.cwd()) if (d / "urdf" / "oc1_bipedal.urdf").is_file()), None)
if ROOT is None:
    raise SystemExit("urdf/oc1_bipedal.urdf not found: run this from the Oc1_rl_training repo root")
URDF, MESHES = ROOT / "urdf" / "oc1_bipedal.urdf", ROOT / "meshes"
SOLE = np.array([-0.065, -0.032, -0.105])   # sole centre in the foot frame
TOE = np.array([0.0, -1.0, 0.0])            # toe direction in the foot frame
LINE_HALF = 0.5                              # each reference line runs 0.5 m both ways
GREEN, RED = [0.05, 0.65, 0.15, 0.9], [0.85, 0.1, 0.1, 0.9]


def load_spec():
    xml = URDF.read_text()
    xml = re.sub(r'filename="package://[^"]*/meshes/', 'filename="', xml)
    # discardvisual=false: keep visual-only geoms (the URDF visuals and the reference lines)
    xml = re.sub(r"(<robot[^>]*>)",
                 rf'\1<mujoco><compiler meshdir="{MESHES}" discardvisual="false"/></mujoco>', xml, count=1)
    spec = mujoco.MjSpec.from_string(xml)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[1, 1, 1], rgb2=[0.85, 0.87, 0.9],
                     width=256, height=1536)
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]
    return spec


def foot_poses(spec, ankle_offset):
    m = spec.copy().compile()
    d = mujoco.MjData(m)
    if ankle_offset:
        d.qpos[m.jnt_qposadr[m.joint("left_ankle_pitch").id]] = ankle_offset
    mujoco.mj_kinematics(m, d)
    out = {}
    for foot in ("lf", "rf"):
        R = d.xmat[m.body(foot).id].reshape(3, 3)
        p = d.xpos[m.body(foot).id] + R @ SOLE
        pitch = math.degrees(math.atan2(R[0, 2], R[2, 2]))
        out[foot] = (p, R, pitch)
    return out


GROUP = {"lf": 4, "rf": 5}   # each foot's reference lines in its own display group


def decorate(spec, poses, view):
    """Green = level reference in the world, red = sole-aligned, attached to the foot."""
    for foot, (p, R, _) in poses.items():
        grp = 2 if view else GROUP[foot]   # viewer shows groups 0-2 by default
        toe_w = R @ TOE
        toe_w[2] = 0.0
        toe_w /= np.linalg.norm(toe_w)                       # level direction along the foot
        yaw = math.atan2(toe_w[1], toe_w[0])
        q_level = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        wb = spec.worldbody
        side = 1.0 if p[1] > 0 else -1.0
        normal = np.array([-toe_w[1], toe_w[0], 0.0]) * side    # horizontal, pointing outward
        back = -0.01 * normal        # green sits 1 cm behind the red, so red is drawn on top
        wb.add_geom(name=f"{foot}_level_line", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.003, 0, 0],
                    fromto=list(p + back - LINE_HALF * toe_w) + list(p + back + LINE_HALF * toe_w),
                    rgba=GREEN, contype=0, conaffinity=0, group=grp)
        wb.add_geom(name=f"{foot}_level_plate", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.12, 0.08, 0.0005],
                    pos=list(p - [0, 0, 0.002]), quat=list(q_level), rgba=[0.05, 0.65, 0.15, 0.35],
                    contype=0, conaffinity=0, group=grp)
        b = spec.body(foot)
        b.add_geom(name=f"{foot}_sole_line", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.0015, 0, 0],
                   fromto=list(SOLE - LINE_HALF * TOE) + list(SOLE + LINE_HALF * TOE), rgba=RED,
                   contype=0, conaffinity=0, group=grp, mass=0)
        b.add_geom(name=f"{foot}_sole_plate", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.08, 0.12, 0.0005],
                   pos=list(SOLE), rgba=[0.85, 0.1, 0.1, 0.35], contype=0, conaffinity=0, group=grp, mass=0)
        # side camera on the outer side of the foot, looking across it (orthographic = no perspective)
        cam = wb.add_camera(name=f"{foot}_side", pos=list(p + 1.5 * normal))
        xaxis = np.cross([0, 0, 1.0], normal)                       # screen-right
        Rc = np.column_stack([xaxis, [0.0, 0.0, 1.0], normal])        # camera looks along -z
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, Rc.ravel())
        cam.quat = list(q)
        try:
            cam.orthographic = True
            cam.fovy = 0.45                                         # 0.45 m tall view
        except AttributeError:
            cam.fovy = 18
    return spec


def build(ankle_offset, view=False):
    spec = load_spec()
    poses = foot_poses(spec, ankle_offset)
    spec = decorate(spec, poses, view)
    m = spec.compile()
    m.vis.global_.offwidth, m.vis.global_.offheight = 1200, 900     # room for the PNG renders
    m.opt.gravity[:] = 0.0                                          # nothing moves by itself
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    d = mujoco.MjData(m)
    if ankle_offset:
        d.qpos[m.jnt_qposadr[m.joint("left_ankle_pitch").id]] = ankle_offset
    mujoco.mj_forward(m, d)
    return m, d, poses


def render(m, d, cam, w=900, h=600):
    r = mujoco.Renderer(m, h, w)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[[0, 1, GROUP[cam.split("_")[0]]]] = 1   # robot + this foot's lines only
    r.update_scene(d, camera=cam, scene_option=opt)
    img = r.render().copy()
    r.close()
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--view", action="store_true", help="also open the interactive MuJoCo viewer")
    ap.add_argument("--offset", type=float, default=0.0296,
                    help="left ankle offset for the third picture (rad, default 0.0296); 0 = skip")
    ap.add_argument("--out", default="foot_tilt.png")
    args = ap.parse_args()

    m0, d0, p0 = build(0.0)
    panels = [(render(m0, d0, "lf_side"), f"LEFT foot, all joints 0:  sole pitch {p0['lf'][2]:+.2f} deg"),
              (render(m0, d0, "rf_side"), f"RIGHT foot, all joints 0:  sole pitch {p0['rf'][2]:+.2f} deg")]
    if args.offset:
        m1, d1, p1 = build(args.offset)
        panels.append((render(m1, d1, "lf_side"),
                       f"LEFT foot, left_ankle_pitch = {args.offset:+.4f} rad:  pitch {p1['lf'][2]:+.2f} deg"))
    for img, title in panels:
        print(title)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.2))
        for ax, (img, title) in zip(np.atleast_1d(axs), panels):
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle("Side view, true scale (orthographic).  GREEN = level reference,  RED = along the sole.  "
                     "Level foot: red lies inside the green.  Tilted foot: the lines cross.", fontsize=10)
        fig.tight_layout()
        fig.savefig(args.out, dpi=120)
        print(f"picture written to {args.out}")
    except ImportError:
        print("matplotlib is not installed (pip install matplotlib) - use --view to see it instead")

    if args.view:
        import mujoco.viewer
        print("viewer: press [ or ] to switch to the lf_side / rf_side cameras")
        mv, dv, _ = build(0.0, view=True)
        mujoco.viewer.launch(mv, dv)


if __name__ == "__main__":
    main()