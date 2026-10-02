"""Manually test OC1 bipedal joint angles in the MuJoCo viewer.

Loads urdf/oc1_bipedal.urdf, pins the base in place, and adds one position
actuator per movable joint so every joint gets a slider in the viewer's
"Control" panel (right side). Drag a slider to command that joint's angle.

Usage:
    .venv/bin/python scripts/joint_test.py              # physics, no gravity
    .venv/bin/python scripts/joint_test.py --gravity    # physics with gravity
    .venv/bin/python scripts/joint_test.py --no-collision   # links pass through each other
    .venv/bin/python scripts/joint_test.py --export model.xml   # save MJCF only
"""

import argparse
import math
import re
from pathlib import Path

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = ROOT / "urdf" / "oc1_bipedal.urdf"
MESH_DIR = ROOT / "meshes"

# Default slider range for "continuous" joints (URDF gives them no limits).
CONTINUOUS_RANGE = (-math.pi, math.pi)


def load_urdf_xml() -> str:
    """Read the URDF and point MuJoCo at the mesh directory.

    MuJoCo cannot resolve ROS "package://" URIs, and by default it drops
    <visual> geometry from URDFs. Keep the meshes for display; the <collision>
    boxes are what actually make contact.
    """
    xml = URDF_PATH.read_text()
    xml = re.sub(r'filename="package://[^"]*/meshes/', 'filename="', xml)
    compiler = (
        "<mujoco>"
        f'<compiler meshdir="{MESH_DIR}" discardvisual="false" '
        'fusestatic="false" balanceinertia="true" strippath="false"/>'
        "</mujoco>"
    )
    return xml.replace('<robot name="oc1_bipedal">', f'<robot name="oc1_bipedal">{compiler}', 1)


def exclude_resting_contacts(spec: mujoco.MjSpec) -> None:
    """Keep self-collision on, except for pairs that can't meaningfully collide.

    - Bodies on either side of a joint (including everything fastened to them)
      always touch at the joint axis. MuJoCo already skips parent/child pairs,
      but not when the parent is welded to the world, as the pinned torso is.
    - Any other pair whose collision boxes already overlap in the zero pose
      would otherwise push apart forever and fight the actuators.
    """
    model = spec.compile()
    weld = model.body_weldid
    groups = {}
    for b in range(1, model.nbody):
        groups.setdefault(weld[b], []).append(b)

    pairs = set()
    for b in range(1, model.nbody):
        if weld[b] == b and model.body_jntnum[b] > 0:
            parent = weld[model.body_parentid[b]]
            for b1 in groups.get(parent, [0]):
                for b2 in groups[b]:
                    pairs.add((min(b1, b2), max(b1, b2)))

    def add(b1, b2):
        spec.add_exclude(bodyname1=model.body(b1).name, bodyname2=model.body(b2).name)

    for b1, b2 in pairs:
        if b1 > 0:
            add(b1, b2)

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    resting = {
        tuple(sorted((model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])))
        for c in data.contact[: data.ncon]
    }
    for b1, b2 in sorted(resting):
        if b1 == 0:
            continue  # touching the floor at rest is a real contact
        print(f"  excluding overlapping pair at rest: {model.body(b1).name} <-> {model.body(b2).name}")
        add(b1, b2)


def build_spec(gravity: bool, kp: float, collisions: bool = True) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_string(load_urdf_xml())

    if not gravity:
        spec.option.gravity = [0, 0, 0]

    # Visual meshes (group 1) never collide; the URDF <collision> boxes
    # (group 0) collide with each other and with the floor.
    for geom in spec.geoms:
        if geom.group != 0 or not collisions:
            geom.contype = 0
            geom.conaffinity = 0

    # Some damping so the joints settle instead of oscillating.
    for joint in spec.joints:
        if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
            joint.damping[0] = 0.5  # damping is a vector in recent MuJoCo
            joint.armature = 0.01

    # One position actuator per hinge -> one slider each in the viewer.
    for joint in spec.joints:
        if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        has_limits = joint.limited == mujoco.mjtLimited.mjLIMITED_TRUE or (
            joint.limited == mujoco.mjtLimited.mjLIMITED_AUTO and joint.range[0] < joint.range[1]
        )
        lo, hi = joint.range if has_limits else CONTINUOUS_RANGE
        act = spec.add_actuator(name=joint.name, target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=kp, kv=1.0)
        act.ctrlrange = [lo, hi]
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE

    # Scene dressing: floor + light.
    world = spec.worldbody
    world.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
    world.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], pos=[0, 0, -1.0],
        rgba=[0.3, 0.3, 0.35, 1], contype=int(collisions), conaffinity=int(collisions),
    )
    if collisions:
        exclude_resting_contacts(spec)
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gravity", action="store_true", help="enable gravity (default: off)")
    parser.add_argument("--kp", type=float, default=20.0, help="position actuator gain")
    parser.add_argument("--no-collision", action="store_true", help="disable all contacts")
    parser.add_argument("--export", metavar="PATH", help="write the generated MJCF to PATH and exit")
    args = parser.parse_args()

    spec = build_spec(args.gravity, args.kp, collisions=not args.no_collision)
    model = spec.compile()

    print(f"Loaded {model.nbody} bodies, {model.njnt} joints, {model.nu} actuators")
    for i in range(model.nu):
        lo, hi = model.actuator_ctrlrange[i]
        print(f"  {model.actuator(i).name:20s} [{math.degrees(lo):7.1f}, {math.degrees(hi):7.1f}] deg")

    if args.export:
        Path(args.export).write_text(spec.to_xml())
        print(f"Wrote {args.export}")
        return

    mujoco.viewer.launch(model)


if __name__ == "__main__":
    main()
