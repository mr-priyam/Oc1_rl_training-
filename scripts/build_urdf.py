#!/usr/bin/env python3
"""
build_urdf.py - turn the raw Onshape export of oc1_bipedal into a clean, simulation-ready URDF.

    python build_urdf.py oc1_bipedal_onshape.urdf oc1_bipedal.urdf

Geometry, masses and joint sign conventions are preserved exactly (verify_urdf.py --reference
proves it). What changes:

  1. The tree is re-rooted at torso_ss. Onshape rooted the export at the right ankle motor
     (through a dummy "root" link), so the right leg was stored upside down. Joints on that path
     are reversed and their frames/axes rewritten so a positive joint angle still moves the leg
     exactly the way it does in Onshape.
  2. The base link (torso_ss) gets a simulation-friendly frame: origin midway between the two
     hip-pitch joints, x forward, y left, z up, so the robot spawns upright with an identity
     base orientation.
  3. The electronics are merged into one "electronics" link fixed to the torso (combined mass,
     centre of mass and inertia; every board keeps its mesh as a visual). The IMU stays its own
     link so its frame can carry an IMU sensor in simulation.
  4. <collision> is added to the structural links (same mesh and origin as the visual, the rule
     used in the earlier URDF). Bearings, printed spacers/couplers and electronics get none.
  5. The exporter's placeholder effort/velocity (1, 1) are replaced by RobStride ratings.

Edit the CONFIG block and re-run this after every Onshape re-export instead of hand-editing.
Needs only Python 3 + numpy.
"""
import math
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, deque

import numpy as np

# ============================== CONFIG ==============================
NEW_ROOT = "torso_ss"
DROP_LINKS = ["root"]  # Onshape's dummy world link (the joint holding it is dropped too)

# "pelvis_zup": origin midway between the hip-pitch joints, x forward, y left, z up
# "cad":        keep torso_ss's Onshape part frame
BASE_FRAME = "pelvis_zup"
LEFT_HIP_JOINT, RIGHT_HIP_JOINT = "left_hip_pitch", "right_hip_pitch"
LEFT_FOOT, RIGHT_FOOT = "lf", "rf"
# Which leg is written first. Simulators number joints in file order, so this sets the joint /
# action order: oc1_rl/robot.py expects the right leg first.
FIRST_LEG = "right"

ELECTRONICS_LINK = "electronics"
ELECTRONICS = ["battery", "battery_1", "rpi5", "buck_converter", "can_module",
               "power_distribution_board", "discharge_resistor"]

# Weighed masses in kg; None keeps the CAD value. The electronics' CAD masses are placeholders
# (the 74 x 89 x 199 mm battery is 1.3 g, i.e. a density of about 1 kg/m^3), so fill these in.
# Inertia is scaled with the mass and the centre of mass stays where CAD puts it.
MEASURED_MASS_KG = {
    "battery": None,
    "battery_1": None,
    "rpi5": None,
    "buck_converter": None,
    "can_module": None,
    "power_distribution_board": None,
    "discharge_resistor": None,
    "imu": None,
}

NO_COLLISION = {"bearing", "bearing_1",
                "lr2_3d_printed_spacer", "lr2_3d_printed_spacer_1",
                "ly3_3d_printed_coupler", "ly3_3d_printed_coupler_1",
                "ly3_3d_printed_spacer", "ly3_3d_printed_spacer_1",
                ELECTRONICS_LINK, "imu"}

# (peak torque N*m, no-load speed rad/s) from the RobStride datasheets
ACTUATORS = {"RS04": (120.0, 20.94),   # 200 rpm
             "RS03": (60.0, 20.42)}    # 195 rpm
JOINT_ACTUATOR = {
    "left_hip_pitch": "RS04", "left_hip_roll": "RS04", "left_hip_yaw": "RS03",
    "left_knee_pitch": "RS04", "left_ankle_pitch": "RS03",
    "right_hip_pitch": "RS04", "right_hip_roll": "RS04", "right_hip_yaw": "RS03",
    "right_knee_pitch": "RS04", "right_ankle_pitch": "RS03",
}
# Mechanical range in rad as (lower, upper); a joint listed here becomes "revolute".
# All joints use +-0.5 rad, as in the first URDF. Edit per joint once you know the hard stops.
# Joints not listed keep what Onshape exported: only left_knee_pitch has a range there,
# the rest stay "continuous" (no position limit) until you add their hard stops here.
JOINT_RANGE = {name: (-0.5, 0.5) for name in JOINT_ACTUATOR}  # same as the first URDF
# hip roll: outward swing only (the two joints have opposite sign conventions)
JOINT_RANGE["left_hip_roll"] = (-0.5, 0.0)
JOINT_RANGE["right_hip_roll"] = (0.0, 0.5)
# Positive joint direction. In this CAD every RS04 joint turns about the motor's own -z for +q and
# every RS03 joint about its +x, so the sim matches each motor model everywhere or nowhere.
# Check once on the robot: command +0.1 rad on one RS04 and one RS03 and compare with the sim.
# If a model turns the other way, set it to True here and re-run.
FLIP_SIGN = {"RS04": False, "RS03": False}
# ====================================================================


# ----------------------------- math --------------------------------
def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p),
                              math.cos(y), math.sin(y))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def R_to_rpy(R):
    """Inverse of rpy_to_R (URDF convention R = Rz(yaw) Ry(pitch) Rx(roll))."""
    c = math.hypot(R[0, 0], R[1, 0])
    p = math.atan2(-R[2, 0], c)
    if c > 1e-10:
        r = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock, pitch = +-pi/2: only roll -/+ yaw is defined, take yaw = 0
        y = 0.0
        r = math.atan2(R[0, 1], R[1, 1]) if R[2, 0] < 0 else math.atan2(-R[0, 1], R[1, 1])
    rpy = (r, p, y)
    assert np.abs(rpy_to_R(*rpy) - R).max() < 1e-9, "rpy round-trip failed"
    return rpy


def make_T(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def inv(T):
    R, p = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ p)


def fmt(x, sig=12, tiny=1e-12):
    return "0" if abs(x) < tiny else f"{x:.{sig}g}"


def fmt_vec(v, sig=12):
    return " ".join(fmt(float(x), sig) for x in v)


def clean(s):  # "0 -0 0" -> "0 0 0"
    return " ".join("0" if t in ("-0", "+0") else t for t in s.split())


def unchanged(A, B):
    return B is not None and np.abs(A - B).max() < 1e-12


def origin_xml(T, raw=None, T_old=None):
    """Re-use the exported numbers when the transform did not change (keeps diffs readable)."""
    if raw is not None and unchanged(T, T_old):
        return f'<origin xyz="{clean(raw[0])}" rpy="{clean(raw[1])}"/>'
    return f'<origin xyz="{fmt_vec(T[:3, 3])}" rpy="{fmt_vec(R_to_rpy(T[:3, :3]))}"/>'


# ----------------------------- parsing -----------------------------
def parse_vec(s, default="0 0 0"):
    return np.array([float(v) for v in (s or default).split()])


def parse_origin(el):
    """Returns (T, (xyz_string, rpy_string))."""
    o = el.find("origin")
    if o is None:
        return np.eye(4), ("0 0 0", "0 0 0")
    xyz, rpy = o.get("xyz", "0 0 0"), o.get("rpy", "0 0 0")
    return make_T(rpy_to_R(*parse_vec(rpy)), parse_vec(xyz)), (xyz, rpy)


class Elem:  # a <visual> or <collision>
    def __init__(self, T, geom_tag, geom_attrs, material=None, name=None, raw=None, T_old=None):
        self.T, self.geom_tag, self.geom_attrs = T, geom_tag, dict(geom_attrs)
        self.material, self.name, self.raw, self.T_old = material, name, raw, T_old


def parse_elem(el):
    g = list(el.find("geometry"))[0]
    mat = None
    m = el.find("material")
    if m is not None:
        c = m.find("color")
        mat = (m.get("name"), c.get("rgba") if c is not None else None)
    T, raw = parse_origin(el)
    return Elem(T, g.tag, g.attrib, mat, el.get("name"), raw, T.copy())


def parse_urdf(path):
    root = ET.parse(path).getroot()
    links, joints = {}, {}
    for l in root.findall("link"):
        d = {"inertial": None, "visuals": [], "collisions": []}
        i = l.find("inertial")
        if i is not None:
            ie = i.find("inertia")
            I = np.array([[float(ie.get("ixx")), float(ie.get("ixy")), float(ie.get("ixz"))],
                          [float(ie.get("ixy")), float(ie.get("iyy")), float(ie.get("iyz"))],
                          [float(ie.get("ixz")), float(ie.get("iyz")), float(ie.get("izz"))]])
            T, raw = parse_origin(i)
            d["inertial"] = {"mass": float(i.find("mass").get("value")), "T": T, "I": I,
                             "raw_origin": raw, "raw_mass": i.find("mass").get("value"),
                             "raw_inertia": dict(ie.attrib)}
        d["visuals"] = [parse_elem(v) for v in l.findall("visual")]
        d["collisions"] = [parse_elem(c) for c in l.findall("collision")]
        links[l.get("name")] = d
    for j in root.findall("joint"):
        lim = j.find("limit")
        ax = j.find("axis")
        T, raw = parse_origin(j)
        joints[j.get("name")] = {
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "T": T, "raw_origin": raw,
            "axis": parse_vec(ax.get("xyz") if ax is not None else None, "1 0 0"),
            "raw_axis": ax.get("xyz") if ax is not None else None,
            "limit": dict(lim.attrib) if lim is not None else None,
        }
    return root.get("name"), links, joints


# ----------------------------- build -------------------------------
def build(src, dst):
    robot_name, links, joints = parse_urdf(src)
    report = []
    warn = []

    # world poses of every link at q = 0 in the exported tree
    child_joints = defaultdict(list)
    has_parent = set()
    for jn, j in joints.items():
        child_joints[j["parent"]].append(jn)
        has_parent.add(j["child"])
    old_roots = [l for l in links if l not in has_parent]
    if len(old_roots) != 1:
        sys.exit(f"export must have exactly one root, found {old_roots}")
    Wo = {old_roots[0]: np.eye(4)}
    dq = deque(old_roots)
    while dq:
        p = dq.popleft()
        for jn in child_joints[p]:
            Wo[joints[jn]["child"]] = Wo[p] @ joints[jn]["T"]
            dq.append(joints[jn]["child"])

    # ---- re-root: undirected graph without the dummy root and the electronics
    drop = set(DROP_LINKS)
    merged = set(ELECTRONICS)
    adj = defaultdict(list)
    for jn, j in joints.items():
        if j["parent"] in drop or j["child"] in drop:
            report.append(f"dropped joint {jn} ({j['parent']} -> {j['child']})")
            continue
        if j["child"] in merged or j["parent"] in merged:
            if j["type"] != "fixed" or j["parent"] in merged:
                sys.exit(f"{jn}: electronics must be leaf links on fixed joints")
            continue
        adj[j["parent"]].append((j["child"], jn))
        adj[j["child"]].append((j["parent"], jn))

    new_parent = {}           # link -> (parent link, joint name, reversed?)
    order = [NEW_ROOT]
    seen = {NEW_ROOT}
    dq = deque([NEW_ROOT])
    while dq:
        p = dq.popleft()
        for c, jn in adj[p]:
            if c in seen:
                continue
            seen.add(c)
            new_parent[c] = (p, jn, joints[jn]["child"] == p)
            order.append(c)
            dq.append(c)
    expected = set(links) - drop - merged
    if seen != expected:
        sys.exit(f"links not connected to {NEW_ROOT}: {sorted(expected - seen)}")
    if len(new_parent) != len(expected) - 1:
        sys.exit("the joint graph is not a tree (a loop is present)")
    reversed_joints = [jn for c, (p, jn, rev) in new_parent.items() if rev]

    # ---- base frame
    Rt = Wo[NEW_ROOT][:3, :3]
    if BASE_FRAME == "pelvis_zup":
        pL = Wo[joints[LEFT_HIP_JOINT]["child"]][:3, 3]   # joint frame = old child frame at q=0
        pR = Wo[joints[RIGHT_HIP_JOINT]["child"]][:3, 3]
        origin = (pL + pR) / 2

        def com_world(n):
            ine = links[n]["inertial"]
            return (Wo[n] @ ine["T"])[:3, 3]

        up = origin - (com_world(LEFT_FOOT) + com_world(RIGHT_FOOT)) / 2
        lat = pL - pR

        def snap(v, exclude=()):
            vt = Rt.T @ (v / np.linalg.norm(v))
            cand = [i for i in range(3) if i not in exclude]
            i = max(cand, key=lambda k: abs(vt[k]))
            e = np.zeros(3)
            e[i] = math.copysign(1.0, vt[i])
            return e, i, math.degrees(math.acos(min(1.0, abs(vt[i]))))

        z_t, iz, ang_z = snap(up)
        y_t, iy, ang_y = snap(lat, exclude=(iz,))
        if ang_z > 15 or ang_y > 15:
            sys.exit(f"torso axes are {ang_z:.1f} deg / {ang_y:.1f} deg away from up / lateral; "
                     "set BASE_FRAME = 'cad'")
        x_t = np.cross(y_t, z_t)
        R_tb = np.column_stack([x_t, y_t, z_t])
        W_base = make_T(Rt @ R_tb, origin)
        names = "xyz"

        def axname(e):
            i = int(np.argmax(np.abs(e)))
            return ("+" if e[i] > 0 else "-") + names[i]

        report.append(f"base frame: x(fwd)=torso {axname(x_t)}, y(left)=torso {axname(y_t)}, "
                      f"z(up)=torso {axname(z_t)}; legs are {ang_z:.2f} deg off the torso axis, "
                      f"hips {ang_y:.2f} deg off; origin = midpoint of the hip-pitch joints")
    elif BASE_FRAME == "cad":
        W_base = Wo[NEW_ROOT].copy()
    else:
        sys.exit(f"unknown BASE_FRAME {BASE_FRAME!r}")

    # ---- new link frames (world, q = 0)
    Wn = {NEW_ROOT: W_base}
    for c in order[1:]:
        p, jn, rev = new_parent[c]
        # a joint frame coincides with the child frame; for a reversed joint that is the
        # old child's frame, i.e. the old frame of the new parent
        Wn[c] = Wo[p].copy() if rev else Wo[c].copy()
    Wn[ELECTRONICS_LINK] = W_base.copy()

    def re_express(link, T_local):  # element frame: old link frame -> new link frame
        return inv(Wn[link]) @ Wo[link] @ T_local

    def scaled_inertial(name, ine):
        m, I = ine["mass"], ine["I"]
        new_m = MEASURED_MASS_KG.get(name)
        if new_m is not None:
            I = I * (new_m / m)
            report.append(f"mass override {name}: {m:.6g} -> {new_m:.6g} kg")
            m = new_m
        elif name in MEASURED_MASS_KG and m < 0.005:
            warn.append(f"{name}: CAD mass {m * 1000:.3g} g looks like a placeholder; "
                        "put the weighed mass in MEASURED_MASS_KG")
        return m, I

    out_links = {}
    for n in order:
        L = links[n]
        d = {"inertial": None, "visuals": [], "collisions": []}
        if L["inertial"] is not None:
            m, I = scaled_inertial(n, L["inertial"])
            T = re_express(n, L["inertial"]["T"])
            R = T[:3, :3]
            d["inertial"] = {"mass": m, "com": T[:3, 3], "I": R @ I @ R.T, "raw": None}
            if unchanged(T, L["inertial"]["T"]) and m == L["inertial"]["mass"]:
                ine = L["inertial"]
                d["inertial"]["raw"] = (ine["raw_origin"], ine["raw_mass"], ine["raw_inertia"])
        for v in L["visuals"]:
            d["visuals"].append(Elem(re_express(n, v.T), v.geom_tag, v.geom_attrs, v.material,
                                     v.name, v.raw, v.T_old))
        for c in L["collisions"]:
            d["collisions"].append(Elem(re_express(n, c.T), c.geom_tag, c.geom_attrs, None,
                                        c.name, c.raw, c.T_old))
        if not d["collisions"] and n not in NO_COLLISION:
            for k, v in enumerate(d["visuals"]):
                nm = f"{n}_collision" if len(d["visuals"]) == 1 else f"{n}_collision_{k}"
                d["collisions"].append(Elem(v.T.copy(), v.geom_tag, v.geom_attrs, None, nm,
                                            v.raw, v.T_old))
        out_links[n] = d

    # ---- electronics: one rigid link with the combined inertial
    e = {"inertial": None, "visuals": [], "collisions": []}
    m_tot, first, parts = 0.0, 0.0, []
    for n in ELECTRONICS:
        L = links[n]
        Tc = inv(W_base) @ Wo[n]  # component frame expressed in the base frame
        m, I = scaled_inertial(n, L["inertial"])
        Ti = Tc @ L["inertial"]["T"]
        parts.append((m, Ti[:3, 3], Ti[:3, :3] @ I @ Ti[:3, :3].T))
        for v in L["visuals"]:
            e["visuals"].append(Elem(Tc @ v.T, v.geom_tag, v.geom_attrs, v.material, n))
    m_tot = sum(p[0] for p in parts)
    com = sum(p[0] * p[1] for p in parts) / m_tot
    I_tot = np.zeros((3, 3))
    for m, c, I in parts:
        dv = c - com
        I_tot += I + m * (dv @ dv * np.eye(3) - np.outer(dv, dv))
    e["inertial"] = {"mass": m_tot, "com": com, "I": I_tot, "raw": None}
    out_links[ELECTRONICS_LINK] = e
    report.append(f"merged {len(ELECTRONICS)} electronics links into '{ELECTRONICS_LINK}': "
                  f"{m_tot * 1000:.4g} g total")

    # ---- joints
    out_joints = {}
    for c in order[1:]:
        p, jn, rev = new_parent[c]
        j = joints[jn]
        oj = {"name": jn, "type": j["type"], "parent": p, "child": c,
              "T": inv(Wn[p]) @ Wn[c], "axis": None, "limit": None,
              "raw_origin": j["raw_origin"], "T_old": None if rev else j["T"], "raw_axis": None}
        if j["type"] != "fixed":
            a_world = Wo[j["child"]][:3, :3] @ j["axis"]
            a = Wn[c][:3, :3].T @ (-a_world if rev else a_world)
            a = a / np.linalg.norm(a)
            a[np.abs(a) < 1e-12] = 0.0
            oj["axis"] = a
            if (not rev and j["raw_axis"] is not None
                    and np.abs(a - j["axis"] / np.linalg.norm(j["axis"])).max() < 1e-12):
                oj["raw_axis"] = j["raw_axis"]
            if jn not in JOINT_ACTUATOR:
                sys.exit(f"actuated joint {jn} has no entry in JOINT_ACTUATOR")
            act = JOINT_ACTUATOR[jn]
            flip = FLIP_SIGN.get(act, False)
            if flip:
                oj["axis"] = -a
                oj["raw_axis"] = None
                report.append(f"positive direction flipped: {jn}")
            if not p.lower().startswith(act.lower()):
                warn.append(f"{jn}: parent link {p} does not look like an {act}")
            effort, vel = ACTUATORS[act]
            lim = {"effort": effort, "velocity": vel}
            if jn in JOINT_RANGE:
                oj["type"] = "revolute"
                lim["lower"], lim["upper"] = JOINT_RANGE[jn]
            elif j["type"] == "revolute":
                lo, hi = float(j["limit"]["lower"]), float(j["limit"]["upper"])
                lim["lower"], lim["upper"] = (-hi, -lo) if flip else (lo, hi)
            oj["limit"] = lim
        out_joints[c] = oj
    out_joints[ELECTRONICS_LINK] = {"name": "electronics_fixed", "type": "fixed",
                                    "parent": NEW_ROOT, "child": ELECTRONICS_LINK,
                                    "T": np.eye(4), "axis": None, "limit": None,
                                    "raw_origin": None, "T_old": None, "raw_axis": None}
    report.append("reversed joints (frames rewritten, sign kept): " + ", ".join(reversed_joints))
    still_cont = [jn for jn in JOINT_ACTUATOR
                  if out_joints[[c for c in out_joints if out_joints[c]["name"] == jn][0]]["type"]
                  == "continuous"]
    if still_cont:
        warn.append("no position limits yet (continuous): " + ", ".join(still_cont)
                    + "  -> add hard stops to JOINT_RANGE")

    # ---- write
    kids = defaultdict(list)
    for c, oj in out_joints.items():
        kids[oj["parent"]].append(c)
    size = {}

    def subtree(n):
        size[n] = 1 + sum(subtree(k) for k in kids[n])
        return size[n]

    subtree(NEW_ROOT)

    def contains(n, target):
        return n == target or any(contains(k, target) for k in kids[n])

    def section(n):
        left_first = FIRST_LEG == "left"
        if contains(n, LEFT_FOOT):
            return (0 if left_first else 1), "LEFT LEG"
        if contains(n, RIGHT_FOOT):
            return (1 if left_first else 0), "RIGHT LEG"
        if n == ELECTRONICS_LINK:
            return 3, "ELECTRONICS (merged into one link)"
        return 2, "TORSO FRAME PLATES (IMU hangs off part_7)"

    lines = []
    w = lines.append

    def write_elem(tag, el, ind):
        nm = f' name="{el.name}"' if el.name else ""
        w(f"{ind}<{tag}{nm}>")
        w(f"{ind}  {origin_xml(el.T, el.raw, el.T_old)}")
        attrs = " ".join(f'{k}="{v}"' for k, v in el.geom_attrs.items())
        w(f"{ind}  <geometry><{el.geom_tag} {attrs}/></geometry>")
        if tag == "visual" and el.material:
            mname, rgba = el.material
            if rgba:
                w(f'{ind}  <material name="{mname}"><color rgba="{rgba}"/></material>')
            else:
                w(f'{ind}  <material name="{mname}"/>')
        w(f"{ind}</{tag}>")

    def write_link(n, ind="  "):
        d = out_links[n]
        w(f'{ind}<link name="{n}">')
        if d["inertial"] is not None:
            ine = d["inertial"]
            I = ine["I"]
            w(f"{ind}  <inertial>")
            if ine["raw"] is not None:  # untouched: keep the exported numbers
                (xyz, rpy), mass, ia = ine["raw"]
                w(f'{ind}    <origin xyz="{clean(xyz)}" rpy="{clean(rpy)}"/>')
                w(f'{ind}    <mass value="{mass}"/>')
                w(f'{ind}    <inertia ' + " ".join(f'{k}="{ia[k]}"' for k in
                                                  ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"))
                  + "/>")
            else:
                w(f'{ind}    <origin xyz="{fmt_vec(ine["com"])}" rpy="0 0 0"/>')
                w(f'{ind}    <mass value="{fmt(ine["mass"], 10)}"/>')
                w(f'{ind}    <inertia ' + " ".join(
                    f'{k}="{fmt(I[a, b], 10, 1e-14)}"' for k, a, b in
                    (("ixx", 0, 0), ("ixy", 0, 1), ("ixz", 0, 2),
                     ("iyy", 1, 1), ("iyz", 1, 2), ("izz", 2, 2))) + "/>")
            w(f"{ind}  </inertial>")
        for v in d["visuals"]:
            write_elem("visual", v, ind + "  ")
        for c in d["collisions"]:
            write_elem("collision", c, ind + "  ")
        w(f"{ind}</link>")

    def write_joint(oj, ind="  "):
        if oj["type"] != "fixed":
            act = JOINT_ACTUATOR.get(oj["name"], "")
            w(f"{ind}<!-- actuated: {oj['name']} ({act}) -->")
        w(f'{ind}<joint name="{oj["name"]}" type="{oj["type"]}">')
        w(f'{ind}  <parent link="{oj["parent"]}"/>')
        w(f'{ind}  <child link="{oj["child"]}"/>')
        w(f"{ind}  {origin_xml(oj['T'], oj['raw_origin'], oj['T_old'])}")
        if oj["axis"] is not None:
            ax = clean(oj["raw_axis"]) if oj["raw_axis"] else fmt_vec(oj["axis"])
            w(f'{ind}  <axis xyz="{ax}"/>')
        if oj["limit"] is not None:
            lim = oj["limit"]
            rng = (f' lower="{fmt(lim["lower"])}" upper="{fmt(lim["upper"])}"'
                   if "lower" in lim else "")
            w(f'{ind}  <limit effort="{fmt(lim["effort"])}" velocity="{fmt(lim["velocity"])}"'
              f"{rng}/>")
        w(f"{ind}</joint>")

    def walk(n):
        for c in sorted(kids[n], key=lambda k: (size[k], k)):
            write_joint(out_joints[c])
            write_link(c)
            walk(c)

    base_txt = ("origin midway between the hip-pitch joints, x forward, y left, z up"
                if BASE_FRAME == "pelvis_zup" else "Onshape part frame of torso_ss")
    w('<?xml version="1.0"?>')
    w("<!--")
    w(f"  {robot_name}: generated by build_urdf.py from the Onshape export.")
    w(f"  Root / floating base: {NEW_ROOT}. Base frame: {base_txt}.")
    w("  The tree runs parent to child from the torso down each leg; joint sign conventions")
    w("  are identical to the Onshape export.")
    w(f"  '{ELECTRONICS_LINK}': " + ", ".join(ELECTRONICS) + " merged into one rigid link.")
    w("  Collision geometry = the visual mesh on structural links.")
    w("  Edit build_urdf.py and re-run it instead of hand-editing this file.")
    w("-->")
    w(f'<robot name="{robot_name}">')
    w("")
    w(f"  <!-- ==================== BASE ==================== -->")
    write_link(NEW_ROOT)
    for sec_id, title in sorted({section(c) for c in kids[NEW_ROOT]}):
        w("")
        w(f"  <!-- ==================== {title} ==================== -->")
        for c in sorted([k for k in kids[NEW_ROOT] if section(k)[0] == sec_id],
                        key=lambda k: (size[k], k)):
            write_joint(out_joints[c])
            write_link(c)
            walk(c)
    w("</robot>")
    with open(dst, "w") as f:
        f.write("\n".join(lines) + "\n")

    n_act = sum(1 for oj in out_joints.values() if oj["type"] != "fixed")
    print(f"wrote {dst}: {len(out_links)} links, {len(out_joints)} joints "
          f"({n_act} actuated), root = {NEW_ROOT}")
    for r in report:
        print("  -", r)
    for x in warn:
        print("  WARNING:", x)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    build(sys.argv[1], sys.argv[2])