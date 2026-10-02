#!/usr/bin/env python3
"""
verify_urdf.py - check that a URDF is structurally sound, physically plausible and simulates.

  python verify_urdf.py oc1_bipedal.urdf --expect-root torso_ss
        structure, joints, inertia, mass balance (no meshes needed)
  python verify_urdf.py oc1_bipedal.urdf --mesh-dir meshes
        + mesh files, part densities, and MuJoCo tests (load, self-collision, drop test)
  python verify_urdf.py oc1_bipedal.urdf --reference oc1_bipedal_onshape.urdf
        + proof that every mesh, mass and joint direction matches the original Onshape export
  python verify_urdf.py oc1_bipedal.urdf --mesh-dir meshes --view
        + open the robot in the MuJoCo viewer (dropped onto a floor, joints held stiff)
  python verify_urdf.py oc1_bipedal.urdf --plot skeleton.png
        + stick-figure picture of the kinematic tree

Exit code 0 = no FAIL (warnings allowed), 1 = at least one FAIL.
Needs numpy. Uses mujoco and matplotlib when installed:  pip install mujoco matplotlib
"""
import argparse
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, deque

import numpy as np

ACTUATED = ("revolute", "continuous", "prismatic")


# ================================================================ report
class Report:
    def __init__(self):
        self.counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
        self.warnings, self.failures = [], []

    def section(self, title):
        print(f"\n== {title} " + "=" * max(3, 74 - len(title)))

    def _emit(self, kind, msg):
        self.counts[kind] += 1
        print(f"  [{kind}] {msg}")
        if kind == "WARN":
            self.warnings.append(msg)
        elif kind == "FAIL":
            self.failures.append(msg)

    def ok(self, msg):
        self._emit("PASS", msg)

    def warn(self, msg):
        self._emit("WARN", msg)

    def fail(self, msg):
        self._emit("FAIL", msg)

    @staticmethod
    def info(msg):
        print(f"         {msg}")


# ================================================================ math
def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def transform(xyz, rpy):
    """URDF origin: rotate by roll (X), then pitch (Y), then yaw (Z), all about fixed axes."""
    T = np.eye(4)
    T[:3, :3] = rot_z(rpy[2]) @ rot_y(rpy[1]) @ rot_x(rpy[0])
    T[:3, 3] = xyz
    return T


def axis_angle(axis, q):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K


def inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def rot_angle(R):
    return math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))


def rpy_of(R):
    p = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    if math.hypot(R[0, 0], R[1, 0]) < 1e-9:
        return 0.0, p, math.atan2(-R[0, 1], R[1, 1])
    return math.atan2(R[2, 1], R[2, 2]), p, math.atan2(R[1, 0], R[0, 0])


# ================================================================ parsing
def floats(s, n):
    vals = [float(v) for v in s.split()]
    if len(vals) != n or not all(math.isfinite(v) for v in vals):
        raise ValueError(f"expected {n} finite numbers, got '{s}'")
    return np.array(vals)


def parse_origin(el):
    o = el.find("origin")
    if o is None:
        return np.eye(4)
    return transform(floats(o.get("xyz", "0 0 0"), 3), floats(o.get("rpy", "0 0 0"), 3))


class Model:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.problems = []
        root = ET.parse(path).getroot()
        if root.tag != "robot":
            self.problems.append(f"top-level element is <{root.tag}>, expected <robot>")
        self.name = root.get("name")
        self.links, self.joints = {}, {}
        self.dup_links, self.dup_joints = [], []
        for l in root.findall("link"):
            name = l.get("name")
            if name in self.links:
                self.dup_links.append(name)
            try:
                self.links[name] = self._link(l)
            except Exception as e:  # noqa: BLE001
                self.problems.append(f"link {name}: {e}")
                self.links[name] = {"inertial": None, "visuals": [], "collisions": []}
        for j in root.findall("joint"):
            name = j.get("name")
            if name in self.joints:
                self.dup_joints.append(name)
            try:
                self.joints[name] = self._joint(j)
            except Exception as e:  # noqa: BLE001
                self.problems.append(f"joint {name}: {e}")

    @staticmethod
    def _geom(el):
        g = el.find("geometry")
        if g is None or len(g) == 0:
            raise ValueError("visual/collision without geometry")
        s = g[0]
        d = {"T": parse_origin(el), "kind": s.tag, "name": el.get("name")}
        if s.tag == "mesh":
            d["filename"] = s.get("filename")
            d["scale"] = floats(s.get("scale", "1 1 1"), 3)
        return d

    def _link(self, l):
        d = {"inertial": None,
             "visuals": [self._geom(v) for v in l.findall("visual")],
             "collisions": [self._geom(c) for c in l.findall("collision")]}
        i = l.find("inertial")
        if i is not None:
            ie, me = i.find("inertia"), i.find("mass")
            if ie is None or me is None:
                raise ValueError("<inertial> needs <mass> and <inertia>")
            g = lambda k: float(ie.get(k, "0"))  # noqa: E731
            I = np.array([[g("ixx"), g("ixy"), g("ixz")],
                          [g("ixy"), g("iyy"), g("iyz")],
                          [g("ixz"), g("iyz"), g("izz")]])
            d["inertial"] = {"mass": float(me.get("value")), "T": parse_origin(i), "I": I}
        return d

    @staticmethod
    def _joint(j):
        ax = j.find("axis")
        lim = j.find("limit")
        p, c = j.find("parent"), j.find("child")
        if p is None or c is None:
            raise ValueError("joint needs <parent> and <child>")
        return {"type": j.get("type"), "parent": p.get("link"), "child": c.get("link"),
                "T": parse_origin(j),
                "axis": floats(ax.get("xyz"), 3) if ax is not None else np.array([1.0, 0, 0]),
                "limit": {k: float(v) for k, v in lim.attrib.items()} if lim is not None else None}

    # ---- kinematics (valid once the tree checks passed)
    def build_tree(self):
        self.parent, self.kids = {}, defaultdict(list)
        for jn, j in self.joints.items():
            self.parent[j["child"]] = (j["parent"], jn)
            self.kids[j["parent"]].append((j["child"], jn))
        self.root = [l for l in self.links if l not in self.parent][0]
        self.order = [self.root]
        dq = deque([self.root])
        while dq:
            n = dq.popleft()
            for c, _ in self.kids[n]:
                self.order.append(c)
                dq.append(c)

    def fk(self, q=None):
        q = q or {}
        W = {self.root: np.eye(4)}
        for n in self.order[1:]:
            p, jn = self.parent[n]
            j = self.joints[jn]
            M = np.eye(4)
            v = q.get(jn, 0.0)
            if j["type"] in ("revolute", "continuous"):
                M[:3, :3] = axis_angle(j["axis"], v)
            elif j["type"] == "prismatic":
                M[:3, 3] = j["axis"] / np.linalg.norm(j["axis"]) * v
            W[n] = W[p] @ j["T"] @ M
        return W

    def mass_props(self, W, links=None):
        m_tot, mc = 0.0, np.zeros(3)
        items = []
        for n in (links if links is not None else self.links):
            ine = self.links[n]["inertial"]
            if ine is None:
                continue
            T = W[n] @ ine["T"]
            items.append((ine["mass"], T[:3, 3], T[:3, :3] @ ine["I"] @ T[:3, :3].T))
            m_tot += ine["mass"]
            mc += ine["mass"] * T[:3, 3]
        if m_tot == 0:
            return 0.0, np.zeros(3), np.zeros((3, 3))
        com = mc / m_tot
        I = np.zeros((3, 3))
        for m, c, Ic in items:
            d = c - com
            I += Ic + m * (d @ d * np.eye(3) - np.outer(d, d))
        return m_tot, com, I

    def subtree(self, n):
        out, st = [], [n]
        while st:
            x = st.pop()
            out.append(x)
            st.extend(c for c, _ in self.kids[x])
        return out

    def rigid_group(self, n):
        """Links rigidly attached to n through fixed joints (not crossing actuated joints)."""
        out, st, seen = [], [n], {n}
        while st:
            x = st.pop()
            out.append(x)
            nbrs = [(c, jn) for c, jn in self.kids[x]]
            if x in self.parent:
                nbrs.append(self.parent[x])
            for y, jn in nbrs:
                if y not in seen and self.joints[jn]["type"] == "fixed":
                    seen.add(y)
                    st.append(y)
        return out


# ================================================================ meshes
def resolve_mesh(filename, urdf_dir, mesh_dir):
    fn = filename[7:] if filename.startswith("file://") else filename
    base = os.path.basename(fn)
    cands = []
    if mesh_dir:
        cands.append(os.path.join(mesh_dir, base))
    if fn.startswith("package://"):
        pkg, _, rel = fn[len("package://"):].partition("/")
        d = urdf_dir
        for _ in range(6):
            cands.append(os.path.join(d, pkg, rel))
            if os.path.basename(d) == pkg:
                cands.append(os.path.join(d, rel))
            d = os.path.dirname(d)
        for p in os.environ.get("ROS_PACKAGE_PATH", "").split(os.pathsep):
            if p:
                cands.append(os.path.join(p, pkg, rel))
        cands += [os.path.join(urdf_dir, rel), os.path.join(urdf_dir, "meshes", base),
                  os.path.join(urdf_dir, "..", "meshes", base)]
    else:
        cands += [fn if os.path.isabs(fn) else os.path.join(urdf_dir, fn),
                  os.path.join(urdf_dir, "meshes", base)]
    for c in cands:
        if os.path.isfile(c):
            return os.path.normpath(c)
    return None


def read_stl(path):
    data = open(path, "rb").read()
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * n == len(data):
            dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            return np.frombuffer(data, dtype=dt, count=n, offset=84)["v"].astype(float)
    txt = data.decode("ascii", errors="ignore")
    num = r"([-+0-9.eE]+)"
    v = re.findall(rf"vertex\s+{num}\s+{num}\s+{num}", txt)
    if not v:
        raise ValueError("not a readable STL")
    return np.array(v, float).reshape(-1, 3, 3)


def mesh_stats(path, scale):
    tri = read_stl(path) * scale
    vol = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6)
    pts = tri.reshape(-1, 3)
    return {"n": len(tri), "volume": abs(vol), "bbox": pts.max(0) - pts.min(0)}


# ================================================================ checks
def check_structure(rep, m, expect_root):
    rep.section("1. File and kinematic tree")
    for p in m.problems:
        rep.fail(p)
    for d in m.dup_links:
        rep.fail(f"duplicate link name '{d}'")
    for d in m.dup_joints:
        rep.fail(f"duplicate joint name '{d}'")
    ok = not m.problems and not m.dup_links and not m.dup_joints
    parent = {}
    for jn, j in m.joints.items():
        for side in ("parent", "child"):
            if j[side] not in m.links:
                rep.fail(f"joint {jn}: {side} link '{j[side]}' does not exist")
                ok = False
        if j["parent"] == j["child"]:
            rep.fail(f"joint {jn} connects link '{j['child']}' to itself")
            ok = False
        if j["child"] in parent:
            rep.fail(f"link '{j['child']}' has two parent joints: {parent[j['child']]} and {jn}")
            ok = False
        parent[j["child"]] = jn
    roots = [l for l in m.links if l not in parent]
    if len(roots) != 1:
        rep.fail(f"a URDF needs exactly one root link, found {len(roots)}: {roots[:6]}")
        return False
    if not ok:
        return False
    m.build_tree()
    unreached = set(m.links) - set(m.order)
    if unreached:
        rep.fail(f"links not connected to the root (part of a loop): {sorted(unreached)}")
        return False
    rep.ok(f"{len(m.links)} links, {len(m.joints)} joints, single root, no loops, "
           "every link reachable")
    if expect_root:
        if m.root == expect_root:
            rep.ok(f"root (parent of everything) is '{m.root}' as expected")
        else:
            rep.fail(f"root is '{m.root}', expected '{expect_root}'")
    else:
        rep.ok(f"root (parent of everything) is '{m.root}'")
    if not m.links[m.root]["inertial"]:
        rep.warn(f"root '{m.root}' has no mass: for a floating-base robot the base should be a "
                 "real body (often the cause of a robot rooted at a dummy 'root'/'world' link)")
    # chains of actuated joints, root -> leaves
    chains = []

    def walk(n, chain):
        acts = [(c, jn) for c, jn in m.kids[n]]
        if not acts:
            if chain and chain[-1][1] == n:
                chains.append(chain)
            return
        for c, jn in acts:
            walk(c, chain + [(jn, c)] if m.joints[jn]["type"] in ACTUATED else chain)

    walk(m.root, [])
    seen = set()
    for ch in chains:
        key = tuple(jn for jn, _ in ch)
        if key in seen:
            continue
        seen.add(key)
        Report.info(f"{m.root} -> " + " -> ".join(jn for jn, _ in ch) + f" -> [{ch[-1][1]}]")
    return True


def check_joints(rep, m):
    rep.section("2. Joints")
    valid = {"revolute", "continuous", "prismatic", "fixed", "floating", "planar"}
    act, cont, placeholder = [], [], []
    for jn, j in m.joints.items():
        t = j["type"]
        if t not in valid:
            rep.fail(f"{jn}: unknown joint type '{t}'")
            continue
        if t in ("floating", "planar"):
            rep.warn(f"{jn}: '{t}' joints are rarely supported by simulators inside a robot")
        if t in ACTUATED:
            act.append(jn)
            n = np.linalg.norm(j["axis"])
            if n < 1e-9:
                rep.fail(f"{jn}: axis is zero")
            elif abs(n - 1) > 1e-6:
                rep.warn(f"{jn}: axis {j['axis']} is not unit length (parsers normalise it)")
            lim = j["limit"]
            if t in ("revolute", "prismatic"):
                if lim is None or "lower" not in lim or "upper" not in lim:
                    rep.fail(f"{jn}: {t} joint needs <limit lower upper effort velocity>")
                elif lim["lower"] >= lim["upper"]:
                    rep.fail(f"{jn}: limit lower {lim['lower']} >= upper {lim['upper']}")
            if t == "continuous":
                cont.append(jn)
            if lim is not None:
                if lim.get("effort", 0) <= 0 or lim.get("velocity", 0) <= 0:
                    rep.fail(f"{jn}: effort and velocity limits must be > 0")
                elif lim["effort"] <= 1 or lim["velocity"] <= 1:
                    placeholder.append(jn)
    rep.ok(f"{len(act)} actuated joints, axes valid")
    if placeholder:
        rep.warn("effort/velocity look like exporter placeholders (<= 1): " + ", ".join(placeholder))
    if cont:
        rep.warn(f"{len(cont)} continuous joints have no position limits (a leg joint can spin "
                 "360 deg in sim): " + ", ".join(cont))
    return act


def check_inertia(rep, m):
    rep.section("3. Mass and inertia of every link")
    good, tiny, frames = 0, [], []
    for n, L in m.links.items():
        ine = L["inertial"]
        if ine is None:
            if L["visuals"] or L["collisions"]:
                rep.warn(f"{n}: has geometry but no <inertial> (PyBullet then gives it 1 kg)")
            else:
                frames.append(n)
            continue
        mass, I = ine["mass"], ine["I"]
        if not math.isfinite(mass) or mass <= 0:
            rep.fail(f"{n}: mass {mass} must be > 0")
            continue
        ev = np.linalg.eigvalsh(I)
        if ev.min() <= 0:
            rep.fail(f"{n}: inertia matrix is not positive definite (eigenvalues {ev})")
            continue
        a, b, c = sorted(ev)
        if a + b < c * (1 - 1e-6):
            rep.fail(f"{n}: inertia violates the triangle inequality ({a:.3g} + {b:.3g} < {c:.3g})")
            continue
        good += 1
        if mass < 0.01:
            tiny.append(f"{n} ({mass * 1000:.3g} g)")
    rep.ok(f"{good} links have positive mass and a physically valid inertia tensor")
    if frames:
        Report.info("frame-only links (no mass, no geometry): " + ", ".join(frames))
    if tiny:
        rep.warn("links lighter than 10 g, check the CAD material/mass: " + ", ".join(tiny))


def check_meshes(rep, m, mesh_dir):
    rep.section("4. Mesh files")
    urdf_dir = os.path.dirname(m.path)
    found, missing, stats = {}, set(), {}
    for n, L in m.links.items():
        for g in L["visuals"] + L["collisions"]:
            if g["kind"] != "mesh":
                continue
            fn = g["filename"]
            if fn in found or fn in missing:
                continue
            p = resolve_mesh(fn, urdf_dir, mesh_dir)
            if p:
                found[fn] = p
            else:
                missing.add(fn)
    if not found and missing:
        rep.warn(f"none of the {len(missing)} mesh files were found; pass --mesh-dir to check "
                 "meshes and run the MuJoCo tests")
        return None
    for fn in sorted(missing):
        rep.fail(f"mesh not found: {fn}")
    for fn, p in found.items():
        if p.lower().endswith(".stl"):
            try:
                stats[fn] = mesh_stats(p, 1.0)
            except Exception as e:  # noqa: BLE001
                rep.fail(f"{os.path.basename(p)}: {e}")
    if not missing:
        rep.ok(f"all {len(found)} mesh files found")
    big = [os.path.basename(f) for f, s in stats.items() if np.linalg.norm(s["bbox"]) > 3]
    small = [os.path.basename(f) for f, s in stats.items() if np.linalg.norm(s["bbox"]) < 1e-3]
    if big:
        rep.warn("meshes larger than 3 m (exported in millimetres?): " + ", ".join(big))
    if small:
        rep.warn("meshes smaller than 1 mm: " + ", ".join(small))
    heavy = [f"{os.path.basename(f)} ({s['n']} tris)" for f, s in stats.items() if s["n"] > 200000]
    if heavy:
        rep.warn("very dense meshes slow down collision checking: " + ", ".join(heavy))
    # density = mass / mesh volume catches wrong materials and placeholder masses
    odd = []
    for n, L in m.links.items():
        ine = L["inertial"]
        vis = [g for g in L["visuals"] if g["kind"] == "mesh"]
        if ine is None or not vis or not all(g["filename"] in stats for g in vis):
            continue
        vol = sum(stats[g["filename"]]["volume"] * abs(np.prod(g["scale"])) for g in vis)
        if vol <= 0:
            continue
        rho = ine["mass"] / vol
        if rho < 150 or rho > 12000:
            odd.append(f"{n}: {rho:.4g} kg/m^3")
    if odd:
        rep.warn("implausible density (mass / mesh volume); plastic ~1000-1400, aluminium ~2700, "
                 "steel ~7850 kg/m^3:")
        for o in odd:
            Report.info(o)
    elif stats:
        rep.ok("every link's density (mass / mesh volume) is plausible")
    dirs = {os.path.dirname(p) for p in found.values()}
    return dirs.pop() if len(dirs) == 1 and not missing else None


def check_collisions(rep, m, act):
    rep.section("5. Collision geometry")
    with_c = [n for n, L in m.links.items() if L["collisions"]]
    rep.ok(f"{len(with_c)} of {len(m.links)} links have collision geometry")
    mirror = 0
    for n in with_c:
        L = m.links[n]
        for c in L["collisions"]:
            if any(v["kind"] == c["kind"] and v.get("filename") == c.get("filename")
                   and np.abs(v["T"] - c["T"]).max() < 1e-9 for v in L["visuals"]):
                mirror += 1
    Report.info(f"{mirror} collision elements coincide exactly with their visual mesh")
    heavy_no = [f"{n} ({L['inertial']['mass']:.2f} kg)" for n, L in m.links.items()
                if not L["collisions"] and L["inertial"] and L["inertial"]["mass"] > 0.05]
    if heavy_no:
        Report.info("no collision (fine for internal parts): " + ", ".join(heavy_no))
    # distal bodies (feet, hands) must collide or they pass through the floor
    for jn in act:
        c = m.joints[jn]["child"]
        sub = m.subtree(c)
        if any(m.joints[m.parent[x][1]]["type"] in ACTUATED for x in sub if x != c):
            continue
        grp = m.rigid_group(c)
        if any(m.links[x]["collisions"] for x in grp):
            rep.ok(f"end body after {jn} ('{c}') has collision geometry")
        else:
            rep.warn(f"end body after {jn} ('{c}') has no collision: it will fall through the floor")


def feet_of(m, act):
    feet = []
    for jn in act:
        c = m.joints[jn]["child"]
        if not any(m.joints[m.parent[x][1]]["type"] in ACTUATED for x in m.subtree(c) if x != c):
            feet.append((jn, c))
    return feet


def check_mass_balance(rep, m, act):
    rep.section("6. Mass balance (zero joint angles)")
    W = m.fk()
    tot, com, _ = m.mass_props(W)
    rep.ok(f"total mass {tot:.3f} kg, centre of mass in base frame "
           f"[{com[0]:+.4f} {com[1]:+.4f} {com[2]:+.4f}] m")
    # left / right joints
    pairs = [(jn, "right_" + jn[5:]) for jn in m.joints
             if jn.startswith("left_") and "right_" + jn[5:] in m.joints]
    if pairs:
        rows = []
        for l, r in pairs:
            ml = m.mass_props(W, m.subtree(m.joints[l]["child"]))[0]
            mr = m.mass_props(W, m.subtree(m.joints[r]["child"]))[0]
            rows.append((l[5:], ml, mr))
        Report.info("mass carried below each joint (kg):   left    right   diff")
        for name, ml, mr in rows:
            Report.info(f"  {name:22s} {ml:7.3f} {mr:7.3f} {ml - mr:+7.3f}")
        name, ml, mr = max(rows, key=lambda r: max(r[1], r[2]))  # the whole leg
        rel = abs(ml - mr) / max(ml, mr)
        if rel > 0.05 or abs(ml - mr) > 0.5:
            rep.warn(f"left leg {ml:.3f} kg vs right leg {mr:.3f} kg ({rel * 100:.1f}% apart)")
        else:
            rep.ok(f"whole legs balanced: {ml:.3f} vs {mr:.3f} kg ({rel * 100:.1f}% apart)")
    # mirrored link names (lX / rX)
    diffs = []
    for n in m.links:
        if n.startswith("l") and "r" + n[1:] in m.links:
            a, b = m.links[n]["inertial"], m.links["r" + n[1:]]["inertial"]
            if a and b and abs(a["mass"] - b["mass"]) / max(a["mass"], b["mass"]) > 0.01:
                diffs.append(f"{n}/r{n[1:]}: {a['mass']:.3f}/{b['mass']:.3f} kg")
    if diffs:
        Report.info("mirrored parts with different mass (fine if these are weighed values): "
                    + "; ".join(diffs))
    feet = feet_of(m, act)
    if len(feet) == 2:
        pts = [m.mass_props(W, m.rigid_group(c))[1] for _, c in feet]
        mid = (pts[0] + pts[1]) / 2
        below = -mid[2]
        horiz = math.hypot(mid[0], mid[1])
        if below > 0.1 and horiz < 0.5 * below:
            rep.ok(f"base frame is z-up: feet are {below:.3f} m below the base origin")
            off = com - mid
            Report.info(f"centre of mass relative to the midpoint between the feet: "
                        f"x {off[0] * 1000:+.1f} mm, y {off[1] * 1000:+.1f} mm")
            if abs(off[1]) > 0.02:
                rep.warn(f"centre of mass is {off[1] * 1000:+.1f} mm off-centre sideways")
            for jn, c in feet:
                R = W[c][:3, :3]
                k = int(np.argmax(np.abs(R[2, :])))
                tilt = math.degrees(math.acos(min(1.0, abs(R[2, k]))))
                Report.info(f"foot '{c}' at zero pose: tilted {tilt:.2f} deg from level")
            for n in m.links:
                if "imu" in n.lower():
                    T = W[n]
                    r, p, y = rpy_of(T[:3, :3])
                    Report.info(f"IMU link '{n}' in base frame: xyz [{T[0, 3]:+.4f} {T[1, 3]:+.4f} "
                                f"{T[2, 3]:+.4f}] rpy [{r:+.4f} {p:+.4f} {y:+.4f}]")
            return True
        rep.warn("feet are not below the base along -z: the robot spawns lying down unless "
                 "you rotate the base")
    return False


def check_axes(rep, m, act, zup):
    W = m.fk()
    axes = {}
    for jn in act:
        a = W[m.joints[jn]["child"]][:3, :3] @ m.joints[jn]["axis"]
        axes[jn] = a / np.linalg.norm(a)
    # +q relative to the motor housing: joints whose parent is the same part (e.g. one motor
    # model) should turn the same way about that part, otherwise sim and real signs disagree
    by_mesh = defaultdict(list)
    for jn in act:
        p = m.joints[jn]["parent"]
        vis = [g for g in m.links[p]["visuals"] if g["kind"] == "mesh"]
        if len(vis) != 1:
            continue
        a = (W[p] @ vis[0]["T"])[:3, :3].T @ axes[jn]
        k = int(np.argmax(np.abs(a)))
        by_mesh[os.path.basename(vis[0]["filename"])].append(
            (jn, ("-" if a[k] < 0 else "+") + "xyz"[k]))
    for mesh, lst in sorted(by_mesh.items()):
        if len(lst) < 2:
            continue
        labels = sorted({lab for _, lab in lst})
        if len(labels) == 1:
            rep.ok(f"all {len(lst)} joints driven by {mesh} turn about its {labels[0]} axis for +q "
                   "(one sign convention per motor model)")
        else:
            rep.warn(f"joints driven by {mesh} turn opposite ways relative to the motor for +q: "
                     + ", ".join(f"{jn} {lab}" for jn, lab in lst))
    if not zup:
        return
    Report.info("actuated joint axes in the base frame (x fwd, y left, z up):")
    for jn in act:
        a = axes[jn]
        Report.info(f"  {jn:22s} [{a[0]:+.3f} {a[1]:+.3f} {a[2]:+.3f}]")
    pairs = [(jn, "right_" + jn[5:]) for jn in act if jn.startswith("left_")
             and "right_" + jn[5:] in axes]
    mirrored, opposite = [], []
    for l, r in pairs:
        k = int(np.argmax(np.abs(axes[l])))
        # mirror image through the sagittal plane: pitch axes keep their sign, roll/yaw flip
        sym = (axes[l][k] * axes[r][k] > 0) if k == 1 else (axes[l][k] * axes[r][k] < 0)
        (mirrored if sym else opposite).append(l[5:])
    if mirrored:
        Report.info("the same +q gives mirror-image motion on both legs: " + ", ".join(mirrored))
    if opposite:
        Report.info("the same +q gives OPPOSITE motion on the two legs (e.g. one hip flexes, the "
                    "other extends); negate one side for symmetric gaits: " + ", ".join(opposite))


def check_reference(rep, m, ref):
    rep.section("7. Equivalence with the reference URDF")
    act_m = {jn for jn, j in m.joints.items() if j["type"] in ACTUATED}
    act_r = {jn for jn, j in ref.joints.items() if j["type"] in ACTUATED}
    if act_m != act_r:
        rep.fail(f"actuated joints differ: only here {sorted(act_m - act_r)}, "
                 f"only in reference {sorted(act_r - act_m)}")
        return None
    rep.ok(f"same {len(act_m)} actuated joints")

    def vis_list(model):
        out = []
        for n, L in model.links.items():
            for g in L["visuals"]:
                if g["kind"] == "mesh":
                    out.append((os.path.basename(g["filename"]), n, g["T"]))
        return out

    vm, vr = vis_list(m), vis_list(ref)
    cnt_m = defaultdict(int)
    cnt_r = defaultdict(int)
    for f, _, _ in vm:
        cnt_m[f] += 1
    for f, _, _ in vr:
        cnt_r[f] += 1
    if dict(cnt_m) != dict(cnt_r):
        rep.fail("the two files do not contain the same meshes")
        return None
    anchors = [f for f in cnt_m if cnt_m[f] == 1]
    root_meshes = [os.path.basename(g["filename"]) for g in m.links[m.root]["visuals"]
                   if g["kind"] == "mesh"]
    anchor = next((f for f in root_meshes if f in anchors), anchors[0] if anchors else None)
    if anchor is None:
        rep.warn("no mesh is used exactly once; cannot anchor the comparison")
        return None

    def placements(model, vis, q):
        W = model.fk(q)
        Ts = [W[n] @ T for _, n, T in vis]
        a = Ts[[i for i, (f, _, _) in enumerate(vis) if f == anchor][0]]
        Ai = inv(a)
        return [Ai @ T for T in Ts], W, Ai

    P_m, _, _ = placements(m, vm, {})
    P_r, _, _ = placements(ref, vr, {})
    match, used = [], set()
    for i, (f, _, _) in enumerate(vm):
        best = min((k for k, (g, _, _) in enumerate(vr) if g == f and k not in used),
                   key=lambda k: np.linalg.norm(P_m[i][:3, 3] - P_r[k][:3, 3]) +
                   rot_angle(P_m[i][:3, :3].T @ P_r[k][:3, :3]))
        used.add(best)
        match.append(best)

    def err(q_m, q_r):
        a, b = placements(m, vm, q_m)[0], placements(ref, vr, q_r)[0]
        pe = max(np.linalg.norm(a[i][:3, 3] - b[k][:3, 3]) for i, k in enumerate(match))
        re_ = max(rot_angle(a[i][:3, :3].T @ b[k][:3, :3]) for i, k in enumerate(match))
        return pe, re_

    tol = 1e-6
    pe, re_ = err({}, {})
    if pe > tol or re_ > tol:
        rep.fail(f"at zero joint angles the meshes are NOT where the reference has them "
                 f"(max {pe * 1000:.3g} mm, {math.degrees(re_):.3g} deg)")
        return None
    rep.ok(f"zero pose: all {len(vm)} meshes exactly where the reference has them "
           f"(max {pe * 1e6:.2g} um)")
    names = sorted(act_m)
    flips, broken = [], []
    for jn in names:  # each joint alone: same direction, flipped, or different motion?
        if max(err({jn: 0.3}, {jn: 0.3})) < tol:
            continue
        (flips if max(err({jn: 0.3}, {jn: -0.3})) < tol else broken).append(jn)
    if broken:
        rep.fail("these joints move the robot differently from the reference: " + ", ".join(broken))
        return None
    if flips:
        rep.warn("positive direction flipped vs the reference (fine if intended, e.g. to match "
                 "the real motors): " + ", ".join(flips))
    sign = {jn: (-1.0 if jn in flips else 1.0) for jn in names}
    rng = np.random.default_rng(1)
    configs = [{}] + [{jn: float(rng.uniform(-0.6, 0.6)) for jn in names} for _ in range(25)]
    m_m = m.mass_props(m.fk())[0]
    m_r = ref.mass_props(ref.fk())[0]
    same_mass = abs(m_m - m_r) < 1e-6 * max(1.0, m_r)
    perr = rerr = cerr = ierr = 0.0
    for q in configs:
        qr = {jn: sign[jn] * v for jn, v in q.items()}
        pe, re_ = err(q, qr)
        perr, rerr = max(perr, pe), max(rerr, re_)
        if same_mass:
            _, W_m, A_m = placements(m, vm, q)
            _, W_r, A_r = placements(ref, vr, qr)
            _, c_m, I_m = m.mass_props(W_m)
            _, c_r, I_r = ref.mass_props(W_r)
            c_m = A_m[:3, :3] @ c_m + A_m[:3, 3]
            c_r = A_r[:3, :3] @ c_r + A_r[:3, 3]
            I_m = A_m[:3, :3] @ I_m @ A_m[:3, :3].T
            I_r = A_r[:3, :3] @ I_r @ A_r[:3, :3].T
            cerr = max(cerr, np.linalg.norm(c_m - c_r))
            ierr = max(ierr, np.abs(I_m - I_r).max() / np.abs(I_r).max())
    if perr < tol and rerr < tol:
        rep.ok(f"{len(configs)} random joint configurations: every mesh moves identically "
               f"(max error {perr * 1e6:.2g} um, {math.degrees(rerr) * 3600:.2g} arcsec)"
               + ("" if flips else "; all joint directions identical"))
    else:
        rep.fail(f"random configurations differ from the reference by up to {perr * 1000:.3g} mm")
    if same_mass:
        if cerr < tol and ierr < tol:
            rep.ok(f"total mass {m_m:.4f} kg, centre of mass and inertia identical in every "
                   f"configuration (max COM error {cerr * 1e6:.2g} um)")
        else:
            rep.fail(f"mass distribution differs (COM error {cerr * 1000:.3g} mm, "
                     f"relative inertia error {ierr:.2g})")
    else:
        rep.warn(f"total mass differs: {m_m:.4f} kg here vs {m_r:.4f} kg in the reference "
                 "(expected if you entered weighed masses)")
    changed = []
    for jn in sorted(act_m):
        a, b = m.joints[jn], ref.joints[jn]
        la, lb = a["limit"] or {}, b["limit"] or {}
        if a["type"] != b["type"] or la != lb:
            changed.append(f"{jn}: {b['type']} {lb or ''} -> {a['type']} {la}")
    if changed:
        Report.info("joint type/limit changes vs reference:")
        for c in changed:
            Report.info("  " + c)
    return flips


# ================================================================ MuJoCo
def mj_xml(urdf_path, mesh_dir, discard_visual, free_root=None, floor=False):
    r = ET.parse(urdf_path).getroot()
    for old in r.findall("mujoco"):
        r.remove(old)
    mj = ET.SubElement(r, "mujoco")
    ET.SubElement(mj, "compiler", meshdir=os.path.abspath(mesh_dir),
                  discardvisual="true" if discard_visual else "false",
                  fusestatic="true", strippath="true", balanceinertia="false")
    if free_root:
        w = ET.SubElement(r, "link", name="world")
        if floor:
            c = ET.SubElement(w, "collision", name="floor")
            ET.SubElement(c, "origin", xyz="0 0 -0.05", rpy="0 0 0")
            g = ET.SubElement(c, "geometry")
            ET.SubElement(g, "box", size="20 20 0.1")
        j = ET.SubElement(r, "joint", name="floating_base", type="floating")
        ET.SubElement(j, "parent", link="world")
        ET.SubElement(j, "child", link=free_root)
    return ET.tostring(r, encoding="unicode")


def geom_label(mujoco, model, g):
    body = model.body(model.geom_bodyid[g]).name or "world"
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        return f"{model.mesh(model.geom_dataid[g]).name} ({body})"
    return body


def mujoco_checks(rep, m, mesh_dir, args, ref_path, ref_flips=()):
    rep.section("8. MuJoCo")
    try:
        import mujoco
    except ImportError:
        rep.warn("mujoco is not installed (pip install mujoco): physics tests skipped")
        return
    try:
        model = mujoco.MjModel.from_xml_string(
            mj_xml(m.path, mesh_dir, True, free_root=m.root, floor=True))
    except Exception as e:  # noqa: BLE001
        rep.fail(f"MuJoCo {mujoco.__version__} cannot load the URDF: {e}")
        return
    hinge = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    urdf_mass = m.mass_props(m.fk())[0]
    rep.ok(f"MuJoCo {mujoco.__version__} loads it: {model.nbody - 1} rigid bodies after merging "
           f"fixed links, {len(hinge)} hinge joints + free base, {model.body_mass.sum():.3f} kg")
    if abs(model.body_mass.sum() - urdf_mass) > 1e-6:
        rep.warn(f"MuJoCo mass {model.body_mass.sum():.4f} kg != URDF mass {urdf_mass:.4f} kg")
    floor = [g for g in range(model.ngeom) if model.geom_bodyid[g] == 0]

    # ---- self-collision at the zero pose, robot lifted far above the floor
    data = mujoco.MjData(model)
    data.qpos[2] = 10.0
    mujoco.mj_forward(model, data)
    pairs = {}
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in floor or c.geom2 in floor:
            continue
        key = tuple(sorted((geom_label(mujoco, model, c.geom1), geom_label(mujoco, model, c.geom2))))
        pairs[key] = min(pairs.get(key, 0.0), c.dist)
    if not pairs:
        rep.ok("no self-collisions at the zero pose")
    else:
        rep.warn(f"{len(pairs)} pairs of collision meshes overlap at the zero pose "
                 "(MuJoCo collides the convex hull of each mesh):")
        for (a, b), dist in sorted(pairs.items(), key=lambda kv: kv[1]):
            Report.info(f"{a}  <->  {b}   depth {-dist * 1000:.1f} mm")
        Report.info("fix: exclude these pairs in the MJCF (<contact><exclude body1 body2/>), or "
                    "use simple box/capsule collision shapes for those parts")

    # ---- drop test: free base on a floor, joints held by stiff springs at zero
    data = mujoco.MjData(model)
    model.opt.timestep = 0.001
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for jid in hinge:
        model.jnt_stiffness[jid] = 300.0
        model.dof_damping[model.jnt_dofadr[jid]] = 5.0
        model.qpos_spring[model.jnt_qposadr[jid]] = 0.0
    mujoco.mj_forward(model, data)
    low = np.inf
    for g in range(model.ngeom):
        if g in floor or model.geom_contype[g] == 0:
            continue
        c, h = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        pts = (corners * h + c) @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        low = min(low, pts[:, 2].min())
    if not np.isfinite(low):
        rep.warn("drop test skipped: the robot has no collision geometry to stand on")
        return model
    data.qpos[2] += -low + 0.005
    h0 = data.qpos[2]
    mujoco.mj_forward(model, data)
    peak, finite = 0.0, True
    for step in range(3000):
        mujoco.mj_step(model, data)
        if step % 50 == 0:
            if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
                finite = False
                break
            peak = max(peak, np.abs(data.qvel[6:]).max())
    warns = {mujoco.mjtWarning(i).name: data.warning[i].number
             for i in range(len(data.warning)) if data.warning[i].number}
    unstable = {k: v for k, v in warns.items()
                if k in ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC")}
    if not finite or peak > 200 or unstable:
        rep.fail(f"drop test went unstable (peak joint speed {peak:.0f} rad/s, warnings "
                 f"{warns or 'none'}): usually overlapping collision meshes or a bad inertia")
        return model
    if data.qpos[2] < 0:
        rep.fail(f"robot sank through the floor (base at {data.qpos[2]:.2f} m): check that the "
                 "feet have collision geometry")
        return model
    w, x, y, z = data.qpos[3:7]
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
    rep.ok(f"3 s drop test is numerically stable (peak joint speed {peak:.1f} rad/s"
           + (f", simulator warnings {warns}" if warns else "") + ")")
    if tilt < 15 and data.qpos[2] > 0.8 * h0:
        Report.info(f"robot stands on its own with stiff joints: base height {data.qpos[2]:.3f} m, "
                    f"tilt {tilt:.1f} deg")
    else:
        Report.info(f"robot fell over (base height {data.qpos[2]:.3f} m, tilt {tilt:.0f} deg). "
                    "Not an error by itself: there is no balance controller")

    # ---- independent check of the reference equivalence with MuJoCo's own URDF parser
    if ref_path:
        try:
            mn = mujoco.MjModel.from_xml_string(mj_xml(m.path, mesh_dir, False))
            mr = mujoco.MjModel.from_xml_string(mj_xml(ref_path, mesh_dir, False))
        except Exception as e:  # noqa: BLE001
            rep.warn(f"could not load both files for the MuJoCo comparison: {e}")
        else:
            perr = rerr = 0.0
            rng = np.random.default_rng(2)
            names = [mn.joint(i).name for i in range(mn.njnt)
                     if mn.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]

            def mesh_labels(mm):
                return [mm.mesh(mm.geom_dataid[g]).name for g in range(mm.ngeom)
                        if mm.geom_group[g] == 1 and mm.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]

            la0, lb0 = mesh_labels(mn), mesh_labels(mr)
            common = sorted(l for l in set(la0) if la0.count(l) == 1 and lb0.count(l) == 1)
            root_mesh = [os.path.splitext(os.path.basename(g["filename"]))[0]
                         for g in m.links[m.root]["visuals"] if g["kind"] == "mesh"]
            anchor_label = next((l for l in root_mesh if l in common), common[0] if common else None)
            match = None
            for trial in range(10 if anchor_label else 0):
                q = {n: (0.0 if trial == 0 else float(rng.uniform(-0.6, 0.6))) for n in names}
                poses = []
                for mm in (mn, mr):
                    dd = mujoco.MjData(mm)
                    for n, v in q.items():
                        flip = -1.0 if (mm is mr and n in ref_flips) else 1.0
                        dd.qpos[mm.joint(n).qposadr[0]] = flip * v
                    mujoco.mj_kinematics(mm, dd)
                    gs = [g for g in range(mm.ngeom) if mm.geom_group[g] == 1
                          and mm.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
                    lab = [mm.mesh(mm.geom_dataid[g]).name for g in gs]
                    Ts = []
                    for g in gs:
                        T = np.eye(4)
                        T[:3, :3] = dd.geom_xmat[g].reshape(3, 3)
                        T[:3, 3] = dd.geom_xpos[g]
                        Ts.append(T)
                    Ai = inv(Ts[lab.index(anchor_label)])
                    poses.append((lab, [Ai @ T for T in Ts]))
                (la, Pa), (lb, Pb) = poses
                if sorted(la) != sorted(lb):
                    rep.warn("MuJoCo comparison: the two models have different mesh sets")
                    break
                if match is None:
                    match, used = [], set()
                    for i, l in enumerate(la):
                        k = min((k for k, l2 in enumerate(lb) if l2 == l and k not in used),
                                key=lambda k: np.linalg.norm(Pa[i][:3, 3] - Pb[k][:3, 3]))
                        used.add(k)
                        match.append(k)
                for i, k in enumerate(match):
                    perr = max(perr, np.linalg.norm(Pa[i][:3, 3] - Pb[k][:3, 3]))
                    rerr = max(rerr, rot_angle(Pa[i][:3, :3].T @ Pb[k][:3, :3]))
            else:
                if anchor_label is None:
                    rep.warn("MuJoCo comparison skipped: no mesh is used exactly once in both")
                elif perr < 1e-6 and rerr < 1e-6:
                    rep.ok(f"MuJoCo's own URDF parser agrees: every mesh identical to the "
                           f"reference in 10 configurations (max {perr * 1e6:.3g} um)")
                else:
                    rep.fail(f"MuJoCo sees a difference vs the reference: {perr * 1000:.3g} mm, "
                             f"{math.degrees(rerr):.3g} deg")
    return model


def view(m, mesh_dir):
    import mujoco
    import mujoco.viewer
    model = mujoco.MjModel.from_xml_string(mj_xml(m.path, mesh_dir, False, m.root, True))
    data = mujoco.MjData(model)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            model.jnt_stiffness[jid] = 300.0
            model.dof_damping[model.jnt_dofadr[jid]] = 5.0
    mujoco.mj_forward(model, data)
    low = min(((data.geom_xpos[g][2] - model.geom_rbound[g]) for g in range(model.ngeom)
               if model.geom_bodyid[g] != 0 and model.geom_contype[g]), default=-1.0)
    data.qpos[2] += -low
    print("viewer: geom group 0 = collision, 1 = visual (keys 0 / 1 toggle them)")
    mujoco.viewer.launch(model, data)


def plot(m, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: --plot skipped")
        return
    W = m.fk()

    def com(n):
        ine = m.links[n]["inertial"]
        return (W[n] @ ine["T"])[:3, 3] if ine else W[n][:3, 3]

    tot, c_all, _ = m.mass_props(W)
    act = [jn for jn, jj in m.joints.items() if jj["type"] in ACTUATED]
    feet = [m.mass_props(W, m.rigid_group(c))[1] for _, c in feet_of(m, act)]
    fig, axs = plt.subplots(1, 2, figsize=(11, 7.5), gridspec_kw={"width_ratios": [2, 1]})
    for ax, (i, j, lab) in zip(axs, [(1, 2, "front view: y (left) / z (up)"),
                                     (0, 2, "side view: x (forward) / z (up)")]):
        for n in m.order[1:]:
            p, jn = m.parent[n]
            a, b = com(p), com(n)
            ax.plot([a[i], b[i]], [a[j], b[j]], color="0.65", lw=1)
        for jn, jj in m.joints.items():
            if jj["type"] in ACTUATED:
                pt = W[jj["child"]][:3, 3]
                left = jn.startswith("left")
                ax.plot(pt[i], pt[j], "o", color="tab:blue" if left else "tab:red", ms=6)
                if i == 0 and left:
                    ax.annotate(jn[5:], (pt[i], pt[j]), fontsize=7, xytext=(5, 3),
                                textcoords="offset points")
        for f in feet:
            ax.plot(f[i], f[j], "s", color="tab:purple", ms=7)
        ax.plot(0, 0, "k+", ms=14, mew=2)
        ax.plot(c_all[i], c_all[j], "*", color="tab:green", ms=15)
        ax.set_title(lab, fontsize=10)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    handles = [plt.Line2D([], [], color="tab:blue", marker="o", ls="", label="left joint frames"),
               plt.Line2D([], [], color="tab:red", marker="o", ls="", label="right joint frames"),
               plt.Line2D([], [], color="tab:purple", marker="s", ls="", label="foot centres of mass"),
               plt.Line2D([], [], color="k", marker="+", ls="", label=f"base origin ({m.root})"),
               plt.Line2D([], [], color="tab:green", marker="*", ls="",
                          label=f"robot centre of mass ({tot:.2f} kg)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"{m.name} at zero joint angles: grey lines join each link's centre of mass to "
                 "its parent's (joint frames may sit anywhere along their axis)", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=130)
    print(f"skeleton plot written to {path}")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("urdf")
    ap.add_argument("--mesh-dir", help="folder holding the mesh files (default: search near the URDF)")
    ap.add_argument("--reference", help="original URDF to prove equivalence against")
    ap.add_argument("--expect-root", help="link that must be the root, e.g. torso_ss")
    ap.add_argument("--no-mujoco", action="store_true", help="skip the MuJoCo tests")
    ap.add_argument("--view", action="store_true", help="open the MuJoCo viewer at the end")
    ap.add_argument("--save-mjcf", help="also save MuJoCo's MJCF version (free base + floor)")
    ap.add_argument("--plot", help="write a skeleton picture (PNG)")
    args = ap.parse_args()

    rep = Report()
    print(f"verifying {args.urdf}")
    try:
        m = Model(args.urdf)
    except (ET.ParseError, OSError) as e:
        rep.section("1. File")
        rep.fail(f"cannot read the URDF: {e}")
        return finish(rep)
    tree_ok = check_structure(rep, m, args.expect_root)
    act = check_joints(rep, m)
    check_inertia(rep, m)
    mesh_dir = check_meshes(rep, m, os.path.abspath(args.mesh_dir) if args.mesh_dir else None)
    zup = False
    if tree_ok:
        check_collisions(rep, m, act)
        zup = check_mass_balance(rep, m, act)
        check_axes(rep, m, act, zup)
        flips = None
        if args.reference:
            try:
                ref = Model(args.reference)
                ref.build_tree()
                flips = check_reference(rep, m, ref)
            except Exception as e:  # noqa: BLE001
                rep.section("7. Equivalence with the reference URDF")
                rep.fail(f"cannot use the reference file: {e}")
        if shutil.which("check_urdf"):
            res = subprocess.run(["check_urdf", args.urdf], capture_output=True, text=True)
            (rep.ok if res.returncode == 0 else rep.fail)(
                "ROS check_urdf " + ("accepts the file" if res.returncode == 0 else
                                     "rejects the file:\n" + res.stdout + res.stderr))
        if args.plot:
            plot(m, args.plot)
    if tree_ok and not args.no_mujoco:
        if mesh_dir:
            ref_ok = args.reference if flips is not None else None
            model = mujoco_checks(rep, m, mesh_dir, args, ref_ok, flips or ())
            if model is not None and args.save_mjcf:
                import mujoco
                full = mujoco.MjModel.from_xml_string(mj_xml(m.path, mesh_dir, False, m.root, True))
                mujoco.mj_saveLastXML(args.save_mjcf, full)
                print(f"MJCF written to {args.save_mjcf}")
        else:
            rep.section("8. MuJoCo")
            rep.warn("MuJoCo tests need the meshes in one folder: pass --mesh-dir")
    code = finish(rep)
    if args.view and tree_ok and mesh_dir:
        view(m, mesh_dir)
    return code


def finish(rep):
    c = rep.counts
    print("\n" + "=" * 80)
    verdict = "FAIL" if c["FAIL"] else "PASS"
    print(f"RESULT: {verdict}   ({c['PASS']} passed, {c['WARN']} warnings, {c['FAIL']} failures)")
    for f in rep.failures:
        print(f"  FAIL  {f.splitlines()[0]}")
    for w in rep.warnings:
        print(f"  WARN  {w.splitlines()[0]}")
    return 1 if c["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())