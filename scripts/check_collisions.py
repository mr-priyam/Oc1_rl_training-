"""
check_collisions.py - verify the collision geometry of oc1_bipedal_collision.urdf in MuJoCo.

Usage:
    python check_collisions.py oc1_bipedal_collision.urdf --meshdir meshes
    python check_collisions.py oc1_bipedal_collision.urdf --meshdir meshes --view
    python check_collisions.py oc1_bipedal_collision.urdf --meshdir meshes --effort 60

Checks:
  1. Inventory     - every rigid body has collision geoms.
  2. Rest pose     - robot in the air at zero joint angles must have ZERO contacts.
  3. Joint sweep   - which body pairs touch at the joint limits (single joints and both legs
                     together). Collision is working if contacts appear where parts meet.
  4. Safe limits   - for every joint, the angle where it first touches another part, and a
                     suggested URDF limit a small margin before that.
  5. Drop test     - drops the robot on a floor, joints held at 0 by PD. Tells apart
                     "fell through floor" (collision broken), "collapsed" (joints too weak)
                     and "tipped over" (balance). If it collapses and the URDF effort limits
                     are tiny, it re-runs without the cap to confirm the cause.

Options:
  --effort N    override every joint's torque limit with N Nm (URDF effort="1" caps at 1 Nm)
  --armature A  joint armature in kg*m^2 (default 0.02, stands in for rotor inertia)
  --margin R    safety margin in rad for the suggested limits (default 0.05 rad ~ 3 deg)
  --view        open the MuJoCo viewer on the drop test (C = contacts, F = forces)
"""
import argparse
import itertools
import os
import re
import sys

import numpy as np
import mujoco
import mujoco.viewer

FEET = {"lf", "rf"}


# ============================================================================ loading
def load_spec(urdf_path, meshdir):
    urdf_path = os.path.abspath(urdf_path)
    text = open(urdf_path).read()
    # discardvisual=true -> only <collision> geoms are kept, so what you see is what collides.
    compiler = (f'<mujoco><compiler meshdir="{meshdir}" discardvisual="true" '
                f'fusestatic="true" strippath="true"/></mujoco>')
    text = re.sub(r'(<robot[^>]*>)', r'\1\n' + compiler, text, count=1)
    tmp = os.path.join(os.path.dirname(urdf_path), "_mj_tmp.urdf")
    with open(tmp, "w") as f:
        f.write(text)
    try:
        spec = mujoco.MjSpec.from_file(tmp)
    finally:
        os.remove(tmp)
    return spec


def build_model(urdf, meshdir, kp=150.0, kv=5.0, armature=0.02):
    spec = load_spec(urdf, meshdir)
    # implicitfast integrates joint damping stably; armature ~ reflected rotor inertia of a
    # geared actuator. Without these, light links + PD damping make the sim blow up (NaN).
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[5, 5, 0.1], rgba=[0.3, 0.3, 0.35, 1])
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
    spec.worldbody.first_body().add_freejoint(name="root")
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
            j.armature = armature
            a = spec.add_actuator(name=f"pd_{j.name}", target=j.name,
                                  trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_position(kp=kp, kv=kv)
    m = spec.compile()
    free_adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root")]
    return m, free_adr


# ============================================================================ helpers
def nm(m, obj, i):
    return mujoco.mj_id2name(m, obj, i) or f"#{i}"


def hinges(m):
    return [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]


def contact_pairs(m, d, skip_floor=True):
    """{(bodyA, bodyB): max penetration in mm} for the current contacts."""
    out = {}
    for c in d.contact[:d.ncon]:
        g1, g2 = c.geom1, c.geom2
        if skip_floor and "floor" in (nm(m, mujoco.mjtObj.mjOBJ_GEOM, g1),
                                      nm(m, mujoco.mjtObj.mjOBJ_GEOM, g2)):
            continue
        key = tuple(sorted((nm(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g1]),
                            nm(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g2]))))
        out[key] = max(out.get(key, 0.0), -c.dist * 1000)
    return out


def lowest_point(m, d):
    zmin = np.inf
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        zmin = min(zmin, (v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min())
    return zmin


def pose_in_air(m, d, free_adr, joint_values=None, height=3.0):
    """Zero pose (plus given joint values) with the robot high in the air, then forward."""
    d.qpos[:] = m.qpos0
    d.qpos[free_adr + 2] = height
    for adr, q in (joint_values or {}).items():
        d.qpos[adr] = q
    mujoco.mj_forward(m, d)


def feet_clearance_height(m, d, free_adr, clearance):
    """Root height that puts the lowest point `clearance` above the floor at zero pose."""
    pose_in_air(m, d, free_adr, height=0.0)
    return -lowest_point(m, d) + clearance


# ============================================================================ checks
def check_inventory(m):
    print("\n[1] INVENTORY - collision geoms per rigid body (after fixed joints are merged)")
    bad = 0
    for b in range(1, m.nbody):
        n = sum(1 for g in range(m.ngeom)
                if m.geom_bodyid[g] == b and m.geom_contype[g] | m.geom_conaffinity[g])
        bad += n == 0
        print(f"    {nm(m, mujoco.mjtObj.mjOBJ_BODY, b):28s} {n:2d} geoms"
              + ("   <-- NO COLLISION GEOMS" if n == 0 else ""))
    print("    PASS" if not bad else f"    FAIL: {bad} bodies have no collision geometry")
    return bad == 0


def check_rest_pose(m, d, free_adr):
    print("\n[2] REST POSE - zero joint angles, robot in the air")
    pose_in_air(m, d, free_adr)
    pairs = contact_pairs(m, d)
    if not pairs:
        print("    PASS: no self-contacts at rest")
        return True
    print("    FAIL: these body pairs overlap at rest (false contacts):")
    for (a, b), pen in sorted(pairs.items(), key=lambda x: -x[1]):
        print(f"      {a:28s} <-> {b:28s} {pen:6.1f} mm")
    print("    Fix: <contact><exclude body1=\"..\" body2=\"..\"/></contact>, or a smaller "
          "primitive instead of the mesh.")
    return False


def check_sweep(m, d, free_adr, steps=11):
    print("\n[3] JOINT SWEEP - kinematic (joints are forced to the angle, nothing pushes back),")
    print("    so 'mm' = how far PAST first contact that pose goes, not a simulation result")
    found = {}

    def test(desc, jv):
        pose_in_air(m, d, free_adr, jv)
        for pair, pen in contact_pairs(m, d).items():
            if pair not in found or pen > found[pair][0]:
                found[pair] = (pen, desc)

    for j in hinges(m):
        adr, (lo, hi) = m.jnt_qposadr[j], m.jnt_range[j]
        for q in np.linspace(lo, hi, steps):
            test(f"{nm(m, mujoco.mjtObj.mjOBJ_JOINT, j)}={q:+.2f}", {adr: q})

    def jid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    for l, r in [("left_hip_roll", "right_hip_roll"), ("left_hip_yaw", "right_hip_yaw"),
                 ("left_knee_pitch", "right_knee_pitch")]:
        if jid(l) < 0 or jid(r) < 0:
            continue
        for sl, sr in itertools.product([0, 1], repeat=2):
            ql, qr = m.jnt_range[jid(l)][sl], m.jnt_range[jid(r)][sr]
            test(f"{l}={ql:+.2f}, {r}={qr:+.2f}",
                 {m.jnt_qposadr[jid(l)]: ql, m.jnt_qposadr[jid(r)]: qr})

    if not found:
        print("    No self-contacts in the tested range.")
        return
    print("    Self-contacts detected (collision is working):")
    for (a, b), (pen, pose) in sorted(found.items(), key=lambda x: -x[1][0]):
        both = "," in pose
        tag = "  [both legs]" if both else ""
        print(f"      {a:26s} <-> {b:26s} {pen:6.1f} mm at {pose}{tag}")
    if any("," in p for _, p in found.values()):
        print("    [both legs] = legs hit each other only when BOTH move that way together.")
        print("    Joint limits can't prevent that; keep self-collision ON in sim/RL so it's")
        print("    physically blocked (or add a leg-leg contact penalty to the reward).")


def check_safe_limits(m, d, free_adr, margin, resolution=0.002):
    print(f"\n[4] SAFE LIMITS - angle where each joint alone first hits another part "
          f"(margin {margin:.3f} rad)")
    print(f"    {'joint':20s} {'side':5s} {'URDF limit':>10s} {'1st contact':>12s} "
          f"{'suggested':>10s}  hits")
    suggestions = {}
    for j in hinges(m):
        jn = nm(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        adr, rng = m.jnt_qposadr[j], m.jnt_range[j].copy()
        new = rng.copy()
        for side, limit in (("lower", rng[0]), ("upper", rng[1])):
            n = max(2, int(abs(limit) / resolution) + 1)
            hit = None
            for q in np.linspace(0.0, limit, n)[1:]:
                pose_in_air(m, d, free_adr, {adr: q})
                pairs = contact_pairs(m, d)
                if pairs:
                    hit = (q, pairs)
                    break
            if hit is None:
                print(f"    {jn:20s} {side:5s} {limit:+10.3f} {'none':>12s} {'keep':>10s}")
                continue
            q, pairs = hit
            sugg = q - np.sign(q) * margin
            if np.sign(sugg) != np.sign(q):
                sugg = 0.0
            new[0 if side == "lower" else 1] = sugg
            who = ", ".join(sorted({f"{a}<->{b}" for a, b in pairs}))
            print(f"    {jn:20s} {side:5s} {limit:+10.3f} {q:+12.3f} {sugg:+10.3f}  {who}")
        if not np.allclose(new, rng):
            suggestions[jn] = new
    if suggestions:
        print("\n    Paste into the URDF (replace the lower/upper of these joints):")
        for jn, (lo, hi) in suggestions.items():
            print(f'      {jn:20s} lower="{lo:.3f}" upper="{hi:.3f}"')
    else:
        print("    All limits are already collision-free.")
    return suggestions


def simulate_drop(m, d, free_adr, seconds=3.0):
    mujoco.mj_resetData(m, d)          # clean state (time, warmstart, velocities)
    d.qpos[free_adr + 2] = feet_clearance_height(m, d, free_adr, 0.05)
    d.qvel[:] = 0
    d.ctrl[:] = 0
    mujoco.mj_forward(m, d)
    z0 = d.qpos[free_adr + 2]
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    touching, impact_pen, settled_pen = set(), 0.0, 0.0
    nsteps = int(seconds / m.opt.timestep)
    unstable = False
    for i in range(nsteps):
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qacc)) or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            unstable = True
            break
        for c in d.contact[:d.ncon]:
            if floor in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == floor else c.geom1
                touching.add(nm(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[other]))
                pen = -c.dist * 1000
                impact_pen = max(impact_pen, pen)
                if i > nsteps - int(0.5 / m.opt.timestep):
                    settled_pen = max(settled_pen, pen)
    root_body = m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root")]
    tilt = np.degrees(np.arccos(np.clip(d.xmat[root_body][8], -1, 1)))
    joint_err = max(abs(d.qpos[m.jnt_qposadr[j]]) for j in hinges(m))
    return dict(z0=z0, z1=d.qpos[free_adr + 2], low=lowest_point(m, d),
                touching=touching, impact_pen=impact_pen, settled_pen=settled_pen,
                tilt=tilt, joint_err=joint_err, unstable=unstable)


def classify(r):
    if r["unstable"]:
        return "unstable"
    if not r["touching"] or r["low"] < -0.02:
        return "through"
    if r["joint_err"] > 0.2:          # PD could not hold the joints at 0
        return "collapsed"
    if r["tilt"] > 30 or not r["touching"] <= FEET:
        return "tipped"
    return "standing"


def report_drop(r, label):
    verdict = classify(r)
    print(f"    [{label}]")
    print(f"      root height {r['z0']:.3f} -> {r['z1']:.3f} m, lowest point {r['low']*1000:+.1f} mm")
    print(f"      floor penetration: impact {r['impact_pen']:.1f} mm, "
          f"settled {r['settled_pen']:.1f} mm")
    print(f"      torso tilt {r['tilt']:.0f} deg, worst joint error {r['joint_err']:.2f} rad "
          f"(joints should stay at 0)")
    print(f"      touched floor: {sorted(r['touching']) or 'NONE'}")
    msg = {
        "through": "FAIL - fell through the floor: collision is NOT active",
        "collapsed": "collision OK - but joints COLLAPSED (actuators could not hold the weight)",
        "tipped": "collision OK - joints held, but the robot tipped over (balance: CoM outside feet, "
                  "no balance controller)",
        "standing": "PASS - standing on its feet",
        "unstable": "simulation blew up (NaN) - lower PD gains or the timestep; not a collision result",
    }[verdict]
    print(f"      -> {msg}")
    return verdict


def check_drop(m, d, free_adr, effort_override):
    print("\n[5] DROP TEST - 5 cm above floor, joints held at 0 by PD (kp=150)")
    caps = [m.jnt_actfrcrange[j][1] if m.jnt_actfrclimited[j] else np.inf for j in hinges(m)]
    if effort_override is not None:
        for j in hinges(m):
            m.jnt_actfrclimited[j] = 1
            m.jnt_actfrcrange[j] = [-effort_override, effort_override]
        label = f"effort overridden to {effort_override:g} Nm"
    else:
        label = f"URDF effort limits (max {max(caps):g} Nm)"
    r = simulate_drop(m, d, free_adr)
    verdict = report_drop(r, label)

    if verdict == "collapsed" and effort_override is None and max(caps) < 10:
        print("      Cause check: URDF effort limits are tiny, re-running with the cap removed...")
        saved = m.jnt_actfrclimited.copy()
        m.jnt_actfrclimited[:] = 0
        r2 = simulate_drop(m, d, free_adr)
        v2 = report_drop(r2, "torque cap removed (diagnostic only)")
        m.jnt_actfrclimited[:] = saved
        if v2 in ("standing", "tipped"):
            print("      DIAGNOSIS: the collapse is caused by effort=\"1\" in the URDF (1 Nm cap).")
            print("      Put the real RS03/RS04 torque ratings into each joint's effort=\"...\",")
            print("      or re-run with --effort <Nm> to test.")
        elif v2 == "collapsed":
            print("      Still collapses without the cap -> raise PD gains or check inertias.")
    elif verdict == "tipped":
        print("      A rigid robot with no balance controller usually tips; that is expected")
        print("      and does not mean collision is wrong.")
    return verdict not in ("through", "unstable")


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf")
    ap.add_argument("--meshdir", default="meshes", help="STL folder, relative to the URDF")
    ap.add_argument("--effort", type=float, default=None,
                    help="override every joint torque limit (Nm)")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="safety margin (rad) for suggested limits")
    ap.add_argument("--armature", type=float, default=0.02,
                    help="joint armature kg*m^2 (reflected rotor inertia), default 0.02")
    ap.add_argument("--view", action="store_true", help="open viewer on the drop test")
    a = ap.parse_args()

    m, free_adr = build_model(a.urdf, a.meshdir, armature=a.armature)
    d = mujoco.MjData(m)
    print(f"Loaded: {m.nbody-1} bodies, {m.ngeom-1} collision geoms, {m.nu} PD actuators")

    ok1 = check_inventory(m)
    ok2 = check_rest_pose(m, d, free_adr)
    check_sweep(m, d, free_adr)
    check_safe_limits(m, d, free_adr, a.margin)
    ok5 = check_drop(m, d, free_adr, a.effort)
    print("\nSUMMARY: collision geometry",
          "works" if (ok1 and ok2 and ok5) else "has problems - see FAIL lines above")

    if a.view:
        if a.effort is None:
            m.jnt_actfrclimited[:] = 0   # so the robot can actually hold itself in the viewer
            print("Viewer: torque cap removed so the PD can hold the pose.")
        d.qpos[:] = m.qpos0
        d.qpos[free_adr + 2] = feet_clearance_height(m, d, free_adr, 0.05)
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        print("Viewer: C = contact points, F = contact forces, T = transparent, "
              "Space = pause, Backspace = reset")
        mujoco.viewer.launch(m, d)


if __name__ == "__main__":
    sys.exit(main())