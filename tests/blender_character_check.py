"""Measure the character layer inside Blender. Run headless:

    blender -b --factory-startup --python tests/blender_character_check.py

§11 says measure the artifact rather than trusting the call. So this does not
check that ``character()`` returned without raising - it reads the *evaluated*
mesh out of the depsgraph and asserts that the body is the height it was asked
for, that posing a bone moves the vertices weighted to it, that the jaw shape key
moves the mouth downward, that a bound garment follows the armature, and that a
style pack swap leaves no role-tagged placeholder with users.

Exits non-zero on failure.
"""

import math
import os
import sys
import types

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FAILURES = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)
    return bool(condition)


def install():
    """Load prelude.py the way session.install_prelude does: as ``bstudio``."""
    path = os.path.join(ROOT, "blender_ai_studio", "prelude.py")
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()
    module = types.ModuleType("bstudio")
    module.__file__ = path
    exec(compile(source, path, "exec"), module.__dict__)
    sys.modules["bstudio"] = module
    return module


def evaluated(ob, frame=None):
    """World-space vertex positions after modifiers, at ``frame``."""
    if frame is not None:
        bpy.context.scene.frame_set(int(frame))
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    copy = ob.evaluated_get(graph)
    mesh = copy.to_mesh()
    matrix = copy.matrix_world
    out = [matrix @ vertex.co.copy() for vertex in mesh.vertices]
    copy.to_mesh_clear()
    return out


def measure_body(B, spec):
    """Build one character and assert what the pipeline depends on."""
    name, kind, preset, height, beard = spec
    built = B.character(name, height_m=height, preset=preset, kind=kind,
                        beard=beard, clothed=True)
    mesh, rig = built["mesh"], built["rig"]

    points = evaluated(mesh)
    low = min(p.z for p in points)
    high = max(p.z for p in points)
    tall = high - low
    check(abs(tall - height) < 0.02,
          f"{name}: measured {tall:.3f} m, asked for {height:.3f} m")
    check(abs(low) < 0.02, f"{name}: feet at z={low:.3f}, expected the floor")

    triangles = sum(len(p.vertices) - 2 for p in mesh.data.polygons)
    check(triangles > 0, f"{name}: no geometry")
    check(triangles < 400_000, f"{name}: {triangles} tris is over the gate")

    expected_bones = (len(B.QUAD_JOINTS) if kind == "quadruped"
                      else len(B.SMPLX_JOINTS))
    check(len(rig.data.bones) == expected_bones,
          f"{name}: {len(rig.data.bones)} bones, expected {expected_bones}")

    keys = mesh.data.shape_keys
    check(keys is not None, f"{name}: no shape keys at all")
    blocks = keys.key_blocks
    missing = [c for c in B.VISEME_CHANNELS if c not in blocks]
    check(not missing, f"{name}: missing shape keys {missing}")

    bones = {b.name for b in rig.data.bones}
    groups = {g.index: g.name for g in mesh.vertex_groups}
    unweighted = sum(
        1 for v in mesh.data.vertices
        if not any(groups.get(e.group) in bones and e.weight > 0.0
                   for e in v.groups))
    check(unweighted == 0, f"{name}: {unweighted} vertices weighted to no bone")

    slots = [m.get("style_role") for m in mesh.data.materials]
    check(all(slots), f"{name}: material slot without a style_role tag: {slots}")
    return built


def check_motion(B, built):
    """Pose one shoulder over twelve frames and watch the hand move."""
    name = built["name"]
    joint = "left_shoulder" if built["kind"] == "human" else "neck"
    poses = [{joint: (0.0, 0.0, 0.9 * (index / 11.0))} for index in range(12)]
    stats = B.apply_motion(built, poses, start=1, frames=24, retime="loop")
    check(stats["bones"] == 1, f"{name}: keyed {stats['bones']} bones, wanted 1")
    check(stats["frames"] > 0, f"{name}: no frames written")

    rest = evaluated(built["mesh"], frame=1)
    posed = evaluated(built["mesh"], frame=12)
    moved = max((a - b).length for a, b in zip(rest, posed))
    check(moved > 0.02,
          f"{name}: posing {joint} moved the mesh {moved:.4f} m - the armature "
          "modifier or the weights are not connected")
    return moved


def _bone_angle(rig, joint, frame):
    """The bone's rotation away from rest at one frame, in degrees.

    Read off the pose after a frame change rather than off the action, because
    what a shot renders is the evaluated pose and that is the only thing worth
    asserting about.
    """
    bpy.context.scene.frame_set(int(frame))
    bpy.context.view_layer.update()
    quaternion = rig.pose.bones[joint].rotation_quaternion
    return 2.0 * math.degrees(math.acos(min(1.0, abs(quaternion.w))))


def _ramp(joint, frames, span=0.9):
    """A take that only ever goes one way, so a retime is readable from one
    number: the angle at a frame says which source frame it came from."""
    return [{joint: (0.0, 0.0, span * index / (frames - 1.0))}
            for index in range(frames)]


def check_retime(B, built):
    """All four §5.2 retime policies, measured on the posed rig.

    `trim` and `timewarp` were declared by shot.schema.json and implemented by
    nobody: both fell through apply_motion's else branch and behaved as
    `hold_last_pose`, which is invisible unless you measure the frames. So each
    policy is checked for the thing that distinguishes it from the others, and
    the three that are not `hold_last_pose` are checked for *not* being it.
    """
    name, rig = built["name"], built["rig"]
    joint = "left_shoulder" if built["kind"] == "human" else "neck"
    source, window = 12, 24
    poses = _ramp(joint, source)
    results = {}

    def apply(policy):
        rig.animation_data_clear()
        return B.apply_motion(built, poses, start=1, frames=window,
                              retime=policy)

    # loop: the window is two whole cycles, so frame 13 restarts the take.
    stats = apply("loop")
    at = {f: _bone_angle(rig, joint, f) for f in (1, 12, 13, 24)}
    check(abs(at[13] - at[1]) < 0.5,
          f"{name}/loop: frame 13 is {at[13]:.2f} deg against frame 1's "
          f"{at[1]:.2f} - the cycle did not restart")
    check(at[12] > at[1] + 10.0, f"{name}/loop: the take did not play")
    check(stats["cycles"] == 2.0,
          f"{name}/loop: reported {stats['cycles']} cycles, wanted 2")
    results["loop"] = at

    # hold_last_pose: the final pose is frozen, and keyed on the last frame so
    # the hold is in the file rather than in Blender's extrapolation.
    stats = apply("hold_last_pose")
    at = {f: _bone_angle(rig, joint, f) for f in (1, 12, 18, 24)}
    check(abs(at[24] - at[12]) < 0.5,
          f"{name}/hold_last_pose: frame 24 is {at[24]:.2f} deg against the "
          f"take's last frame at {at[12]:.2f}")
    check(_last_key_frame(rig) == window,
          f"{name}/hold_last_pose: the last key is at frame "
          f"{_last_key_frame(rig)}, so the hold is not written down")
    check(stats["held_frames"] == window - source,
          f"{name}/hold_last_pose: reported {stats['held_frames']} held frames")
    results["hold_last_pose"] = at

    # trim: the take ends. Nothing is keyed past its last frame - which is the
    # whole difference from hold_last_pose, and is invisible in the pose alone.
    stats = apply("trim")
    check(_last_key_frame(rig) == source,
          f"{name}/trim: the last key is at frame {_last_key_frame(rig)}, "
          f"expected {source} - trim is still behaving as hold_last_pose")
    check(stats["trimmed_frames"] == 0,
          f"{name}/trim: reported {stats['trimmed_frames']} trimmed on a take "
          "shorter than the window")

    # trim the other way round: a take longer than the window really is cut.
    rig.animation_data_clear()
    long_take = _ramp(joint, 40)
    stats = B.apply_motion(built, long_take, start=1, frames=window,
                           retime="trim")
    check(_last_key_frame(rig) == window,
          f"{name}/trim: a 40-frame take in a 24-frame window ended at "
          f"{_last_key_frame(rig)}")
    check(stats["trimmed_frames"] == 40 - window,
          f"{name}/trim: reported {stats['trimmed_frames']} discarded frames, "
          f"expected {40 - window}")

    # timewarp: the take is stretched onto the window. The last output frame is
    # the last source frame, and the middle of the window is the middle of the
    # take - neither of which a hold or a loop produces.
    stats = apply("timewarp")
    at = {f: _bone_angle(rig, joint, f) for f in (1, 12, 13, 24)}
    ends = _bone_angle(rig, joint, window)
    check(abs(at[1]) < 0.5, f"{name}/timewarp: frame 1 is not the rest pose")
    check(ends > 40.0,
          f"{name}/timewarp: the window ends at {ends:.2f} deg; the take's own "
          "last pose is about 51.6 deg, so it was not stretched to fit")
    midpoint = (at[12] + at[13]) / 2.0
    check(abs(midpoint - ends / 2.0) < 1.5,
          f"{name}/timewarp: the middle of the window is {midpoint:.2f} deg "
          f"against half the take at {ends / 2.0:.2f} - the warp is not linear")
    check(abs(stats["speed"] - (source - 1) / float(window - 1)) < 5e-4,
          f"{name}/timewarp: reported speed {stats['speed']}, expected about "
          f"{(source - 1) / float(window - 1):.4f}")
    check(_key_count(rig) >= window,
          f"{name}/timewarp: {_key_count(rig)} keys for {window} frames; a warp "
          "keys every frame because no two of them are the same pose")
    results["timewarp"] = at

    # The bug this whole function exists for: an unknown policy used to become
    # hold_last_pose in silence.
    rig.animation_data_clear()
    raised = False
    try:
        B.apply_motion(built, poses, start=1, frames=window, retime="reverse")
    except ValueError:
        raised = True
    check(raised, f"{name}: an unknown retime policy was accepted silently")

    rig.animation_data_clear()
    return results


def _fcurves(rig):
    action = getattr(getattr(rig, "animation_data", None), "action", None)
    if action is None:
        return []
    module = sys.modules["bstudio"]
    return module._fcurves(action)


def _last_key_frame(rig):
    frames = [point.co[0] for curve in _fcurves(rig)
              for point in curve.keyframe_points]
    return int(round(max(frames))) if frames else 0


def _key_count(rig):
    return max((len(curve.keyframe_points) for curve in _fcurves(rig)),
               default=0)


def check_visemes(B, built, pack):
    """Ramp jaw_open and confirm the mouth vertices actually drop."""
    name = built["name"]
    frames = 16
    curves = {"jaw_open": [index / (frames - 1.0) for index in range(frames)],
              "lip_round": [0.0] * frames}
    stats = B.apply_visemes(built, curves, start=1, profile=pack.visemes)
    check("jaw_open" in stats["channels"], f"{name}: jaw_open was not keyed")

    mesh = built["mesh"]
    group = mesh.vertex_groups.get("face_mouth")
    check(group is not None, f"{name}: no face_mouth vertex group")
    heaviest, best = None, 0.0
    for vertex in mesh.data.vertices:
        for entry in vertex.groups:
            if entry.group == group.index and entry.weight > best:
                heaviest, best = vertex.index, entry.weight
    check(heaviest is not None, f"{name}: face_mouth group is empty")

    shut = evaluated(mesh, frame=1)[heaviest]
    open_ = evaluated(mesh, frame=frames)[heaviest]
    drop = shut.z - open_.z
    check(drop > 0.004,
          f"{name}: jaw_open moved the mouth {drop * 1000:.2f} mm downward")
    return drop, stats["peaks"].get("jaw_open")


def check_garment(B, built):
    """A shell over the torso, bound by proximity, must follow the armature."""
    name, rig = built["name"], built["rig"]
    joints, dims = built["joints"], built["dims"]
    waist = joints["spine1"]
    chest = joints["spine3"]
    verts, faces = B._prism(
        (waist[0], waist[1], waist[2] - 0.05 * dims["factor"]),
        (chest[0], chest[1], chest[2] + 0.06 * dims["factor"]),
        0.175 * dims["girth"], 0.205 * dims["girth"], sides=14)
    mesh = bpy.data.meshes.new(name + "_cuirass")
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.update()
    mesh.materials.append(B.role_material("metal_armor"))
    cloth = bpy.data.objects.new(name + "_cuirass", mesh)
    bpy.context.collection.objects.link(cloth)

    stats = B.bind_garment(built, cloth, offset=0.004)
    check(stats["bound"] >= stats["of"] * 0.95,
          f"{name}: bound {stats['bound']} of {stats['of']} garment vertices")
    # bind_garment reports the bone *names* it bound to, because that is what
    # the follow test has to pose. A count cannot say which.
    check(isinstance(stats["bones"], list),
          f"{name}: bind_garment reported bones as {type(stats['bones']).__name__}")
    check(len(stats["bones"]) >= 2,
          f"{name}: garment reached only {len(stats['bones'])} bone(s)")
    check(all(rig.pose.bones.get(b) for b in stats["bones"]),
          f"{name}: bind_garment named a bone the rig has not got")

    # This check keys its own pose. It used to measure between frames 1 and 12
    # of whatever action check_motion happened to leave on the rig, so any
    # later check that cleared the rig's animation - as the retime checks must -
    # made a perfectly bound garment report zero movement.
    # Pose a bone the garment is actually bound to, which is what build_garment
    # does. Posing a fixed bone works for a cuirass and is meaningless for a
    # boot, and that is exactly how a correctly bound sandal came to fail its
    # gate.
    joint = stats["bones"][0]
    B.apply_motion(built,
                   [{joint: (0.0, 0.0, 0.9 * index / 11.0)}
                    for index in range(12)],
                   start=1, frames=12, retime="trim")

    rest = evaluated(cloth, frame=1)
    posed = evaluated(cloth, frame=12)
    moved = max((a - b).length for a, b in zip(rest, posed))
    check(moved > 0.002,
          f"{name}: the garment moved {moved:.4f} m with the body")
    return stats, moved


def check_style(B, pack_name):
    """A pack swap must leave no role placeholder in use."""
    applied = B.style(pack_name, repo_root=ROOT, target=(0.0, 0.0, 1.0),
                      postfx=False)
    check(applied.get("materials"),
          f"{pack_name}: swapped no materials")
    stale = [m.name for m in bpy.data.materials
             if m.name.startswith("role_") and m.users > 0]
    check(not stale, f"{pack_name}: placeholders still in use: {stale}")
    for material in bpy.data.materials:
        role = material.get("style_role")
        if role and material.users:
            check(material.name.startswith(pack_name + "_"),
                  f"{pack_name}: {material.name} carries role {role} but is "
                  "not from this pack")
    check(applied.get("lights") or applied.get("world"),
          f"{pack_name}: no lighting was built")
    return applied


CAST = (
    # name, kind, preset, height_m, beard
    ("rostam", "human", "heroic", 2.10, True),
    ("sohrab", "human", "youth", 2.00, False),
    ("rakhsh", "quadruped", "giant", 1.95, False),
)


def main():
    B = install()
    from blender_ai_studio import styles                          # noqa: E402
    B.reset(engine="EEVEE", res=(640, 360), fps=24)
    pack = styles.load("miniature")

    built = {}
    for spec in CAST:
        built[spec[0]] = measure_body(B, spec)
        body = built[spec[0]]
        print(f"  {spec[0]:<8} {body['verts']:5d} verts  {body['faces']:5d} faces  "
              f"{len(body['roles'])} roles  {len(body['bones'])} bones")

    # Visemes before motion: with no armature action in the way, a mouth vertex
    # that drops can only have been moved by the shape key.
    for name in ("rostam", "sohrab"):
        drop, peak = check_visemes(B, built[name], pack)
        print(f"  {name:<8} jaw_open peak {peak}  mouth drop "
              f"{drop * 1000:.2f} mm")

    for name, body in built.items():
        moved = check_motion(B, body)
        print(f"  {name:<8} pose moved the mesh {moved * 100:.2f} cm")

    # All four §5.2 retime policies, on one rig. trim and timewarp were
    # declared by shot.schema.json and implemented by nobody until h_assemble
    # needed them.
    angles = check_retime(B, built["rostam"])
    for policy in ("loop", "hold_last_pose", "timewarp"):
        row = angles.get(policy) or {}
        print(f"  retime {policy:<15} "
              + "  ".join(f"f{frame}={value:5.1f}deg"
                          for frame, value in sorted(row.items())))

    stats, moved = check_garment(B, built["rostam"])
    print(f"  garment  {stats['bound']}/{stats['of']} verts on "
          f"{len(stats['bones'])} bones, moved {moved * 100:.2f} cm")

    for name in styles.PACKS:
        applied = check_style(B, name)
        print(f"  {name:<10} {applied.get('engine')}  "
              f"{len(applied.get('materials') or {})} roles  "
              f"world={applied.get('world')}  "
              f"lights={','.join(applied.get('lights') or []) or 'none'}")

    if FAILURES:
        print("\nFAILED %d check(s):" % len(FAILURES))
        for line in FAILURES:
            print("  -", line)
        return 1
    print("\nOK - the character layer builds, deforms, speaks and restyles.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
