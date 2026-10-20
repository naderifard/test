"""h_assemble, in Blender: compile one resolved shot into ``scene.blend``.

Nothing is decided here. ``shot.json`` was written by the host step and carries
every number this script needs -- frame range, staging, retime policy, offsets,
the frame each mouth curve starts on, the resolved camera. This half appends the
assets, keys what the shot says to key, applies the style pack, and then
*measures the result* against section 11-H rather than reporting that it ran.

Three things are worth knowing before reading the code.

**Assets are appended, not rebuilt.** ``libraries.load`` brings a character's
whole collection across with its armature modifier and vertex groups intact, so
the thing standing in the shot is byte-for-byte the thing c_rigged gated. A
shot that rebuilt its characters from parameters would be free to disagree with
the asset, and the disagreement would be invisible.

**A prop is parented to a bone, and its world matrix is assigned afterwards.**
Blender's bone parenting puts a child at the bone's *tail*; d_env builds every
prop with its origin at the grip, and c_rigged records the bone's rest *head*.
Setting ``matrix_world`` once the parent is in place makes Blender solve for the
basis, which is both shorter and less wrong than composing the parent inverse by
hand -- and the result is measured below, so a mace half a metre out of the fist
fails the stage instead of shipping.

**The camera tracks an empty.** A Track To constraint on a target at the aim
point means a move only has to key a location: the aim stays exact through a
push or an orbit, which hand-keyed rotations do not.

The measurements at the end deliberately sample more than one frame. Feet on the
ground and the subject in frustum are properties of a *shot*, not of frame one,
and a character who leaves the frame halfway through the line is exactly the
failure a single-frame check would pass.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from pipeline import jsonio                                      # noqa: E402
from pipeline.blender import _common                             # noqa: E402
from pipeline.stages import _base                                # noqa: E402
from pipeline.stages.h_assemble import ease                      # noqa: E402

import bpy                                                       # noqa: E402
import mathutils                                                 # noqa: E402
from bpy_extras.object_utils import world_to_camera_view         # noqa: E402

B = _common.install()

# Frames to measure the shot at. The ends catch a character walking out of
# frame; the middle catches the pose the shot is mostly in.
SAMPLE_FRACTIONS = (0.0, 0.5, 1.0)

# How far a prop may sit from the bone it is held by before it is not held.
PROP_TOLERANCE_M = 0.02

GROUND_TOLERANCE_M = 0.02

# Blender bones point along their own +Y; a prop is modelled up its own +Z.
PROP_ALIGN = mathutils.Matrix.Rotation(math.radians(-90.0), 4, "X")


def _abs(job, relative):
    return os.path.join(job.project_root, str(relative).replace("/", os.sep))


# Points sampled along a band's top edge. One is not enough: the bands are wide
# planes seen obliquely, so the highest point of the edge inside the frame is
# rarely a corner.
EDGE_SAMPLES = 13


def _top_edge(ob):
    """The two world-space corners along the top of a backdrop plane."""
    points = sorted(_common.world_points(ob), key=lambda p: p.z)
    return points[-2], points[-1]


def _edge_height(cam, ob):
    """How high this band reaches in the frame, in frame heights.

    Sampled along the top edge and filtered to points *in front of* the camera.
    Projecting the raw bounding box instead gives nonsense the moment a plane
    is wide and close: a corner behind the camera still returns an x and a y,
    and s06's near band reported 4.2 frame heights that way -- which read as a
    band towering over the shot when it is simply beside it.
    """
    scene = bpy.context.scene
    left, right = _top_edge(ob)
    best = None
    for index in range(EDGE_SAMPLES):
        point = left.lerp(right, index / float(EDGE_SAMPLES - 1))
        projected = world_to_camera_view(scene, cam, point)
        if projected.z <= 0.0:
            continue                      # behind the camera; it occludes nothing
        if not -0.25 <= projected.x <= 1.25:
            continue                      # off to the side of the frame
        best = projected.y if best is None else max(best, projected.y)
    return best


def _backdrop_visibility(cam, env_collection):
    """Where each backdrop band reaches in the frame, and whether the plate shows.

    The bands are parallel planes at increasing distance, so the painted plate
    is visible exactly where it rises above every band in front of it. If a land
    band reaches the top of the frame the plate does not appear at all -- and
    nothing in the render says so, because a flat coloured band is a perfectly
    plausible sky.
    """
    bands, plate_top, land_top = {}, None, 0.0
    for ob in (env_collection.all_objects if env_collection else []):
        if not ob.name.startswith("backdrop"):
            continue
        top = _edge_height(cam, ob)
        bands[ob.name] = None if top is None else round(top, 4)
        if top is None:
            continue
        painted = any(slot.material is not None
                      and slot.material.get("style_plate")
                      for slot in ob.material_slots)
        if painted:
            plate_top = round(top, 4)
        else:
            land_top = max(land_top, top)
    visible = (plate_top is not None and land_top < 1.0
               and plate_top > land_top)
    return {
        "backdrop_bands": bands,
        "backdrop_land_top": round(land_top, 4),
        "backdrop_plate_top": plate_top,
        "backdrop_plate_visible": bool(visible),
        # Frame height runs 0..1: the plate shows in the gap between the tallest
        # land band and the top edge.
        "backdrop_plate_frame_fraction": (
            round(max(0.0, min(1.0, plate_top) - min(1.0, land_top)), 4)
            if plate_top is not None else 0.0),
    }


def _plate_pixels():
    """The size of every reference plate that survived into this shot.

    An image whose pixels did not travel with the append still has a datablock,
    a name and a live link, and reports (0, 0) -- rendering as flat black, which
    is indistinguishable from a deliberately dark backdrop. So the pixels are
    counted here rather than the material.
    """
    found = {}
    for material in bpy.data.materials:
        plate = material.get("style_plate")
        if not plate:
            continue
        for node in (material.node_tree.nodes if material.use_nodes else []):
            if node.bl_idname == "ShaderNodeTexImage" and node.image:
                found[plate] = list(node.image.size)
    return found


# -- staging -----------------------------------------------------------------
def _add_env(job, shot):
    env = shot["env"]
    name = shot["provenance"].get("env_collection") or f"env_{env['id']}"
    collection = _common.append_collection(_abs(job, env["blend"]), name)
    ground = next((ob for ob in collection.all_objects
                   if ob.name.startswith("ground")), None)
    return collection, ground


def _add_character(job, member, provenance):
    """Append one character and stand them where the shot says."""
    collection = _common.append_collection(_abs(job, member["assets"]["body"]),
                                           provenance["collection"])
    rig = bpy.data.objects.get(provenance["rig"])
    mesh = bpy.data.objects.get(provenance["mesh"])
    if rig is None or mesh is None:
        raise RuntimeError(
            f"{member['character']}: {provenance['collection']} arrived without "
            f"{provenance['rig']}/{provenance['mesh']}")
    B.place(rig, location=member["position_m"],
            rotation=(0.0, 0.0, member["facing_deg"]))
    return {"collection": collection, "rig": rig, "mesh": mesh}


def _stand_on(rig, ground, meshes=()):
    """Drop a character onto the set under their mark, and say how far.

    An image set's ground is the scene's own terrain, not the plane z=0, so a
    mark on a slope or a rock would otherwise leave the feet in the air or in
    the stone. Sampling only under the rig's origin was not enough: the div's
    feet are half a metre apart and one of them landed on a rise, 0.5 m into
    it. So the terrain is sampled under the figure's whole footprint -- every
    vertex within a few centimetres of its lowest -- and the figure stands on
    the highest point it touches. Returns the change applied, or None.
    """
    if ground is None:
        return None
    bpy.context.view_layer.update()
    low_points = []
    for ob in meshes:
        points = _common.world_points(ob)
        if not points:
            continue
        floor = min(p.z for p in points)
        low_points.extend(p for p in points if p.z < floor + 0.04)
    if not low_points:
        low_points = [rig.matrix_world.translation.copy()]
    step = max(1, len(low_points) // 40)
    lift = None
    for point in low_points[::step]:
        under = _ground_under(ground, point)
        if under is not None:
            need = under - point.z
            lift = need if lift is None else max(lift, need)
    if lift is None:
        return None
    rig.location.z += lift
    return round(lift, 4)


CONTACT_BAND_M = 0.25        # the same band _terrain_contact measures
FLAT_SEARCH_M = 0.6
FLAT_STEP_M = 0.15
FOOT_RADIUS_M = 0.28
FLAT_DISTANCE_COST = 0.08    # metres of unevenness worth a metre of move


def _unevenness(ground, x, y, radius=FOOT_RADIUS_M):
    """Height range of the terrain over a footprint disc, metres."""
    heights = []
    for i in range(8):
        a = i * math.pi / 4.0
        for r in (0.0, radius * 0.5, radius):
            h = _ground_under(ground, mathutils.Vector(
                (x + r * math.cos(a), y + r * math.sin(a), 0.0)))
            if h is not None:
                heights.append(h)
    return (max(heights) - min(heights)) if len(heights) > 4 else None


def _flat_spot(rig, ground):
    """Nudge a character's mark to the flattest footing within reach.

    Marks come from words ("right, middle") solved into metres, blind to the
    picture the set was built from; on the mountain court that put Siamak on
    the lip of a ledge with a toe over the drop. The terrain around the mark
    is searched, and the mark moves (at most ``FLAT_SEARCH_M``) to where the
    footprint is most even, a little distance costing a little evenness so a
    good mark stays put. Returns the move in metres, or None.
    """
    x0, y0 = rig.location.x, rig.location.y
    here = _unevenness(ground, x0, y0)
    if here is None or here < 0.05:
        return None
    best = (here, x0, y0)
    steps = int(FLAT_SEARCH_M / FLAT_STEP_M)
    for i in range(-steps, steps + 1):
        for j in range(-steps, steps + 1):
            dx, dy = i * FLAT_STEP_M, j * FLAT_STEP_M
            d = math.hypot(dx, dy)
            if d > FLAT_SEARCH_M or d == 0:
                continue
            u = _unevenness(ground, x0 + dx, y0 + dy)
            if u is None:
                continue
            cost = u + FLAT_DISTANCE_COST * d
            if cost < best[0]:
                best = (cost, x0 + dx, y0 + dy)
    if best[1] == x0 and best[2] == y0:
        return None
    rig.location.x, rig.location.y = best[1], best[2]
    return round(math.hypot(best[1] - x0, best[2] - y0), 3)


def _contact_lift(ground, meshes):
    """How far the figure must rise (negative: drop) so that nothing in its
    lowest quarter metre is inside the terrain, at the current frame; None
    off the terrain. A band and not just the lowest points: on a ledge, a
    raised toe over the rock face sank 0.24 m into the stone while the
    planted heel stood exactly on it."""
    low_points = []
    for ob in meshes:
        points = _common.world_points(ob)
        if not points:
            continue
        floor = min(p.z for p in points)
        low_points.extend(p for p in points if p.z < floor + CONTACT_BAND_M)
    step = max(1, len(low_points) // 80)
    lift = None
    for point in low_points[::step]:
        under = _ground_under(ground, point)
        if under is not None:
            need = under - point.z
            lift = need if lift is None else max(lift, need)
    return lift


FOLLOW_STEP = 3
FOLLOW_SMOOTH = 2          # frames either side, in samples


def _follow_terrain(rig, ground, meshes, shot):
    """Keep the footprint on the terrain for the whole take, not frame one.

    Learned motion travels: a bow that takes a step, a fight that gives ground.
    On an image set the ground under the new spot is wherever the depth put
    it, so a character dropped once at the start floated 0.25 m over a dip by
    the end of the move. The contact lift is sampled every few frames with the
    take playing, smoothed so a footprint switching from one foot to the other
    on rough ground does not jitter, and keyed on the rig object's height.
    Returns the largest correction, metres.
    """
    scene = bpy.context.scene
    frames = list(range(shot["frame_start"], shot["frame_end"] + 1,
                        FOLLOW_STEP))
    if frames[-1] != shot["frame_end"]:
        frames.append(shot["frame_end"])
    base = rig.location.z
    lifts = []
    for frame in frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        lift = _contact_lift(ground, meshes)
        lifts.append(lifts[-1] if lift is None and lifts else (lift or 0.0))
    smooth = []
    for i in range(len(lifts)):
        window = lifts[max(0, i - FOLLOW_SMOOTH):i + FOLLOW_SMOOTH + 1]
        # Never below the contact itself: averaging across a ledge's edge
        # sank Siamak 0.24 m into the rock. A little float at a step is the
        # lesser error; a foot inside stone is the one everyone sees.
        smooth.append(max(sum(window) / len(window), lifts[i]))
    for frame, lift in zip(frames, smooth):
        rig.location.z = base + lift
        rig.keyframe_insert("location", index=2, frame=frame)
    scene.frame_set(shot["frame_start"])
    return round(max((abs(v) for v in smooth), default=0.0), 4)


def _rotate(poses, offset):
    """A phase into a cycle: the take starts ``offset`` frames in.

    Only meaningful for a looping take, which is why the host only derives an
    offset for one. Two characters sharing ``idle_human`` and breathing in
    unison is the thing this exists to prevent.
    """
    if not poses or not offset:
        return poses
    index = int(offset) % len(poses)
    return poses[index:] + poses[:index]


LEAN_BONES = ("spine1", "spine2", "spine3")


def _lean(poses, degrees):
    """Every pose bent forward ``degrees`` at the spine, spread over its three
    bones -- as a bow bends (negative pitch)."""
    if not degrees:
        return poses
    from pipeline import motion_library as ml
    from pipeline import motion_learned
    bend = ml.q_axis((1.0, 0.0, 0.0), -float(degrees) / len(LEAN_BONES))
    out = []
    for pose in poses:
        bent = dict(pose)
        for name in LEAN_BONES:
            q = motion_learned._quat(pose.get(name, (0.0, 0.0, 0.0)))
            bent[name] = ml.q_rotvec(ml.q_mul(q, bend))
        out.append(bent)
    return out


def _apply_take(job, built, member, shot, provenance):
    """Body and head, fitted to the shot's window by the shot's own policy."""
    body = jsonio.read(_abs(job, member["assets"]["motion"]), default={})
    # A giant fighting a man bends to him: the shared fight loop is a man's,
    # and on the div it punched the air above Siamak's head.
    body["poses"] = _lean(body.get("poses") or [],
                          member.get("lean_deg") or 0.0)
    head = jsonio.read(_abs(job, member["assets"]["head_pose"]), default={})
    if member["assets"].get("visemes"):
        from pipeline import motion_learned
        head["poses"] = motion_learned.cap_head_up(head.get("poses") or [])
    frames = shot["frame_count"]
    retime = member["retime"]
    offset = int(member.get("offset_frames") or 0)
    scale = float(provenance.get("root_scale") or 1.0)

    if retime == "loop":
        poses, root = _rotate(body.get("poses") or [], offset), body.get("root_m")
        root = _rotate(list(root), offset) if root else None
        start = shot["frame_start"]
        window = frames
    else:
        # Not a cycle: the offset slides the performance against the line, so
        # it moves the first frame rather than the phase.
        poses, root = body.get("poses") or [], body.get("root_m")
        start = shot["frame_start"] + offset
        window = max(1, frames - offset)

    stats = B.apply_motion(built["rig"], poses, root=root, start=start,
                           frames=window, retime=retime,
                           axes=body.get("axes") or "blender",
                           root_scale=scale)

    # The head runs on its own cycle (e_motion keeps them separate because a
    # talking head follows the line while the body loops), so it is always
    # looped over the whole window regardless of the body's policy.
    head_stats = B.apply_motion(
        built["rig"],
        _rotate(head.get("poses") or [],
                offset if head.get("loopable", True) else 0),
        start=shot["frame_start"], frames=frames,
        # An action's head (a look, a nod, a bowed grief) plays once and holds
        # like its body; only a cyclic take's head loops.
        retime=head.get("retime") or "loop",
        axes=head.get("axes") or "blender")
    return stats, head_stats


def _apply_visemes(job, built, member, shot, pack):
    """Key each of this character's lines at the frame the host worked out."""
    starts = (shot["provenance"]["cast"][member["character"]]
              .get("viseme_start_frame") or {})
    out = {}
    for relative in member["assets"]["visemes"]:
        curves = jsonio.read(_abs(job, relative), default={})
        line_id = curves.get("line_id") or os.path.basename(
            os.path.dirname(relative))
        stats = B.apply_visemes(built["mesh"], curves.get("channels") or {},
                                start=int(starts.get(line_id,
                                                     shot["frame_start"])),
                                profile=pack.visemes)
        out[line_id] = stats
    return out


def _attach_props(job, built, member, shot, provenance):
    """Put each carried prop in the hand that carries it, and measure the fit."""
    attach = provenance.get("attach") or {}
    records = {}
    for prop_id in provenance.get("props") or []:
        entry = (shot["provenance"]["props"] or {}).get(prop_id)
        if not entry:
            continue
        manifest = jsonio.read(_abs(job, entry["manifest"]), default={})
        collection = _common.append_collection(
            _abs(job, entry["blend"]),
            manifest.get("collection") or f"prop_{prop_id}")
        prop = bpy.data.objects.get(manifest.get("object") or prop_id)
        if prop is None:
            records[prop_id] = {"attached": False,
                                "note": "the prop collection has no mesh"}
            continue
        slot = attach.get(manifest.get("attach") or "hand_r")
        if not slot:
            records[prop_id] = {"attached": False,
                                "note": f"the rig has no {manifest.get('attach')}"}
            continue

        rig = built["rig"]
        bone = rig.pose.bones.get(slot["bone"])
        if bone is None:
            records[prop_id] = {"attached": False,
                                "note": f"no bone {slot['bone']}"}
            continue
        bpy.context.view_layer.update()
        prop.parent = rig
        prop.parent_type = "BONE"
        prop.parent_bone = slot["bone"]
        prop.matrix_world = (rig.matrix_world @ bone.matrix) @ PROP_ALIGN

        bpy.context.view_layer.update()
        head = (rig.matrix_world @ bone.matrix).translation
        gap = (prop.matrix_world.translation - head).length
        records[prop_id] = {
            "attached": True,
            "bone": slot["bone"],
            "grip_gap_m": round(gap, 5),
            "collection": collection.name,
        }
    return records


# -- camera ------------------------------------------------------------------
# -- framing and contact shadows --------------------------------------------
FRAME_MARGIN = 0.03          # of the frame, kept clear at the top and sides
FIT_SAMPLES = 6
MIN_FIT_FOCAL_MM = 18.0


def _screen_box(cam, meshes, stride=25):
    """The meshes' bounding box in normalised camera coordinates (y up)."""
    from bpy_extras.object_utils import world_to_camera_view
    scene = bpy.context.scene
    xs, ys = [], []
    for ob in meshes:
        points = _common.world_points(ob)
        for p in points[::stride]:
            view = world_to_camera_view(scene, cam, p)
            if view.z > 0:
                xs.append(view.x)
                ys.append(view.y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _fit_frame(cam, subject_meshes, shot, others=()):
    """Zoom out until the subject's head and sides stay inside the frame.

    The plan frames by the subject's height; a figure with wings, a spear or a
    raised arm is wider or taller than that, and a clamped camera on an image
    set sits closer than planned. So the subject's screen box is measured at a
    few frames across the shot (the move included) and, if its top or either
    side would leave the frame, the lens widens by what the worst frame needs.
    The bottom may leave: a medium shot is meant to cut at the waist.
    Returns ``(old_focal, new_focal)`` or None.
    """
    if not subject_meshes:
        return None
    scene = bpy.context.scene
    first, last = shot["frame_start"], shot["frame_end"]
    frames = sorted({first + int(round(i * (last - first)
                                        / float(FIT_SAMPLES - 1)))
                     for i in range(FIT_SAMPLES)})
    need = 1.0
    for frame in frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        box = _screen_box(cam, subject_meshes)
        if box is None:
            continue
        x0, _y0, x1, y1 = box
        half = 0.5 - FRAME_MARGIN
        need = max(need, (0.5 - x0) / half, (x1 - 0.5) / half,
                   (y1 - 0.5) / half)
        # Anyone else substantially in frame keeps their head: a foreground
        # figure may be cut at the side (an over-the-shoulder), not at the
        # neck. Soroush, downstage of Keyumars, lost his head to the top edge.
        for meshes in others:
            other = _screen_box(cam, meshes)
            if other is None:
                continue
            ox0, oy0, ox1, oy1 = other
            seen_w = max(0.0, min(ox1, 1.0) - max(ox0, 0.0))
            if seen_w < 0.2 * max(1e-6, ox1 - ox0) or oy0 > 1.0:
                continue
            need = max(need, (oy1 - 0.5) / half)
    scene.frame_set(first)
    if need <= 1.0:
        return None
    old = cam.data.lens
    cam.data.lens = max(MIN_FIT_FOCAL_MM, old / need)
    return round(old, 2), round(cam.data.lens, 2)


SHADOW_STRENGTH = 0.95


def _contact_shadow(rig, meshes, ground, cid):
    """A soft dark footprint under a character, following them over the set.

    An image set's ground is mostly emission (it is a photograph, already
    lit), so the sun's shadow on it is faint and the cast looks pasted on. A
    soft falloff decal under each figure is the oldest fix there is: it rides
    the pelvis across the set (a Copy Location constraint, x and y) and is
    shrink-wrapped onto the terrain so it lies on the rock rather than
    cutting through it. It is geometry nobody sees as geometry: a transparent
    plane whose only visible part is the darkening.
    """
    import bmesh
    lows = [p for ob in meshes for p in _common.world_points(ob)[::20]]
    if not lows:
        return None
    xs = [p.x for p in lows]
    ys = [p.y for p in lows]
    radius_x = max(0.22, 0.5 * (max(xs) - min(xs)) * 0.85)
    radius_y = max(0.22, 0.5 * (max(ys) - min(ys)) * 0.85)
    mesh = bpy.data.meshes.new(f"{cid}_contact_shadow")
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=16, y_segments=16, size=1.0)
    bm.to_mesh(mesh)
    bm.free()
    ob = bpy.data.objects.new(f"{cid}_contact_shadow", mesh)
    bpy.context.scene.collection.objects.link(ob)
    ob.scale = (radius_x * 1.6, radius_y * 1.6, 1.0)
    ob.location = (rig.location.x, rig.location.y, rig.location.z + 0.3)
    copy = ob.constraints.new("COPY_LOCATION")
    copy.target = rig
    if "pelvis" in rig.pose.bones:
        copy.subtarget = "pelvis"
    copy.use_z = False
    wrap = ob.modifiers.new("onto_ground", "SHRINKWRAP")
    wrap.target = ground
    wrap.wrap_method = "PROJECT"
    wrap.use_project_z = True
    wrap.use_negative_direction = True
    wrap.use_positive_direction = True
    wrap.offset = 0.012

    mat = bpy.data.materials.new(f"{cid}_contact_shadow")
    mat.use_nodes = True
    # Dithered, not blended: the style packs' compositor rebuilds the frame
    # from render passes, and a blended (forward) surface is not in them --
    # partly transparent shadow simply vanished. Dithered transparency goes
    # through the deferred pipeline and resolves smooth over the TAA samples.
    try:
        mat.surface_render_method = "DITHERED"
    except (AttributeError, TypeError):
        mat.blend_method = "HASHED"
    tree = mat.node_tree
    for node in list(tree.nodes):
        if node.bl_idname != "ShaderNodeOutputMaterial":
            tree.nodes.remove(node)
    out = next(n for n in tree.nodes
               if n.bl_idname == "ShaderNodeOutputMaterial")
    coords = tree.nodes.new("ShaderNodeTexCoord")
    # Across the ground only: the shrinkwrap moves the vertices down to the
    # terrain, so the object-space z is the drop to the ground, and a 3D
    # length read every point as the rim and the shadow as nothing.
    flat = tree.nodes.new("ShaderNodeVectorMath")
    flat.operation = "MULTIPLY"
    flat.inputs[1].default_value = (1.0, 1.0, 0.0)
    tree.links.new(coords.outputs["Object"], flat.inputs[0])
    length = tree.nodes.new("ShaderNodeVectorMath")
    length.operation = "LENGTH"
    tree.links.new(flat.outputs["Vector"], length.inputs[0])
    inv = tree.nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    inv.use_clamp = True
    tree.links.new(length.outputs["Value"], inv.inputs[1])
    soft = tree.nodes.new("ShaderNodeMath")
    soft.operation = "POWER"
    soft.inputs[1].default_value = 1.3
    tree.links.new(inv.outputs[0], soft.inputs[0])
    alpha = tree.nodes.new("ShaderNodeMath")
    alpha.operation = "MULTIPLY"
    alpha.inputs[1].default_value = SHADOW_STRENGTH
    tree.links.new(soft.outputs[0], alpha.inputs[0])
    dark = tree.nodes.new("ShaderNodeEmission")
    dark.inputs["Color"].default_value = (0.02, 0.015, 0.01, 1.0)
    dark.inputs["Strength"].default_value = 1.0
    clear = tree.nodes.new("ShaderNodeBsdfTransparent")
    mix = tree.nodes.new("ShaderNodeMixShader")
    tree.links.new(alpha.outputs[0], mix.inputs["Fac"])
    tree.links.new(clear.outputs["BSDF"], mix.inputs[1])
    tree.links.new(dark.outputs["Emission"], mix.inputs[2])
    tree.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    mat["style_plate"] = "contact_shadow"    # the style pack leaves it alone
    mat["style_role"] = "shadow"
    ob.data.materials.append(mat)
    ob.visible_shadow = False
    return ob.name


def _build_camera(shot, pack):
    """Camera, aim target and move. The constraint keeps the aim exact."""
    spec = shot["camera"]
    aim = mathutils.Vector(spec["aim_point_m"])
    target = bpy.data.objects.new("aim_target", None)
    target.empty_display_size = 0.15
    target.location = aim
    bpy.context.scene.collection.objects.link(target)

    cam = B.camera(location=spec["position_m"], target=spec["aim_point_m"],
                   focal=spec["focal_mm"], name="shot_camera")
    cam.data.sensor_width = float(pack.camera.get("sensor_mm") or 36.0)
    cam.data.shift_y = float(pack.camera.get("shift_y") or 0.0)
    if spec.get("dof"):
        cam.data.dof.use_dof = True
        cam.data.dof.focus_object = target
        cam.data.dof.aperture_fstop = float(pack.camera.get("fstop") or 2.8)
    else:
        cam.data.dof.use_dof = False

    # Rotation is now the constraint's business, so the move keys locations only
    # and the aim cannot drift off the subject in the middle of a push.
    for existing in list(cam.constraints):
        cam.constraints.remove(existing)
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    cam.rotation_euler = (0.0, 0.0, 0.0)

    move = _key_move(cam, spec.get("move"), aim)
    return cam, target, move


def _key_move(cam, move, aim):
    """Key the camera's path. Returns what was actually keyed."""
    if not move:
        return {"kind": "static", "keys": 0}
    first, last = int(move["from_frame"]), int(move["to_frame"])
    start = mathutils.Vector(move.get("from_position_m") or cam.location)

    if move["kind"] == "orbit":
        angle = math.radians(float(move.get("orbit_deg") or 0.0))
        keys = max(2, min(48, last - first + 1))
        for index in range(keys):
            fraction = index / float(keys - 1)
            frame = first + int(round(fraction * (last - first)))
            turn = mathutils.Matrix.Rotation(angle * ease(fraction,
                                                           move.get("ease")),
                                             4, "Z")
            B.key(cam, frame, location=aim + turn @ (start - aim))
        return {"kind": "orbit", "keys": keys, "from_frame": first,
                "to_frame": last, "orbit_deg": move.get("orbit_deg")}

    end = move.get("to_position_m")
    if not end:
        return {"kind": move["kind"], "keys": 0,
                "note": "no destination; the camera holds"}
    end = mathutils.Vector(end)
    keys = max(2, min(48, last - first + 1))
    for index in range(keys):
        fraction = index / float(keys - 1)
        frame = first + int(round(fraction * (last - first)))
        B.key(cam, frame,
              location=start.lerp(end, ease(fraction, move.get("ease"))))
    return {"kind": move["kind"], "keys": keys, "from_frame": first,
            "to_frame": last}


# -- measurement -------------------------------------------------------------
def _finite(matrix):
    return all(math.isfinite(value) for row in matrix for value in row)


def _sample_frames(shot):
    span = shot["frame_end"] - shot["frame_start"]
    return sorted({shot["frame_start"] + int(round(f * span))
                   for f in SAMPLE_FRACTIONS})


# On an image set the ground is terrain the cast walks over, and a moving
# character's root does not follow every bump in it; this is how far a foot may
# sit off that terrain before it reads as floating or sinking.
IMAGE_SET_GROUND_TOLERANCE_M = 0.12


def _ground_under(ground, point):
    """The terrain height below a world point, or None."""
    graph = bpy.context.evaluated_depsgraph_get()
    surface = ground.evaluated_get(graph)
    inverse = ground.matrix_world.inverted()
    origin = inverse @ mathutils.Vector((point.x, point.y, point.z + 50.0))
    direction = inverse.to_3x3() @ mathutils.Vector((0.0, 0.0, -1.0))
    hit, location, _normal, _index = surface.ray_cast(origin, direction)
    return (ground.matrix_world @ location).z if hit else None


def _terrain_contact(meshes, ground, band=0.25, samples=60):
    """How far the figure is from touching the terrain, signed.

    The smallest gap between any low vertex and the terrain directly under it:
    zero when something rests on the ground, positive when nothing does
    (floating), negative when a vertex is inside it (sinking). The single
    lowest vertex is the wrong thing to measure on rock: Keyumars stood on a
    ledge with one foot over a dip, and the dip's 0.24 m read as floating.
    """
    low_points = []
    for ob in meshes:
        points = _common.world_points(ob)
        if points:
            floor = min(p.z for p in points)
            low_points.extend(p for p in points if p.z < floor + band)
    if not low_points:
        return 0.0
    step = max(1, len(low_points) // samples)
    gaps = []
    for point in low_points[::step]:
        under = _ground_under(ground, point)
        if under is not None:
            gaps.append(point.z - under)
    if not gaps:
        return 0.0
    sinking = min(gaps)
    return sinking if sinking < 0 else min(abs(g) for g in gaps)


def _measure(shot, cam, cast_objects, all_meshes, ground=None):
    """Section 11-H, over the shot rather than over one frame.

    Feet are measured against z=0 on a flat set, and against the terrain under
    the character's lowest point on an image set, where the ground is wherever
    the scene's depth put it -- a throne ledge 0.86 m up, a hollow 0.39 m down.
    """
    off_ground, in_frustum, coverage, nan = {}, False, 0.0, []
    terrain = shot["env"].get("strategy") == "image_set" and ground is not None
    tolerance = IMAGE_SET_GROUND_TOLERANCE_M if terrain else GROUND_TOLERANCE_M
    for frame in _sample_frames(shot):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for name, meshes in cast_objects.items():
            if terrain:
                worst = _terrain_contact(meshes, ground)
            else:
                gaps = _common.feet_gap(meshes)
                worst = min(gaps.values()) if gaps else 0.0
            if abs(worst) > tolerance \
                    and abs(worst) > abs(off_ground.get(name, 0.0)):
                off_ground[name] = round(worst, 4)
        framed = _common.framing(cam, [ob for meshes in cast_objects.values()
                                       for ob in meshes])
        in_frustum = in_frustum or framed["subjects_in_frustum"]
        coverage = max(coverage, framed["subject_frame_fraction"])
        for ob in bpy.context.scene.objects:
            if not _finite(ob.matrix_world) and ob.name not in nan:
                nan.append(ob.name)
    return {
        "characters_off_ground": off_ground,
        "subjects_in_frustum": in_frustum,
        "subject_frame_fraction": round(coverage, 5),
        "nan_transforms": nan,
        "objects_without_material": _common.materials_missing(all_meshes),
        "sampled_frames": _sample_frames(shot),
    }


# -- the step ----------------------------------------------------------------
def build(job, unit):
    shot = jsonio.read(_base.out_path(unit, "shot.json"), default=None)
    if shot is None:
        raise RuntimeError("the host step wrote no shot.json to build from")
    pack = job.style()
    fps = shot["fps"]

    # One process builds every shot (§10 rule 1), and this stage appends rather
    # than builds, so each unit needs an empty file and not merely an empty
    # scene - see _common.new_file.
    _common.new_file()
    B.reset(engine="EEVEE", res=shot["resolution"], fps=fps)
    B.set_frames(shot["frame_start"], shot["frame_end"], fps)

    env_collection, ground = _add_env(job, shot)

    cast_objects, motion, head_motion, visemes, props = {}, {}, {}, {}, {}
    dropped, nudged = {}, {}
    for member in shot["cast"]:
        cid = member["character"]
        provenance = shot["provenance"]["cast"][cid]
        built = _add_character(job, member, provenance)
        if shot["env"].get("strategy") == "image_set" and ground is not None:
            nudged[cid] = _flat_spot(built["rig"], ground)
        if shot["env"].get("strategy") == "image_set":
            dropped[cid] = _stand_on(
                built["rig"], ground,
                [ob for ob in built["collection"].all_objects
                 if ob.type == "MESH"])
        stats, head_stats = _apply_take(job, built, member, shot, provenance)
        if shot["env"].get("strategy") == "image_set" and ground is not None:
            stats["terrain_follow_m"] = _follow_terrain(
                built["rig"], ground,
                [ob for ob in built["collection"].all_objects
                 if ob.type == "MESH"], shot)
        motion[cid] = stats
        head_motion[cid] = head_stats
        visemes.update(_apply_visemes(job, built, member, shot, pack))
        props.update(_attach_props(job, built, member, shot, provenance))
        cast_objects[cid] = [ob for ob in built["collection"].all_objects
                             if ob.type == "MESH"]

    cam, aim_target, move = _build_camera(shot, pack)

    shadows = {}
    if shot["env"].get("strategy") == "image_set" and ground is not None:
        for member in shot["cast"]:
            cid = member["character"]
            rig = bpy.data.objects.get(
                shot["provenance"]["cast"][cid].get("rig") or "")
            if rig is not None:
                shadows[cid] = _contact_shadow(rig, cast_objects[cid], ground,
                                               cid)
    subject = (shot["camera"].get("aim_target") or "").split(".")[0]
    fitted = _fit_frame(cam, cast_objects.get(subject)
                        or [ob for obs in cast_objects.values() for ob in obs],
                        shot, [obs for cid, obs in cast_objects.items()
                               if cid != subject])

    # The style pack last, so it swaps the roles on everything that arrived from
    # every asset file, and lights around the point the camera is looking at.
    # postfx included: §5.1 applies the style pack to the shot .blend (stage I),
    # so the compositor travels with the file and j_render only has to render
    # it. The packs' grain is a static noise plate rather than a per-frame one,
    # so it costs the §11 temporal-flicker metric nothing.
    applied = B.style(pack, repo_root=job.repo_root,
                      target=shot["camera"]["aim_point_m"],
                      distance_scale=1.0, materials=True, lighting=True,
                      render=True, postfx=True)

    all_meshes = [ob for ob in bpy.context.scene.objects if ob.type == "MESH"]
    measured = _measure(shot, cam, cast_objects, all_meshes, ground)
    stats = _common.measure(all_meshes, uv=False)

    scene = bpy.context.scene
    scene.frame_set(shot["frame_start"])
    scene["bstudio_shot"] = shot["shot_id"]
    scene["bstudio_style_pack"] = pack.name
    scene["bstudio_frames"] = "%d-%d" % (scene.frame_start, scene.frame_end)
    blend = _common.save_blend(_base.out_path(unit, "scene.blend"))

    return {
        "metrics": {
            "shot_id": shot["shot_id"],
            "style_pack": pack.name,
            "engine": applied.get("engine"),
            "world": applied.get("world"),
            "lights": applied.get("lights"),
            "materials_swapped": sum(
                (applied.get("materials") or {}).values()),
            "postfx": applied.get("postfx"),
            # Reference-image plates the pack left alone. Zero of these on a
            # painted_backdrop set means the backdrop is flat colour.
            "materials_kept": sum((applied.get("materials_kept") or {}).values()),
            "built_frame_range": [scene.frame_start, scene.frame_end],
            "objects": len(bpy.context.scene.objects),
            "meshes": len(all_meshes),
            "env_objects": len(env_collection.all_objects),
            "env_strategy": shot["env"].get("strategy"),
            "env_plate": shot["env"].get("backdrop"),
            "env_plate_px": _plate_pixels(),
            **_backdrop_visibility(cam, env_collection),
            "motion_frames": {cid: s.get("frames") for cid, s in motion.items()},
            "motion_bones": {cid: s.get("bones") for cid, s in motion.items()},
            "motion_retime": {cid: s.get("retime") for cid, s in motion.items()},
            "motion_missing_bones": {cid: s.get("missing_bones") or []
                                     for cid, s in motion.items()},
            "motion_detail": {cid: {k: s.get(k) for k in
                                    ("cycles", "speed", "trimmed_frames",
                                     "held_frames", "keyed_span")}
                              for cid, s in motion.items()},
            "head_bones": {cid: s.get("bones")
                           for cid, s in head_motion.items()},
            "viseme_keys": {line: s.get("keys") for line, s in visemes.items()},
            "viseme_peaks": {line: s.get("peaks")
                             for line, s in visemes.items()},
            "viseme_spans": {line: s.get("span")
                             for line, s in visemes.items()},
            "props_attached": {pid: r.get("attached")
                               for pid, r in props.items()},
            "props": props,
            "camera_keys": move,
            # How far each character was raised or lowered onto an image set's
            # terrain under their mark (None: nothing under them).
            "ground_drop_m": dropped,
            "framing_fit_focal_mm": fitted,
            "mark_nudged_m": nudged,
            "contact_shadows": sorted(shadows),
            "aim_target": aim_target.name,
            "ground": ground.name if ground else None,
            "blend": os.path.basename(blend),
            "tris": stats["tris"],
            **measured,
        },
        "notes": [
            "%s: %d objects, %d tris, %s at %d-%d"
            % (shot["shot_id"], len(bpy.context.scene.objects), stats["tris"],
               applied.get("engine"), scene.frame_start, scene.frame_end)]
        + ["%s: %s, %d bones, %d frames"
           % (cid, s.get("retime"), s.get("bones") or 0, s.get("frames") or 0)
           for cid, s in sorted(motion.items())]
        + ["%s: %d mouth keys over frames %s"
           % (line, s.get("keys") or 0, s.get("span"))
           for line, s in sorted(visemes.items())]
        + ["%s in %s, %.0f mm from the bone" % (pid, r.get("bone"),
                                                (r.get("grip_gap_m") or 0) * 1000)
           for pid, r in sorted(props.items()) if r.get("attached")],
    }


if __name__ == "__main__":
    sys.exit(_base.stage_main(build, from_blender=True))
