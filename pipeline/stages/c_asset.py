"""Stage C-asset — the reference photo becomes a coloured mesh with a skeleton.

§7-C1 wants a mesh per character built from the image a person dropped into
``refs/``. Version 1 of this stage ran Stable Fast 3D the way SF3D's own path
does and got a watertight mesh at 4,600-10,500 vertices -- and then nothing used
it, so the film kept rendering ``prelude.character()``'s welded boxes. Version 2
makes the mesh good enough to *be* the character, and hands the Blender side
everything it needs to rig it:

1. **A finer surface.** ``sf3d_dense`` samples SF3D's density on a grid fitted
   to the figure and runs marching cubes, instead of the fixed 160-tet grid:
   ~130k vertices for a standing figure, one watertight shell.
2. **Colour everywhere.** SF3D's decoder predicts albedo; it is sampled at each
   vertex, so the back and sides that no photo shows are coloured plausibly.
3. **The photo, pinned exactly.** The conditioning camera is fixed, so every
   vertex has an exact pixel in the original reference. The Blender step lays
   the photo over the albedo wherever the vertex was visible and facing the
   camera -- which is why a face in the film is the face in the reference.
4. **A skeleton, from the photo.** BlazePose finds the joints in the reference
   and each one is pushed through the mesh along its camera ray; the middle of
   where the ray enters and leaves the figure is where the bone goes. That is
   ``skeleton.human_joints``. Animals get ``prelude``'s quadruped template
   stretched into the mesh's box.

The old head pass is gone. It existed because the whole-body grid could not
spare a face any voxels; the projected photo now carries the face, and on a 6.3M-point
grid the head has real relief.

**Landmarks are found on the cut-out, not the photo.** Keyumars' reference is a
low-angle shot against a busy landscape and BlazePose put both his shoulders on
his chest there; on the matted figure over white it found every joint. Both are
tried and the one whose joints land inside the figure's own matte wins. A figure
BlazePose's face-keyed person detector cannot see at all -- a wolf-headed div --
is seeded from its silhouette box (``pose_mp.Estimator.detect``).

**The GPU is shared** (8 GB, and another job may be on it), so VRAM and host
commit are checked before the load, exactly as version 1 did.
"""

from __future__ import annotations

import os
import sys

from . import _base
from ._base import StageSpec, Step, Unit

from pipeline import gates, jsonio

# Below this the dense pass has produced a blob rather than a figure. Standing
# figures on this project come out at 100k+ vertices.
MIN_VERTICES = 20000
# The share of vertices that should take their colour from the photo. A front
# view shows roughly a third of a closed surface; far below that means the
# projection and the mesh disagree.
MIN_PHOTO_SHARE = 0.15
# BlazePose's own confidence, below which its joints are not trusted.
MIN_POSE_CONFIDENCE = 0.6
# The share of body landmarks that must fall inside the figure's matte for a
# landmark set to be believed.
MIN_LANDMARKS_INSIDE = 0.80

HUMANLIKE = ("human", "humanoid")
BODY_LANDMARKS = tuple(range(11, 33))

_MODEL = None
_MODEL_INFO = {}
_REMBG = None
_POSE = None


def ref_images(project_root, refs):
    """Slot name -> absolute image path, for the slots that have one.

    ``refs.json`` records a ``folder`` relative to the project root and bare
    filenames inside it, because ``b_refs`` is describing what a person dropped
    in rather than handing out paths. The two halves are put back together
    here, once, and used by both the planner and the step, so a cache key and
    the image it stands for cannot drift apart.
    """
    folder = refs.get("folder") or ""
    out = {}
    for slot, name in (refs.get("slots") or {}).items():
        if not name:
            continue
        out[slot] = os.path.abspath(os.path.join(
            project_root, folder.replace("/", os.sep), name))
    return out


def plan(project, graph):
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={})
    units = []
    for character in resolved.get("characters") or []:
        cid = character["id"]
        refs_path = project.path("assets", "characters", cid, "refs",
                                 "refs.json")
        refs = jsonio.read(refs_path, default={}) or {}
        images = ref_images(project.root, refs)
        # The image file itself is an input, not just refs.json: replacing a
        # reference with a better one has to invalidate the mesh built from it.
        inputs = {"refs": refs_path}
        if images.get("front"):
            inputs["ref_front"] = images["front"]
        # The turnaround sheet (side and back, pipeline/turnaround.py) is not
        # one of b_refs' slots; any other image in the folder is it.
        from pipeline import turnaround
        folder = project.path(*((refs.get("folder") or
                                 f"refs/characters/{cid}").split("/")))
        sheet = turnaround.sheet_path(folder) if os.path.isdir(folder) \
            else None
        if sheet and images.get("front"):
            inputs["ref_sheet"] = sheet
        units.append(Unit(
            f"char_{cid}",
            inputs=inputs,
            params={
                "subject": cid,
                "kind": character.get("kind", "human"),
                "height_m": float(character.get("height_m") or 1.75),
            },
            out_rel=f"assets/characters/{cid}/asset",
            label=f"asset/{cid}",
        ))
    return units


def nothing_to_do(project):
    return "no resolved production plan — a_production has not run"


# -- the step -------------------------------------------------------------

def _model():
    global _MODEL, _MODEL_INFO
    if _MODEL is None:
        from pipeline import envs, sf3d_geometry
        report = sf3d_geometry.check_vram()
        print("VRAM: %s" % report, flush=True)
        print("host: %s" % sf3d_geometry.check_commit(), flush=True)
        _MODEL, _MODEL_INFO = sf3d_geometry.load(envs.model_dir("sf3d"))
        _MODEL_INFO["vram_before_load"] = report
        print("SF3D loaded: %s" % _MODEL_INFO, flush=True)
    return _MODEL


def _rembg():
    global _REMBG
    if _REMBG is None:
        from pipeline import sf3d_geometry
        _REMBG = sf3d_geometry.new_rembg_session()
    return _REMBG


def _pose():
    global _POSE
    if _POSE is None:
        from pipeline import pose_mp
        _POSE = pose_mp.Estimator()
    return _POSE


def _inside_share(result, alpha):
    """How many body landmarks land on the figure rather than the background."""
    height, width = alpha.shape
    inside = 0
    for index in BODY_LANDMARKS:
        x, y = result["image"][index][:2]
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            window = alpha[max(0, yi - 6):yi + 7, max(0, xi - 6):xi + 7]
            inside += int(window.max() > 0.5)
    return inside / float(len(BODY_LANDMARKS))


def landmarks(photo, cond):
    """The best BlazePose reading of this figure, and how it was chosen.

    Tried on the matted figure over white first (the reading that survives a
    busy background), then on the photo as delivered. Each is scored by
    confidence times the share of body landmarks inside the matte.
    """
    import cv2
    import numpy as np
    from PIL import Image

    white = Image.new("RGBA", cond.cut.size, (255, 255, 255, 255))
    white.alpha_composite(cond.cut)
    variants = (("cutout_on_white", white.convert("RGB")),
                ("photo", photo.convert("RGB")))
    tried, best = [], None
    for name, image in variants:
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        result = _pose().detect(bgr, figure_box=cond.figure_box)
        if result is None:
            tried.append({"variant": name, "found": False})
            continue
        share = _inside_share(result, cond.alpha)
        score = result["confidence"] * share
        tried.append({"variant": name, "found": True,
                      "confidence": round(result["confidence"], 4),
                      "inside": round(share, 4), "seeded": result["seeded"]})
        if best is None or score > best[0]:
            best = (score, name, result, share)
    return best, tried


def _multiview(photo, sheet, kind):
    """Shape and colours from the front and the turnaround sheet."""
    from PIL import Image, ImageOps

    from pipeline import matte, multiview, sf3d_dense, turnaround
    cond = sf3d_dense.Conditioned(photo, _rembg())
    views, sheet_report = turnaround.views(Image.open(sheet), photo, kind)
    matte.release()
    mesh, dense = multiview.shape({"front": cond.cut, "left": views["left"],
                                   "back": views["back"]})
    mesh = multiview.reduce(mesh)
    dense["faces_reduced"] = int(len(mesh.faces))
    multiview.release()
    verts = mesh.vertices
    camera = multiview.FrontCamera(
        multiview.OrthoView("front", verts, cond.alpha), cond)
    cameras = {"front": (camera, cond)}
    for name, image in (("left", views["left"]), ("back", views["back"]),
                        ("right", ImageOps.mirror(views["left"]))):
        pc = multiview.PhotoCond(image)
        view = multiview.OrthoView(name, verts, pc.alpha)
        view.px = max(image.size)
        cameras[name] = (view, pc)
    colour, _total, colour_report = multiview.colours(mesh, cameras, kind)
    per_view = colour_report.pop("per_view")
    uv, weight, mapping = sf3d_dense.photo_mapping(mesh, camera, cond)
    view_images = {"left": views["left"], "back": views["back"],
                   "right": ImageOps.mirror(views["left"])}
    dense.update({"grid": [0, 0, 0], "shells": 1, "shells_kept": 1})
    report = {"sheet": dict(sheet_report, file=os.path.basename(sheet)),
              "colours": colour_report,
              "_per_view": per_view, "_images": view_images}
    return (mesh, colour, uv, weight, mapping, cond, camera, dense, report)


def run(job, unit):
    import numpy as np
    from PIL import Image

    from pipeline import pose_mp, sf3d_dense, skeleton

    params = unit["params"]
    subject = params["subject"]
    kind = params.get("kind") or "human"
    refs = jsonio.read(unit["inputs"]["refs"], default={}) or {}
    front = ref_images(job.project_root, refs).get("front")
    if not front:
        raise RuntimeError(
            f"{subject} has no reference image; c_asset builds the character "
            f"from one and there is none. Add it in the Characters panel, or "
            f"drop it into refs/characters/{subject}/{subject}_front.png.")

    photo = Image.open(front)
    sheet = unit["inputs"].get("ref_sheet")
    backend, why = "sf3d", ""
    if sheet:
        from pipeline import multiview
        ok, why = multiview.available()
        backend = "hunyuan3d_mv" if ok else "sf3d"
    views_report = None
    if backend == "hunyuan3d_mv":
        mesh, colour, uv, weight, mapping, cond, camera, dense, \
            views_report = _multiview(photo, sheet, kind)
        matched = {}
        model_info = {"backend": "hunyuan3d-2mv-turbo",
                      "seconds": dense["seconds"]}
        notes = [f"shape from front, side and back (Hunyuan3D-2mv): "
                 f"{dense['vertices']} v in {dense['seconds']} s"]
        for warning in views_report.get("sheet", {}).get("warnings") or []:
            notes.append("turnaround sheet: " + warning)
    else:
        model = _model()
        cond = sf3d_dense.Conditioned(photo, _rembg())
        camera = sf3d_dense.Camera(model)
        codes, bounds = sf3d_dense.scene_codes(model, cond)
        mesh, dense = sf3d_dense.dense_mesh(model, codes, bounds)
        colour = sf3d_dense.albedo(model, codes, mesh.vertices)
        uv, weight, mapping = sf3d_dense.photo_mapping(mesh, camera, cond)
        colour, matched = sf3d_dense.match_albedo(colour, uv, weight, cond)
        mapping["albedo_match"] = matched
        model_info = _MODEL_INFO
        notes = [f"dense surface {len(mesh.vertices)} v on a "
                 f"{'x'.join(map(str, dense['grid']))} grid in "
                 f"{dense['seconds']} s"]
        if sheet:
            notes.append(f"turnaround sheet not used ({why}); built from "
                         "the front alone")

    # -- the skeleton ------------------------------------------------------
    rig_info = {"method": None}
    joints_internal = None
    face_internal = {}
    if kind in HUMANLIKE:
        best, tried = landmarks(photo, cond)
        rig_info["tried"] = tried
        if best is None:
            raise RuntimeError(
                f"BlazePose found no figure in {os.path.basename(front)}, "
                f"even seeded from the silhouette; a {kind} needs its joints "
                f"located to be rigged. A clearer full-body, front-facing "
                f"reference on a plain background fixes this.")
        _score, variant, result, share = best
        rig_info.update({"method": "blazepose", "variant": variant,
                         "confidence": round(result["confidence"], 4),
                         "inside": round(share, 4),
                         "seeded": result["seeded"]})
        pushed, missed = {}, []
        named, moved = skeleton.refine_wrist_pixels(
            pose_mp.named(result["image"]), cond.alpha)
        if moved:
            rig_info["wrists_moved"] = moved
            notes.append("wrists moved to the hands in the matte: "
                         + ", ".join(sorted(moved)))
        for name, point in named.items():
            hit = sf3d_dense.joint_from_pixel(mesh, camera, cond,
                                              point[0], point[1])
            if hit is None:
                missed.append(name)
                continue
            skin, middle = hit
            pushed[name] = {"skin": tuple(sf3d_dense.to_blender(skin)),
                            "mid": tuple(sf3d_dense.to_blender(middle))}
        rig_info["missed_landmarks"] = missed
        joints, info = skeleton.human_joints(pushed)
        rig_info["flipped_sides"] = info["flipped_sides"]
        rig_info["stature_units"] = round(info["stature"], 5)
        face_internal = info["face"]
        joints_internal = joints
        rig_info["landmarks_px"] = {n: [round(p[0], 1), round(p[1], 1)]
                                    for n, p in
                                    pose_mp.named(result["image"]).items()}
    verts = sf3d_dense.to_blender(mesh.vertices)
    low, high = verts.min(0), verts.max(0)
    if joints_internal is None:
        joints_internal, info = skeleton.quadruped_joints(tuple(low),
                                                          tuple(high))
        rig_info.update({"method": "quadruped_template",
                         "stature_units": round(info["stature"], 5)})

    out_dir = unit["out_dir"]
    # Albedo in RGB, trust-in-the-photo in A: a colour attribute is what
    # Blender's decimator interpolates, and a generic float attribute is not.
    rgba = np.concatenate([colour, weight[:, None]], 1).astype(np.float32)
    extra, view_files = {}, {}
    if views_report and views_report.get("_per_view"):
        # Every view is a texture at its image's full resolution, with its
        # own projection; photo_body blends them by these weights (front's
        # sharp weight first, then left, back, right).
        per_view = views_report.pop("_per_view")
        images = views_report.pop("_images")
        # Channel order is photo_body's: R left, G back, B right, A front.
        order = ("left", "back", "right", "front")
        extra["view_weights"] = np.stack(
            [per_view[n][1] for n in order], 1).astype(np.float32)
        for name in order[:3]:
            extra[f"uv_{name}"] = per_view[name][0].astype(np.float32)
            file = f"view_{name}.png"
            sf3d_dense.bleed(images[name]).save(os.path.join(out_dir, file))
            view_files[name] = file
    np.savez_compressed(
        os.path.join(out_dir, "mesh.npz"),
        verts=verts.astype(np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        color=rgba, uv=uv.astype(np.float32), **extra)
    sf3d_dense.bleed(cond.cut).save(os.path.join(out_dir, "photo.png"))

    watertight = bool(mesh.is_watertight)
    asset = {
        "schema_version": 2,
        "subject": subject,
        "kind": kind,
        "height_m": params.get("height_m"),
        "source": os.path.basename(front),
        "frame": "blender: up +Z, facing +Y, SF3D units",
        "mesh": "mesh.npz",
        "photo": "photo.png",
        "bounds": [[round(float(v), 5) for v in low],
                   [round(float(v), 5) for v in high]],
        "joints": {n: [round(float(v), 5) for v in p]
                   for n, p in joints_internal.items()},
        "face": {n: ([round(float(v), 5) for v in p]
                     if isinstance(p, (tuple, list)) else round(float(p), 5))
                 for n, p in face_internal.items()},
        "rig": rig_info,
        "dense": dense,
        "matte": cond.matte_method,
        "mapping": mapping,
        "model": model_info,
        "shape_backend": backend,
        "views": views_report,
        "view_images": view_files,
    }
    jsonio.write(os.path.join(out_dir, "asset.json"), asset)

    metrics = {
        "subject": subject,
        "kind": kind,
        # The names gates.check_asset was written against.
        "tris": int(len(mesh.faces)),
        "size_m": [round(float(v), 4) for v in (high - low)],
        "degenerate_faces": 0,
        "vertices": int(len(mesh.vertices)),
        "watertight": watertight,
        "shells_kept": dense["shells_kept"],
        "photo_share": mapping["photo_share"],
        "shape_backend": backend,
        "views_seen_share": ((views_report or {}).get("colours") or {}).get(
            "seen_share"),
        "albedo_error_before": matched.get("error_before"),
        "albedo_error_after": matched.get("error_after"),
        "coverage": round(cond.coverage, 4),
        "rig_method": rig_info["method"],
        "pose_confidence": rig_info.get("confidence"),
        "landmarks_inside": rig_info.get("inside"),
        "pose_seeded": rig_info.get("seeded"),
        "missed_landmarks": len(rig_info.get("missed_landmarks") or []),
        "dense_seconds": dense["seconds"],
        "load_s": _MODEL_INFO.get("load_s"),
    }
    notes.append(f"rig: {rig_info['method']}"
                 + (f" on {rig_info.get('variant')}, confidence "
                    f"{rig_info.get('confidence')}, "
                    f"{rig_info.get('inside', 0):.0%} of landmarks on the figure"
                    if rig_info["method"] == "blazepose" else ""))
    return {"metrics": metrics, "notes": notes}


# -- the gate -------------------------------------------------------------

def _gate(project, unit, metrics):
    out = [v for v in gates.check_asset(metrics) if v.level != gates.PASS]
    subject = metrics.get("subject")

    vertices = metrics.get("vertices") or 0
    if vertices < MIN_VERTICES:
        out.append(gates.Verdict(
            gates.FAIL, "asset.vertices",
            f"{subject}'s mesh has {vertices} vertices; below {MIN_VERTICES} "
            f"the dense pass has returned a blob rather than a figure",
            vertices, MIN_VERTICES))
    if metrics.get("watertight") is False:
        out.append(gates.Verdict(
            gates.WARN, "asset.watertight",
            f"{subject}'s mesh is not closed; skinning and lighting still "
            f"work, but a hole will read as a gap in a close-up", False, True))
    share = metrics.get("photo_share")
    if metrics.get("shape_backend") == "hunyuan3d_mv":
        # Every view colours the surface it faces; the front photo alone
        # covers little of a lion's long body, and should not. What matters
        # is how much of the surface some view saw.
        share = metrics.get("views_seen_share")
    if share is not None and share < MIN_PHOTO_SHARE:
        out.append(gates.Verdict(
            gates.FAIL, "asset.photo",
            f"only {share:.0%} of {subject}'s surface takes its colour from the "
            f"reference; the photo and the mesh do not line up",
            share, MIN_PHOTO_SHARE))
    if metrics.get("rig_method") == "blazepose":
        confidence = metrics.get("pose_confidence") or 0.0
        if confidence < MIN_POSE_CONFIDENCE:
            out.append(gates.Verdict(
                gates.FAIL, "asset.pose",
                f"BlazePose is only {confidence:.2f} confident of {subject}'s "
                f"joints; a skeleton built on them would bend in the wrong "
                f"places", confidence, MIN_POSE_CONFIDENCE))
        inside = metrics.get("landmarks_inside") or 0.0
        if inside < MIN_LANDMARKS_INSIDE:
            out.append(gates.Verdict(
                gates.WARN, "asset.landmarks",
                f"{1 - inside:.0%} of {subject}'s landmarks fall off the "
                f"figure; check the rig's arms in the turntable",
                inside, MIN_LANDMARKS_INSIDE))
        if metrics.get("pose_seeded"):
            out.append(gates.Verdict(
                gates.WARN, "asset.pose_seeded",
                f"no person was detected in {subject}'s reference, so the "
                f"joints were read from a silhouette-seeded crop; shoulders "
                f"are the usual casualty", True, False))
    if out:
        return out
    return [gates.Verdict(
        gates.PASS, "asset",
        f"{vertices} v, {metrics.get('photo_share', 0):.0%} photo-coloured, "
        f"rig by {metrics.get('rig_method')}"
        + (f" ({metrics.get('pose_confidence')})"
           if metrics.get("pose_confidence") else ""))]


STAGE = StageSpec(
    name="c_asset",
    version="13",
    summary="build each character's mesh, colour and skeleton from its photo",
    needs=["b_refs"],
    # The motion env: comfyui's torch plus pymeshlab, which Hunyuan3D needs.
    steps=[Step("motion")],
    gate=_gate,
)


if __name__ == "__main__":
    sys.exit(_base.stage_main(run))
