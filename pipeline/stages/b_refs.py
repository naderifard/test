"""Stage B — the reference-image intake.

§7-B: images arrive as files a person drops into ``refs/``. There is no image
API and no generation here. What this stage does is *look at them* and say, per
character, what is present and what is missing — because the failure it exists to
prevent is discovering at ``c_face`` that Rostam has no front view, three stages
and twenty minutes after that could have been said.

It reports rather than blocks. A missing reference is a warning: M1 builds
preset bodies and hand-picked identities, so the whole slice renders with an
empty ``refs/``. It becomes an error only when a stage actually needs a pixel.

Face detection is the one measurement that needs a model, and §7-B wants it. It
is attempted through OpenCV's bundled Haar cascade if cv2 happens to be
importable and simply recorded as unknown otherwise: a face detector is not worth
a hard dependency in the host interpreter for a check that only ever warns.
"""

from __future__ import annotations

import os
import sys

from . import _base
from ._base import StageSpec, Step, Unit

from pipeline import gates, jsonio, validate

WANTED = ("front", "face", "side")

IMAGE_EXT = validate.IMAGE_EXT


def _images(folder):
    try:
        names = sorted(n for n in os.listdir(folder)
                       if n.lower().endswith(IMAGE_EXT))
    except OSError:
        return []
    return [os.path.join(folder, n) for n in names]


def plan(project, graph):
    resolved = jsonio.read(project.path("story", "production.resolved.json"),
                           default={})
    units = []
    for character in resolved.get("characters") or []:
        cid = character["id"]
        folder = project.refs_dir(cid)
        units.append(Unit(
            f"char_{cid}",
            inputs={"refs": folder},
            params={"subject": cid, "kind": "character",
                    "expect": list(WANTED)},
            out_rel=f"assets/characters/{cid}/refs",
            label=f"refs/{cid}",
        ))
    for env in resolved.get("envs") or []:
        eid = env["id"]
        units.append(Unit(
            f"env_{eid}",
            inputs={"refs": project.env_refs_dir(eid)},
            params={"subject": eid, "kind": "env", "expect": []},
            out_rel=f"assets/envs/{eid}/refs",
            label=f"refs/env/{eid}",
        ))
    return units


def nothing_to_do(project):
    return "no resolved production plan — a_production has not run"


def _faces(path):
    """Number of faces found, or ``None`` when no detector is available."""
    try:
        import cv2
    except ImportError:
        return None
    try:
        cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades,
                         "haarcascade_frontalface_default.xml"))
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None or cascade.empty():
            return None
        return len(cascade.detectMultiScale(image, 1.1, 5))
    except Exception:                                           # noqa: BLE001
        return None


def _slot(name):
    """Which of the wanted views a filename claims to be."""
    stem = os.path.splitext(os.path.basename(name))[0].lower()
    for slot in WANTED:
        if stem.endswith("_" + slot) or stem == slot:
            return slot
    return None


def run(job, unit):
    folder = unit["inputs"]["refs"]
    subject = unit["params"]["subject"]
    expect = unit["params"].get("expect") or []

    measured, found = [], {}
    for path in _images(folder):
        stats = gates.measure_image(path)
        stats["path"] = os.path.basename(path)
        slot = _slot(path)
        if slot:
            stats["slot"] = slot
            found[slot] = stats["path"]
        if slot in ("front", "face"):
            faces = _faces(path)
            if faces is not None:
                stats["faces"] = faces
        measured.append(stats)

    missing = [slot for slot in expect if slot not in found]
    # Written unconditionally: the graph treats a stage that produced no files
    # as a failure, and "this character has no references yet" is a real,
    # cacheable answer that the next stage needs to read.
    jsonio.write(_base.out_path(unit, "refs.json"), {
        "subject": subject,
        "kind": unit["params"]["kind"],
        "folder": job.project.rel(folder),
        "images": measured,
        "slots": found,
        "missing": missing,
        "expected": expect,
    })
    return {
        "metrics": {
            "subject": subject,
            "count": len(measured),
            # The gate runs before the artifact is committed, so it can only see
            # what the run put in `metrics`. The per-image statistics travel here
            # rather than being re-read off disk.
            "images": measured,
            "slots": sorted(found),
            "missing": missing,
            "faceless": [s["path"] for s in measured
                         if s.get("faces") == 0],
            "smallest_edge": min(
                [min(s.get("width", 0), s.get("height", 0)) for s in measured],
                default=0),
        },
        "notes": ([f"missing {', '.join(missing)}"] if missing else
                  [f"{len(measured)} image(s)"]),
    }


def _gate(project, unit, metrics):
    out = list(gates.check_refs(metrics.get("subject", unit.uid),
                               metrics.get("images") or []))
    missing = metrics.get("missing") or []
    if missing and metrics.get("count"):
        out.append(gates.Verdict(gates.WARN, "refs.views",
                                 f"no {', '.join(missing)} view; the fitters "
                                 "will fall back to presets",
                                 len(missing), 0))
    for name in metrics.get("faceless") or []:
        out.append(gates.Verdict(gates.WARN, "refs.face",
                                 f"no face detected in {name}"))
    return out


STAGE = StageSpec(
    name="b_refs",
    version="1",
    summary="measure the reference images a person dropped into refs/",
    needs=["a_production"],
    steps=[Step("host")],
    gate=_gate,
)


if __name__ == "__main__":
    sys.exit(_base.stage_main(run))

