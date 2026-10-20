"""A stage that exists only to prove the graph works.

It writes a small file, reports measurable metrics, and can be told to fail its
gate on purpose. That is enough to exercise the whole machine: keying, the
subprocess hop, per-unit gates, committing, materialising, cache hits, ``--force``
and a failing gate not poisoning the cache.

Kept in the tree rather than in a test fixture because it is also the fastest
way to check that a *new* interpreter can host a stage at all::

    python -m pipeline.stages._selftest --job job.json --step 0 --result out.json
"""

from __future__ import annotations

import os
import sys

from . import _base
from ._base import StageSpec, Step, Unit

from pipeline import gates


PLAN_FILE = ("_selftest", "plan.json")

DEFAULT_PLAN = [{"uid": "alpha", "text": "alpha"},
                {"uid": "beta", "text": "beta"}]


def plan(project, graph):
    """Units come from ``.cache/_selftest/plan.json`` so a test can drive them.

    Each entry is ``{"uid": ..., "text": ..., "should_fail": bool}``. The text
    is a *parameter*, so changing it changes the key and the unit rebuilds —
    which is the property the cache test needs.
    """
    from pipeline import jsonio

    entries = jsonio.read(project.cache(*PLAN_FILE), default=None)
    if entries is None:            # absent, not empty: an empty plan is a plan
        entries = DEFAULT_PLAN
    units = []
    for entry in entries:
        uid = entry["uid"]
        params = {"text": entry.get("text", uid)}
        if entry.get("should_fail"):
            params["should_fail"] = True
        units.append(Unit(uid, inputs=entry.get("inputs") or {}, params=params,
                          out_rel=entry.get("out_rel") or "_selftest",
                          label=f"selftest/{uid}"))
    return units


def nothing_to_do(project):
    return "the selftest plan is empty"


def run(job, unit):
    text = unit["params"].get("text", "")
    target = _base.out_path(unit, "note.txt")
    with open(target, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")
    return {
        "metrics": {
            "bytes_written": os.path.getsize(target),
            "interpreter": os.path.basename(sys.executable),
            "python": ".".join(str(n) for n in sys.version_info[:3]),
            "utf8_ok": text.encode("utf-8").decode("utf-8") == text,
        },
        "notes": [f"wrote {os.path.basename(target)}"],
    }


def _gate(project, unit, metrics):
    if unit.params.get("should_fail"):
        return [gates.Verdict(gates.FAIL, "selftest.forced",
                              "this unit was told to fail")]
    if metrics.get("bytes_written", 0) <= 0:
        return [gates.Verdict(gates.FAIL, "selftest.empty", "wrote nothing")]
    if metrics.get("bytes_written", 0) < 4:
        return [gates.Verdict(gates.WARN, "selftest.small",
                              f"only {metrics['bytes_written']} bytes")]
    return [gates.Verdict(gates.PASS, "selftest",
                          f"{metrics['bytes_written']} bytes in "
                          f"{metrics['interpreter']}")]


STAGE = StageSpec(
    name="_selftest",
    version="1",
    summary="proves the graph, the store and the gates connect",
    steps=[Step("host")],
    gate=_gate,
)


if __name__ == "__main__":
    sys.exit(_base.stage_main(run))

