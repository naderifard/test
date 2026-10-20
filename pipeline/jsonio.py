"""JSON read/write with the conventions the whole pipeline relies on.

Three rules, applied everywhere:

* UTF-8 without escapes. Dialogue is Persian; ``ensure_ascii=True`` would turn
  every line into ``\\u0631\\u0633\\u062a`` and make artifacts undiffable.
* Sorted keys and a trailing newline. Artifacts are content-addressed, so two
  runs that produce the same data must produce the same *bytes*.
* Atomic writes. A stage killed mid-write must not leave a half-parsed
  ``shot.json`` that the next run happily loads.
"""

from __future__ import annotations

import json
import os
import tempfile


def read(path, default=None):
    """Load JSON. Returns ``default`` when the file is missing."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        if default is None:
            raise
        return default


def write(path, data):
    """Write JSON atomically. Returns the path."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=folder, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(dumps(data))
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise
    return path


def dumps(data):
    """The canonical text form: sorted, indented, UTF-8, newline-terminated."""
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
