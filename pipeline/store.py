"""Content-addressed artifact store.

Every artifact is keyed by the hash of *everything that could change it*: the
stage's name and version, the hashes of its inputs, and its parameters. Change
the voice model and only the voice artifacts and their descendants go stale;
change a light and the character bodies survive untouched.

Layout inside a project::

    .cache/
      objects/<stage>/<key>/          committed artifact files
      objects/<stage>/<key>/_meta.json
      filehash.json                   size+mtime -> sha256 memo

The store is the resume mechanism. The *readable* project tree (``assets/``,
``motion/``, ``renders/`` ...) is materialised out of it by hardlink where the
filesystem allows and by copy where it does not, so a human can open
``assets/characters/rostam/body.blend`` without knowing the store exists.

Hardlinks are attempted first because motion caches and renders are large and a
project may hold hundreds of them; on NTFS a same-volume hardlink is free. They
are safe here only because committed store objects are treated as immutable —
nothing in the pipeline ever opens a materialised file for writing in place.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time

from . import jsonio

CHUNK = 1 << 20


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(value):
    """Stable text for any JSON-able value, for hashing."""
    return jsonio.dumps(value)


class Store:
    """The artifact cache for one project."""

    def __init__(self, cache_dir):
        self.root = os.path.abspath(cache_dir)
        self.objects = os.path.join(self.root, "objects")
        self.logs = os.path.join(self.root, "logs")
        self._memo_path = os.path.join(self.root, "filehash.json")
        self._memo = None
        self._memo_dirty = False

    # -- file hashing -------------------------------------------------------
    def _load_memo(self):
        if self._memo is None:
            self._memo = jsonio.read(self._memo_path, default={})
        return self._memo

    def hash_file(self, path):
        """sha256 of a file, memoised on (size, mtime_ns).

        Motion caches and .blend files run to tens of megabytes and are hashed
        on every graph walk. Re-reading them each time turns a cache *hit* into
        the slowest part of the run, which defeats the point.
        """
        path = os.path.abspath(path)
        stat = os.stat(path)
        stamp = f"{stat.st_size}:{stat.st_mtime_ns}"
        memo = self._load_memo()
        hit = memo.get(path)
        if hit and hit.get("stamp") == stamp:
            return hit["sha256"]
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                block = fh.read(CHUNK)
                if not block:
                    break
                digest.update(block)
        value = digest.hexdigest()
        memo[path] = {"stamp": stamp, "sha256": value}
        self._memo_dirty = True
        return value

    def hash_path(self, path):
        """Hash a file, or a directory as the sorted list of its files."""
        path = os.path.abspath(path)
        if os.path.isdir(path):
            parts = []
            for folder, dirs, files in os.walk(path):
                dirs.sort()
                for name in sorted(files):
                    full = os.path.join(folder, name)
                    rel = os.path.relpath(full, path).replace("\\", "/")
                    parts.append(rel + ":" + self.hash_file(full))
            return _digest("\n".join(parts))
        return self.hash_file(path)

    def flush(self):
        if self._memo_dirty:
            jsonio.write(self._memo_path, self._memo)
            self._memo_dirty = False

    # -- keys ---------------------------------------------------------------
    def key(self, stage, version, inputs, params):
        """The cache key for one unit of work.

        ``inputs`` maps a label to either a path on disk or an already-computed
        hash string. Labels are part of the key: swapping which file feeds
        ``body`` and which feeds ``face`` must produce a different artifact even
        when the file set is identical.
        """
        resolved = {}
        for label in sorted(inputs):
            value = inputs[label]
            if value is None:
                resolved[label] = None
            elif isinstance(value, (list, tuple)):
                resolved[label] = [self._resolve(v) for v in value]
            else:
                resolved[label] = self._resolve(value)
        payload = {
            "stage": stage,
            "version": str(version),
            "inputs": resolved,
            "params": params,
        }
        return _digest(canonical(payload))[:40], resolved

    def _resolve(self, value):
        if isinstance(value, str) and len(value) == 64 and _is_hex(value):
            return value  # already a digest
        if isinstance(value, (dict, list, int, float, bool)):
            return _digest(canonical(value))
        path = os.path.abspath(str(value))
        if not os.path.exists(path):
            return "missing:" + os.path.basename(path)
        return self.hash_path(path)

    # -- entries ------------------------------------------------------------
    def entry_dir(self, stage, key):
        return os.path.join(self.objects, stage, key)

    def has(self, stage, key):
        return os.path.exists(os.path.join(self.entry_dir(stage, key), "_meta.json"))

    def meta(self, stage, key):
        return jsonio.read(os.path.join(self.entry_dir(stage, key), "_meta.json"),
                           default={})

    def commit(self, stage, key, source_dir, meta):
        """Move a finished unit's output directory into the store.

        The move is staged through a sibling ``.incoming`` directory and only
        gains its ``_meta.json`` at the very end, so an interrupted commit
        leaves an entry that :meth:`has` correctly calls absent rather than a
        half-populated one it would call present.
        """
        target = self.entry_dir(stage, key)
        if os.path.exists(target):
            shutil.rmtree(target, ignore_errors=True)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        staging = target + ".incoming"
        if os.path.exists(staging):
            shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(source_dir, staging)
        os.replace(staging, target)
        record = dict(meta)
        record["stage"] = stage
        record["key"] = key
        record["committed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        record["files"] = self._manifest(target)
        jsonio.write(os.path.join(target, "_meta.json"), record)
        self.flush()
        return target

    def _manifest(self, folder):
        listing = {}
        for base, dirs, files in os.walk(folder):
            dirs.sort()
            for name in sorted(files):
                if name == "_meta.json":
                    continue
                full = os.path.join(base, name)
                rel = os.path.relpath(full, folder).replace("\\", "/")
                listing[rel] = {"bytes": os.path.getsize(full),
                                "sha256": self.hash_file(full)}
        return listing

    def materialise(self, stage, key, dest_dir):
        """Publish a committed entry to a readable path in the project tree.

        Returns the list of project-relative files written.
        """
        source = self.entry_dir(stage, key)
        if not self.has(stage, key):
            raise KeyError(f"{stage}/{key} is not in the store")
        written = []
        for rel in sorted(self.meta(stage, key).get("files", {})):
            src = os.path.join(source, rel.replace("/", os.sep))
            dst = os.path.join(dest_dir, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            _link_or_copy(src, dst)
            written.append(dst)
        return written


def _link_or_copy(src, dst):
    if os.path.exists(dst):
        try:
            if os.path.samefile(src, dst):
                return dst
        except OSError:
            pass
        os.unlink(dst)
    try:
        os.link(src, dst)
    except (OSError, AttributeError, NotImplementedError):
        shutil.copy2(src, dst)
    return dst


def _is_hex(text):
    try:
        int(text, 16)
    except ValueError:
        return False
    return True
