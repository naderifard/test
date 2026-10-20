"""A project on disk: the directory layout of §5.3, ``project.json``, and the
paths every stage agrees on.

The layout is the contract. Stages never join paths themselves; they ask a
:class:`Project`, so §5.3 lives in exactly one place and a stage cannot quietly
write its output somewhere the next stage will not look.

Two paths are not in §5.3 and are placed here by decision: ``align.json`` and
``visemes.json`` go in ``audio/<line_id>/`` beside the audio they describe,
because they are per-line artifacts and §5.1 names them without siting them.
"""

from __future__ import annotations

import os
import re

from . import jsonio
from .store import Store

ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")

TREE = (
    "story",
    "refs/characters",
    "refs/envs",
    "performance",
    "audio/voices",
    "assets/characters",
    "assets/props",
    "assets/envs",
    "motion",
    "shots",
    "film",
    ".cache",
)

# Handles bracket the spoken line so a shot does not cut the instant the last
# phoneme ends. They are project-level rather than per-shot because a film with
# shot-by-shot handle lengths reads as unevenly paced.
PROJECT_DEFAULTS = {
    "schema_version": 1,
    "name": "",
    "title": "",
    "fps": 24,
    "resolution": [1920, 1080],
    "style_pack": "miniature",
    "language": "fa",
    "head_handle_s": 0.5,
    "tail_handle_s": 0.7,
    "min_shot_s": 0.0,
    "aligner_backend": "auto",
    "silence_lead_policy": "trim",
    # null means <project>/.cache. See CACHE_DIR_ENV above.
    "cache_dir": None,
}

PROJECTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects")


# Where the artifact store lives, when it is not inside the project.
#
# The store is the only part of a project that grows without bound -- a minute
# of film is a thousand PNGs, and §4.2 has "the store never garbage-collects"
# on its list. On a machine whose system drive is nearly full, or whose project
# directory is inside a cloud-synced folder that will dutifully upload every
# rendered frame, the store needs to be somewhere else. Everything else in a
# project is small, diffable and worth keeping beside the code.
#
# Resolution order, first hit wins:
#   1. $PIPELINE_CACHE_DIR/<project name>   -- one setting for every project
#   2. project.json's "cache_dir"           -- per project; relative paths are
#                                              resolved against the project root
#   3. <project>/.cache                     -- unchanged default
CACHE_DIR_ENV = "PIPELINE_CACHE_DIR"


class Project:
    """One film. Owns its paths and its artifact store."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.name = os.path.basename(self.root)
        self.cache_root = self._resolve_cache_root()
        self.store = Store(self.cache_root)

    def _resolve_cache_root(self):
        shared = os.environ.get(CACHE_DIR_ENV)
        if shared:
            return os.path.abspath(os.path.join(shared, self.name))
        setting = jsonio.read(self.path("project.json"), default={}) \
            .get("cache_dir")
        if setting:
            return os.path.abspath(os.path.join(self.root, setting))
        return self.path(".cache")

    def cache(self, *parts):
        """A path inside the artifact store, wherever the store lives.

        Every ``project.path(".cache", ...)`` in the codebase became a call to
        this, because a store that moves has to move whole: leaving the work
        directory or the run summary behind on the full drive would defeat the
        point.
        """
        joined = "/".join(str(part) for part in parts if part not in (None, ""))
        return os.path.join(self.cache_root, *joined.split("/")) if joined \
            else self.cache_root

    def __repr__(self):
        return f"<Project {self.name} at {self.root}>"

    # -- construction -------------------------------------------------------
    @classmethod
    def create(cls, name, title=None, **settings):
        if not ID_RE.match(name):
            raise ValueError(
                f"{name!r} is not a usable project name: lower-case letters, "
                "digits and underscores, starting with a letter.")
        project = cls(os.path.join(PROJECTS_ROOT, name))
        for folder in TREE:
            os.makedirs(project.path(folder), exist_ok=True)
        if not os.path.exists(project.path("project.json")):
            config = dict(PROJECT_DEFAULTS)
            config["name"] = name
            config["title"] = title or name.replace("_", " ").title()
            config.update({k: v for k, v in settings.items() if v is not None})
            jsonio.write(project.path("project.json"), config)
        return project

    @classmethod
    def open(cls, name_or_path):
        candidate = os.path.abspath(name_or_path)
        if not os.path.isdir(candidate):
            candidate = os.path.join(PROJECTS_ROOT, name_or_path)
        if not os.path.isfile(os.path.join(candidate, "project.json")):
            raise FileNotFoundError(
                f"No project at {name_or_path} (expected a project.json there)")
        return cls(candidate)

    @classmethod
    def list_all(cls):
        if not os.path.isdir(PROJECTS_ROOT):
            return []
        return sorted(name for name in os.listdir(PROJECTS_ROOT)
                      if os.path.isfile(os.path.join(PROJECTS_ROOT, name,
                                                     "project.json")))

    # -- paths --------------------------------------------------------------
    def path(self, *parts):
        joined = "/".join(str(p) for p in parts if p not in (None, ""))
        return os.path.join(self.root, *joined.split("/"))

    def rel(self, path):
        """Project-relative, forward-slashed. This is what goes into JSON."""
        return os.path.relpath(os.path.abspath(path), self.root).replace("\\", "/")

    def refs_dir(self, character_id):
        return self.path("refs", "characters", character_id)

    def env_refs_dir(self, env_id):
        return self.path("refs", "envs", env_id)

    def performance_dir(self, take_id):
        return self.path("performance", take_id)

    def audio_dir(self, line_id):
        return self.path("audio", line_id)

    def voice_ref(self, character_id):
        return self.path("audio", "voices", character_id + ".wav")

    def character_dir(self, character_id):
        return self.path("assets", "characters", character_id)

    def prop_dir(self, prop_id):
        return self.path("assets", "props", prop_id)

    def env_dir(self, env_id):
        return self.path("assets", "envs", env_id)

    def motion_dir(self, take_id):
        return self.path("motion", take_id)

    def shot_dir(self, shot_id):
        return self.path("shots", shot_id)

    def film_dir(self):
        return self.path("film")

    # -- documents ----------------------------------------------------------
    @property
    def config(self):
        merged = dict(PROJECT_DEFAULTS)
        merged.update(jsonio.read(self.path("project.json"), default={}))
        return merged

    def save_config(self, config):
        jsonio.write(self.path("project.json"), config)

    @property
    def production_path(self):
        return self.path("story", "production.json")

    @property
    def production(self):
        return jsonio.read(self.production_path, default={})

    @property
    def rig_overrides_path(self):
        return self.path("story", "rig_overrides.json")

    @property
    def rig_overrides(self):
        return jsonio.read(self.rig_overrides_path, default={})

    def shot(self, shot_id):
        return jsonio.read(self.path("shots", shot_id, "shot.json"), default={})

    # -- convenience over production.json -----------------------------------
    def characters(self):
        return {c["id"]: c for c in self.production.get("characters", [])}

    def envs(self):
        return {e["id"]: e for e in self.production.get("envs", [])}

    def shots(self):
        return {s["id"]: s for s in self.production.get("shots", [])}

    def dialogue_lines(self):
        """Every line in the film, flattened, with its shot id attached."""
        lines = []
        for shot in self.production.get("shots", []):
            for line in shot.get("dialogue", []):
                entry = dict(line)
                entry["shot"] = shot["id"]
                lines.append(entry)
        return lines

    def fps(self):
        return int(self.production.get("fps") or self.config["fps"])
