"""Style packs: the look of the film as data, plus the hooks data cannot express.

§8 asks for three interchangeable looks — ``miniature``, ``stylized``,
``photoreal`` — and for assets to reference *material roles* (``skin``,
``metal_blade``, ``gold_leaf``) rather than concrete materials. A pack is
therefore two things:

* ``style.json`` — the declarative part: palette, one recipe per role, render
  settings, lighting rig, camera defaults, post-processing, the viseme profile.
* ``shading.py`` / ``postfx.py`` — hook modules, imported *inside Blender only*,
  for the parts a JSON file cannot say (node graphs, compositor trees).

Everything here is stdlib-only and importable from any interpreter, because the
host plans with it and Blender builds with it.

Colours may be written as ``"@lapis"`` to reference the pack's palette; the
loader resolves those once, so a pack has one place where its blue is decided.
"""

from __future__ import annotations

import copy
import json
import os

PACKS = ("miniature", "stylized", "photoreal")

HERE = os.path.dirname(os.path.abspath(__file__))

# Every role an asset is allowed to ask for. A pack must define all of them, so
# a missing role is a packaging error caught at load rather than a grey object
# discovered in a render three stages later.
ROLES = (
    "skin", "hair", "eye_white", "eye_iris", "lip",
    "cloth_primary", "cloth_secondary", "cloth_trim",
    "leather", "metal_armor", "metal_blade", "gold_leaf",
    "wood", "stone", "ground", "foliage", "water", "sky",
    "ink_outline", "backdrop",
)


class StyleError(RuntimeError):
    pass


class StylePack:
    """One look, loaded and resolved."""

    def __init__(self, name, data):
        self.name = name
        self.data = data

    def __repr__(self):
        return f"<StylePack {self.name}>"

    # -- the declarative half -----------------------------------------------
    @property
    def render(self):
        return self.data["render"]

    @property
    def lighting(self):
        return self.data["lighting"]

    @property
    def camera(self):
        return self.data["camera"]

    @property
    def postfx(self):
        return self.data["postfx"]

    @property
    def env(self):
        return self.data["env"]

    @property
    def visemes(self):
        return self.data["visemes"]

    def material(self, role):
        """The recipe for a material role, by name."""
        try:
            return self.data["materials"][role]
        except KeyError:
            raise StyleError(
                f"style pack {self.name!r} has no material role {role!r}; "
                f"known roles: {', '.join(sorted(self.data['materials']))}"
            ) from None

    def get(self, *path, default=None):
        node = self.data
        for step in path:
            if not isinstance(node, dict) or step not in node:
                return default
            node = node[step]
        return node

    # -- the hook half ------------------------------------------------------
    def hook(self, module):
        """Import one of the pack's hook modules, or None if it has none.

        Called only inside Blender: these modules import ``bpy``. Kept lazy so
        the host can plan with a pack it could never execute.
        """
        import importlib
        try:
            return importlib.import_module(
                f"blender_ai_studio.styles.{self.name}.{module}")
        except ImportError:
            return None


_CACHE = {}


def load(name):
    """Load a style pack by name. Cached, because every stage asks for it."""
    name = (name or "miniature").strip().lower()
    if name in _CACHE:
        return _CACHE[name]
    if name not in PACKS:
        raise StyleError(f"unknown style pack {name!r}; "
                         f"expected one of {', '.join(PACKS)}")
    path = os.path.join(HERE, name, "style.json")
    if not os.path.exists(path):
        raise StyleError(f"style pack {name!r} has no style.json at {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data = _resolve(data, data.get("palette", {}), name)
    missing = [role for role in ROLES if role not in data.get("materials", {})]
    if missing:
        raise StyleError(f"style pack {name!r} is missing material role(s): "
                         f"{', '.join(missing)}")
    pack = StylePack(name, data)
    _CACHE[name] = pack
    return pack


def _resolve(node, palette, pack_name):
    """Replace every ``"@name"`` with the palette colour it refers to."""
    if isinstance(node, dict):
        return {k: _resolve(v, palette, pack_name) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(v, palette, pack_name) for v in node]
    if isinstance(node, str) and node.startswith("@"):
        key = node[1:]
        if key not in palette:
            raise StyleError(f"style pack {pack_name!r} refers to palette "
                             f"colour {node!r}, which it does not define")
        return copy.deepcopy(palette[key])
    return node


def describe():
    """One line per pack. Used by ``pipeline doctor`` and the GUI."""
    rows = []
    for name in PACKS:
        try:
            pack = load(name)
        except StyleError as exc:
            rows.append((name, f"BROKEN: {exc}"))
            continue
        rows.append((name, f"{pack.render['engine']} "
                           f"{pack.render['samples']} samples, "
                           f"{pack.env['strategy']}, "
                           f"{pack.visemes['profile']} visemes"))
    return rows



