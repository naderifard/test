"""Blender AI Studio's production pipeline: a staged graph over a
content-addressed artifact store.

The entry point is ``python -m pipeline`` (see :mod:`pipeline.cli`).
"""

__all__ = ["cli", "envs", "gates", "graph", "jsonio", "project", "store",
           "validate"]
