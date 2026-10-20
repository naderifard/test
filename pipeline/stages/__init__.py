"""The fourteen stages of §12.

Each module in this package is a standalone CLI: the graph imports it in the
host to plan the work, then re-invokes it as ``python -m
pipeline.stages.<name>`` in whichever interpreter the work needs.
"""
