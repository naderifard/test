"""Per-stage interpreter resolution (§10 rule 2).

These models have genuinely conflicting torch and CUDA requirements, so a stage
declares the *logical* environment it needs — ``host``, ``ml``, ``blender`` —
and this module resolves that to a real executable on this machine.

Two hard-won details are encoded here:

* ``conda run -n <env>`` does not work on this machine. The interpreter inside
  the environment is invoked directly.
* An environment that is merely *named* is not an environment. Resolution
  checks the executable exists and fails with the path it looked at, because a
  stage that silently falls back to the host interpreter fails much later and
  much more confusingly.
"""

from __future__ import annotations

import os
import shutil
import sys

CONDA_ROOT = os.path.expanduser("~/anaconda3/envs")

# Logical env -> candidate conda envs in preference order. The first one that
# exists and imports its probe wins, so a machine missing `comfyui` still runs.
ENV_CANDIDATES = {
    "host": [],                     # whatever interpreter is running the graph
    "ml": ["comfyui", "fooocus"],   # torch + torchaudio + transformers, CUDA
    "audio": ["comfyui", "fooocus"],
    "video": ["moviepy2", "comfyui"],
    # GVHMR, MoMask, MuseTalk and SMPL-X fitting. A venv layered on comfyui
    # (``--system-site-packages``) so it adds packages without a second copy
    # of torch on a nearly full drive; see ENV_PATHS.
    "motion": ["motion", "comfyui"],
}

# Environments that are not conda envs: name -> interpreter. The motion venv
# lives beside the models rather than under anaconda3.
ENV_ROOT = os.environ.get("PIPELINE_ENV_ROOT") or "D:/work/Shahnameh/envs"
ENV_PATHS = {
    "motion": os.path.join(ENV_ROOT, "motion", "Scripts", "python.exe")
    if os.name == "nt" else os.path.join(ENV_ROOT, "motion", "bin", "python"),
}

ENV_PROBE = {
    "ml": "import torch, torchaudio",
    "audio": "import torch, torchaudio",
    "video": "import numpy",
    "motion": "import torch, smplx",
}

_CACHE = {}

# -- where the large, unversioned things live ------------------------------
#
# Model weights and vendored model source are not in the repo (.gitignore has
# `models/` and `*.safetensors`) and they are far too big for the drive the
# project used to sit on. They are also not the kind of thing a stage should
# hardcode: SF3D's checkpoint is one `sf3d` subdirectory among several to come.
#
# So the same rule as the interpreters applies -- a stage names what it needs
# logically, this module turns that into a path on this machine, and a missing
# one fails with the path it looked at rather than deep inside a loader.
MODEL_ROOT = os.environ.get("PIPELINE_MODEL_ROOT") or "D:/work/Shahnameh/models"
VENDOR_ROOT = os.environ.get("PIPELINE_VENDOR_ROOT") or "D:/work/Shahnameh/vendor"

# Hugging Face keeps its cache in ~/.cache/huggingface by default, which on this
# machine is the nearly-full system drive -- and it had already grown to 16 GB
# there. Setting it per-subprocess rather than asking the user to set a global
# environment variable keeps the arrangement in version control, where a future
# reader can see why it is not the default.
HF_HOME = os.environ.get("HF_HOME") or os.path.join(MODEL_ROOT, "hf")

# rembg keeps its ONNX matting models in ~/.u2net unless U2NET_HOME says
# otherwise, and downloads them from a GitHub release on first use. The same
# reasoning as HF_HOME applies twice over here: the default is the full system
# drive, and the download is 176 MB from a host that delivers it to this machine
# at 64 kB/s -- a first run that spends thirteen minutes at 29% and then times
# out is indistinguishable from a hang.
U2NET_HOME = os.environ.get("U2NET_HOME") or os.path.join(MODEL_ROOT, "rembg")

# Logical model name -> subdirectory of MODEL_ROOT. A stage asks for
# ``envs.model_dir("sf3d")`` and gets a directory it can trust exists.
MODEL_DIRS = {
    "sf3d": "sf3d",
    "depth_anything_v2": "depth_anything_v2",
    "mediapipe": "mediapipe",
    "musetalk": "musetalk",
    "gvhmr": "gvhmr",
    "momask": "momask",
    "body_models": "body_models",
}

# Logical env -> extra sys.path entries its subprocess needs. SF3D is a vendored
# checkout rather than an installed package, and its texture path imports two
# extensions that need a C++ toolchain this machine has not got; `shims` holds
# stand-ins that raise if anything actually calls them, so the geometry path --
# which needs neither -- can import the package at all.
VENDOR_PATHS = {
    "ml": ["stable-fast-3d", "shims"],
    # Seed-VC is imported as its own top-level ``modules`` package by
    # pipeline/voice_seedvc.py, which puts it on sys.path itself; listed here so
    # `doctor` can say whether the checkout is present.
    "audio": ["seed-vc"],
    # c_asset runs here: SF3D (and its shims) and Hunyuan3D-2's shape code.
    "motion": ["stable-fast-3d", "shims", "Hunyuan3D-2"],
}


class EnvError(RuntimeError):
    pass


def model_dir(name, required=True):
    """The directory holding one model's weights, checked.

    A stage that wants SF3D asks for it by name. The alternative -- a path
    literal in the stage -- means the failure arrives as a FileNotFoundError
    from inside a loader, naming a file rather than the thing that is missing.
    """
    path = os.path.abspath(os.path.join(MODEL_ROOT,
                                        MODEL_DIRS.get(name, name)))
    if required and not os.path.isdir(path):
        raise EnvError(
            f"the {name!r} model directory is not on this machine: "
            f"{path}\n"
            f"  model root: {MODEL_ROOT} "
            f"(override with $PIPELINE_MODEL_ROOT)")
    return path


def vendor_paths(logical):
    """Extra sys.path entries a logical env's subprocess needs, those that exist."""
    out = []
    for rel in VENDOR_PATHS.get(logical) or []:
        path = os.path.abspath(os.path.join(VENDOR_ROOT, rel))
        if os.path.isdir(path):
            out.append(path)
    return out


def step_env(logical):
    """Environment variables to add for one step's subprocess.

    Returned rather than applied, so the graph stays the only place that builds
    a subprocess environment and a test can read this without spawning anything.
    """
    extra = {"HF_HOME": HF_HOME, "U2NET_HOME": U2NET_HOME}
    paths = vendor_paths(logical)
    if paths:
        extra["PIPELINE_VENDOR_PATH"] = os.pathsep.join(paths)
    dlls = env_dll_dirs(logical)
    if dlls:
        extra["PATH"] = os.pathsep.join(dlls + [os.environ.get("PATH", "")])
    return extra, paths


def env_dll_dirs(logical):
    """The folders a conda env's activation would have put on PATH.

    ``conda run`` does not work on this machine, so env interpreters are run
    directly -- which skips activation, and with it the env's own ``bin`` and
    ``Library/bin`` on PATH. Most of torch does not notice. Its runtime kernel
    compiler does: ``nvrtc-builtins64_124.dll`` sits in the env's ``bin`` and a
    complex ``abs`` inside an STFT failed with "failed to open
    nvrtc-builtins64_124.dll" until those folders were put back.
    """
    if logical in (None, "host", "blender"):
        return []
    try:
        exe = resolve(logical, required=False)
    except EnvError:
        return []
    if not exe:
        return []
    roots = [os.path.dirname(exe)]
    base = _venv_base(exe)
    if base:
        # A venv's torch is its base env's, and so are the DLLs torch loads.
        roots.append(base)
    found = [os.path.join(root, *sub.split("/")) for root in roots
             for sub in ("bin", "Library/bin")]
    return [path for path in found if os.path.isdir(path)]


def _venv_base(exe):
    """The base interpreter's folder of a venv (``pyvenv.cfg``'s ``home``)."""
    cfg = os.path.join(os.path.dirname(os.path.dirname(exe)), "pyvenv.cfg")
    try:
        with open(cfg, encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition("=")
                if key.strip() == "home":
                    return value.strip()
    except OSError:
        return None
    return None


def conda_python(name):
    """Absolute path to a conda env's interpreter, whether or not it exists."""
    if name in ENV_PATHS:
        return ENV_PATHS[name]
    exe = "python.exe" if os.name == "nt" else "bin/python"
    return os.path.join(CONDA_ROOT, name, *exe.split("/"))


def list_envs():
    try:
        return sorted(name for name in os.listdir(CONDA_ROOT)
                      if os.path.exists(conda_python(name)))
    except OSError:
        return []


def resolve(logical, required=True):
    """The interpreter for a logical env, or None when nothing satisfies it."""
    if logical in _CACHE:
        return _CACHE[logical]
    if logical in (None, "host"):
        _CACHE[logical] = sys.executable
        return sys.executable

    if logical == "blender":
        found = blender_executable()
        if not found and required:
            raise EnvError("Blender was not found. Set BLENDER_EXE or "
                           "blender_exe in config.json.")
        _CACHE[logical] = found
        return found

    candidates = ENV_CANDIDATES.get(logical)
    if candidates is None:
        # An explicit env name, used directly.
        candidates = [logical]
    probe = ENV_PROBE.get(logical)
    tried = []
    for name in candidates:
        exe = conda_python(name)
        tried.append(exe)
        if not os.path.exists(exe):
            continue
        if probe and not _probe(exe, probe):
            continue
        _CACHE[logical] = exe
        return exe
    if required:
        raise EnvError(
            f"No interpreter satisfies the {logical!r} environment.\n"
            + "\n".join("  tried " + t for t in tried)
            + (f"\n  each must satisfy: {probe}" if probe else "")
            + f"\n  conda envs present: {', '.join(list_envs()) or '(none)'}")
    _CACHE[logical] = None
    return None


def _probe(exe, statement):
    import subprocess
    try:
        done = subprocess.run([exe, "-c", statement], capture_output=True,
                              timeout=180)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def blender_executable():
    """Reuse the launcher's search so there is one answer to 'where is Blender'."""
    override = os.environ.get("BLENDER_EXE")
    if override and os.path.exists(override):
        return override
    try:
        from blender_ai_studio.blender_link import find_blender_executable
    except ImportError:
        return shutil.which("blender")
    return find_blender_executable() or shutil.which("blender")


def describe():
    """What this machine can actually run. Printed by ``pipeline doctor``."""
    rows = []
    for logical in ("host", "ml", "audio", "video", "motion", "blender"):
        try:
            exe = resolve(logical, required=False)
        except EnvError:
            exe = None
        rows.append((logical, exe or "(missing)"))
    return rows
