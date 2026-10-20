"""Loading SF3D on this machine, and getting a mesh out of it.

Stable Fast 3D turns one image into a textured mesh. This module uses only its
geometry half, and it exists as a separate module from the stage that calls it
because getting the model to load here took a sequence of non-obvious decisions
that are worth keeping written down next to the code that depends on them.

Why not ``SF3D.from_pretrained`` and ``run_image``, the two calls the project's
own README uses:

* **from_pretrained builds on the CPU and loads a full state dict beside it**,
  which means two copies of a 1006 M parameter model in host memory at once.
  With 16 GB of RAM, a few GB of free commit and an auto-managed pagefile on a
  nearly-full drive, that is over budget: it failed outright with "the paging
  file is too small" (OS error 1455) in float32, and in float16 it segfaulted on
  some runs and survived on others depending on what else was resident. A load
  that works four times out of five is not a load. Building the parameters, then
  moving them to the card, *then* loading the checkpoint straight into VRAM
  means host memory never holds either copy.

* **run_image goes on to bake a texture**, which needs ``texture_baker`` and
  ``uv_unwrapper`` -- two compiled extensions that want a C++ toolchain this
  machine has not got. Everything up to ``triplane_to_meshes`` is pure PyTorch
  and needs neither, so this module stops there and the texture arrives by a
  different route (projecting the reference image in Blender).

The other three traps, each of which produced a different confusing failure:

* Constructing the model under ``torch.device("cuda")`` runs the parameter
  initialisers on the GPU, and ``trunc_normal_``'s ``erfinv`` sends PyTorch to
  nvrtc, which on this install cannot find ``nvrtc-builtins64_124.dll``. Build
  on the CPU; the initial values are overwritten by the checkpoint anyway.

* Two parameters are declared with an explicit ``dtype=torch.float32`` in SF3D's
  source and so survive ``set_default_dtype``: ``SF3D.bbox`` and the triplane
  tokenizer's ``embeddings``. The latter is the first thing the backbone sees,
  which put float32 activations into a float16 GroupNorm: "mixed dtype: expect
  parameter to have scalar type of Float". SF3D's own ``run_image`` never meets
  this because it wraps everything in ``torch.autocast``; calling ``.half()`` on
  the whole model afterwards is the same answer without the context manager.
  The decoder is the deliberate exception -- SF3D extracts the isosurface with
  autocast explicitly disabled, because a density field sampled in half
  precision quantises the surface.

* **96 entries of the checkpoint are the same tensors under two names.** SF3D
  registers each per-layer ``Modulation`` twice, once on the layer as
  ``norm1/norm2_modulation`` and once in the tokenizer's ``modulations`` list.
  safetensors refuses to write a tensor twice and records the aliasing in the
  file's ``__metadata__`` instead; its own ``load_model`` reads that map and a
  plain ``load_state_dict`` does not. Without re-linking them the load reports
  96 missing keys, and a partially loaded model is not worth running.

One thing this module does **not** do is download DINOv2. SF3D's tokenizer calls
``Dinov2Model.from_pretrained('facebook/dinov2-large')``, which fetches 1.2 GB of
Meta's weights that ``load_state_dict`` then overwrites completely -- the SF3D
checkpoint carries its own DINOv2 (535 tensors, 24 layers) which is both
fine-tuned and architecturally extended, with per-layer camera modulation Meta's
has not got. Only the architecture is wanted, so it is built from the config.
"""

from __future__ import annotations

import json
import os
import struct

# The isosurface resolution is capped by which tetrahedra grid ships with the
# checkpoint. Only `160_tets.npz` is in the SF3D distribution, so asking for 320
# fails inside the extractor with a missing-file error that says nothing about
# resolution. Named here so a caller can check before it asks.
TETS_AVAILABLE = (160,)

# Measured on an RTX 4070 Laptop (8 GB): 1.93 GB of float16 parameters, and a
# peak of 3.97 GB during inference. The margin is for the other work that may be
# on the card -- this is a laptop GPU that a training run may already be using.
VRAM_PARAMETERS_MB = 1980
VRAM_PEAK_MB = 4070
VRAM_MARGIN_MB = 350


class Sf3dError(RuntimeError):
    pass


def free_vram_mb(device=0):
    """Free VRAM in MiB, or None when there is no CUDA device to ask.

    ``torch.cuda.mem_get_info`` reports what the *driver* has free, which is the
    number that matters here: the competing consumer is another process, and
    torch's own allocator statistics cannot see it.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info(device)
    return int(free / (1024 * 1024))


def vram_report(device=0):
    """What the card has, and who else is on it. For a log, and for a gate."""
    try:
        import torch
    except ImportError:
        return {"available": False}
    if not torch.cuda.is_available():
        return {"available": False}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(device),
        "total_mb": int(total / (1024 * 1024)),
        "free_mb": int(free / (1024 * 1024)),
        "used_by_others_mb": int((total - free) / (1024 * 1024)),
        "needed_mb": VRAM_PEAK_MB + VRAM_MARGIN_MB,
    }


def check_vram(device=0):
    """Refuse to start rather than run out of memory halfway through.

    An out-of-memory error from inside a transformer names a tensor shape, not
    the reason, and it arrives after the model has loaded -- so the run has
    already spent thirty seconds and evicted whatever else was cached. Checking
    first turns that into one sentence naming the actual competitor.
    """
    report = vram_report(device)
    if not report.get("available"):
        return report
    if report["free_mb"] < report["needed_mb"]:
        raise Sf3dError(
            "not enough free VRAM for SF3D: %d MiB free of %d on the %s, and "
            "the inference peak is %d MiB plus a %d MiB margin. Something else "
            "on this GPU is holding %d MiB -- wait for it, or run "
            "`nvidia-smi` to see which process it is."
            % (report["free_mb"], report["total_mb"], report["name"],
               VRAM_PEAK_MB, VRAM_MARGIN_MB, report["used_by_others_mb"]))
    return report


def _patch_dinov2_to_skip_the_download():
    """Build DINOv2 from its config instead of downloading weights.

    See the module docstring: the 1.2 GB download is overwritten by the SF3D
    checkpoint on the next line. The config itself is a few KB and is cached.
    """
    from sf3d.models.tokenizers import dinov2 as module

    def from_config_only(name_or_path, *args, **kwargs):
        config = module.Dinov2Config.from_pretrained(name_or_path)
        return module.Dinov2Model(config)

    module.Dinov2Model.from_pretrained = staticmethod(from_config_only)


# What the load costs in host commit, measured on this machine phase by phase:
# importing sf3d 2.66 GB (torch's CUDA context and the cuDNN/cuBLAS libraries,
# not the model), building the parameters 1.11 GB. Streaming the checkpoint
# removed the two further spikes the obvious sequence had -- a second copy
# during model.to(device) and the whole 1.93 GB state dict -- which between them
# were what left 0.88 GB free and made the load fail.
COMMIT_NEEDED_GB = 4.2

# Building the parameters on the meta device would remove the remaining 1.11 GB,
# and accelerate's init_empty_weights is exactly that tool. It does not work
# here: SF3D's vendored Dinov2Model calls ``.to(self.device)`` inside its own
# __init__, which on a meta parameter raises "Cannot copy out of meta tensor".
# Left as a note rather than a half-working branch.


def free_commit_gb():
    """System commit still available, or None off Windows.

    Commit, not free physical memory: what fails here is a reservation, and it
    fails with "the paging file is too small" (OS error 1455) or, when it lands
    somewhere less graceful, an access violation.
    """
    if os.name != "nt":
        return None
    import ctypes

    class _Status(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = _Status()
    status.dwLength = ctypes.sizeof(_Status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullAvailPageFile / float(2 ** 30)


# How long to wait for host memory before giving up, and how often to look.
#
# Not zero, because the competing consumer is a training run whose footprint
# moves: measured a minute apart, free commit on this machine read 0.20 GB and
# then 2.54 GB. Failing a six-unit stage on the first sample would mean failing
# on the other job's batch boundary. Not unbounded either -- a stage that hangs
# is worse than one that says why it stopped.
COMMIT_WAIT_S = 300
COMMIT_POLL_S = 15


def check_commit(wait_s=COMMIT_WAIT_S, poll_s=COMMIT_POLL_S, log=print):
    """Wait for enough host memory, then report it. Refuse if it never comes.

    The same argument as ``check_vram``, for the other resource this machine is
    short of. A load that dies here takes an access violation with no Python
    traceback, which reads like a bug in the stage rather than a machine that is
    out of memory.
    """
    import time

    free = free_commit_gb()
    if free is None:
        return {"available": False}

    deadline = time.time() + max(0, wait_s)
    waited = 0.0
    while free < COMMIT_NEEDED_GB and time.time() < deadline:
        if log:
            log("waiting for host memory: %.2f GB free, need %.1f GB "
                "(%.0f s of %.0f)" % (free, COMMIT_NEEDED_GB, waited, wait_s))
        time.sleep(poll_s)
        waited += poll_s
        free = free_commit_gb()

    if free < COMMIT_NEEDED_GB:
        raise Sf3dError(
            "not enough host memory to load SF3D: %.2f GB of system commit "
            "free after waiting %.0f s, and the load needs about %.1f GB. "
            "Something else on this machine is holding it -- close it, or wait "
            "for it to finish, and run again. (This is host RAM and pagefile, "
            "not VRAM.)" % (free, waited, COMMIT_NEEDED_GB))
    return {"available": True, "free_commit_gb": round(free, 2),
            "needed_commit_gb": COMMIT_NEEDED_GB,
            "waited_s": round(waited)}


def _alias_map(weights_path):
    """The shared-tensor map safetensors wrote instead of duplicating tensors."""
    with open(weights_path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    return header.get("__metadata__") or {}


def _stream_into(model, weights_path, device):
    """Load the checkpoint one tensor at a time, straight onto the device.

    ``load_file`` reads the whole 1.93 GB file at once, and ``model.to(device)``
    does not release the CPU copy it was built from, so the obvious sequence --
    build, move, load -- holds a CPU model, a GPU model and a whole state dict
    within a few hundred MB of each other. Measured on this machine that leaves
    0.88 GB of system commit before the 1.93 GB mapping is made, and the mapping
    fails with "the paging file is too small" (OS error 1455). It failed in the
    real run and survived in a bare one, which is the signature of a peak that
    happens to fit when nothing else is resident.

    Streaming instead: each parameter's CPU storage is replaced by a tensor read
    directly onto the device, so host memory falls as the load proceeds and the
    peak is one tensor rather than a second copy of the model.
    """
    from safetensors import safe_open

    aliases = _alias_map(weights_path)
    loaded, relinked = set(), 0
    with safe_open(weights_path, framework="pt", device=device) as handle:
        available = set(handle.keys())

        def fetch(name):
            """The tensor for this name, following the alias map if need be."""
            nonlocal relinked
            if name in available:
                return handle.get_tensor(name)
            target = aliases.get(name)
            if target and target in available:
                relinked += 1
                return handle.get_tensor(target)
            return None

        # remove_duplicate=False, because the 96 shared Modulation tensors are
        # exactly what is at stake here: PyTorch deduplicates shared parameters
        # in named_parameters() by default and reports only the canonical name,
        # so the alias names would never be visited and would all come back
        # missing -- while state_dict(), which does not deduplicate, lists them.
        for name, tensor in list(model.named_parameters(remove_duplicate=False)) \
                + list(model.named_buffers(remove_duplicate=False)):
            incoming = fetch(name)
            if incoming is None:
                continue
            if incoming.shape != tensor.shape:
                raise Sf3dError(
                    "checkpoint tensor %s is %s, the model wants %s"
                    % (name, tuple(incoming.shape), tuple(tensor.shape)))
            tensor.data = incoming
            loaded.add(name)

    # What the checkpoint is *required* to carry is what state_dict() holds, not
    # every buffer the model owns: SF3D registers ten non-persistent buffers --
    # the attention mask, the tokenizer's image_mean and image_std, the
    # isosurface helper's grid vertices -- which are computed in __init__ and
    # deliberately excluded from the state dict. Requiring those would refuse a
    # checkpoint that is in fact complete, and they reach the device with the
    # model.to(device) that follows this call.
    persistent = set(model.state_dict().keys())
    # Anything in the file that no parameter or buffer claimed. Alias *targets*
    # are themselves real names, so they are already accounted for.
    return (sorted(persistent - loaded), sorted(available - persistent),
            relinked)


def load(model_dir, weights="model.fp16.safetensors", fp16=True, device=None,
         isosurface_resolution=None):
    """Load SF3D and return ``(model, info)``. ``info`` is measurements."""
    import gc
    import time

    import torch
    from omegaconf import OmegaConf

    started = time.time()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config_path = os.path.join(model_dir, "config.yaml")
    weights_path = os.path.join(model_dir, weights)
    for path in (config_path, weights_path):
        if not os.path.exists(path):
            raise Sf3dError(f"SF3D is missing {path}")

    _patch_dinov2_to_skip_the_download()
    from sf3d.system import SF3D

    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    if isosurface_resolution:
        if isosurface_resolution not in TETS_AVAILABLE:
            raise Sf3dError(
                "isosurface_resolution %d needs %d_tets.npz, and only %s ship "
                "with this checkpoint"
                % (isosurface_resolution, isosurface_resolution,
                   ", ".join("%d_tets.npz" % r for r in TETS_AVAILABLE)))
        config.isosurface_resolution = isosurface_resolution

    if fp16:
        torch.set_default_dtype(torch.float16)
    try:
        model = SF3D(config)          # on the CPU: see the module docstring
    finally:
        if fp16:
            torch.set_default_dtype(torch.float32)
    # Straight from the file onto the device, tensor by tensor. The CPU copy
    # built above is released as it goes, rather than all at the end.
    missing, unexpected, relinked = _stream_into(model, weights_path, device)
    if missing or unexpected:
        raise Sf3dError(
            "refusing to run a partially loaded model: %d missing, %d "
            "unexpected. First few: %s"
            % (len(missing), len(unexpected),
               ", ".join(list(missing)[:4] + list(unexpected)[:4])))
    # Nothing may still be on the meta device. Parameters were built there and
    # are real only because the checkpoint replaced them; one that the streaming
    # pass skipped would survive as a shape with no storage and fail later, deep
    # inside a matmul, with a message about meta tensors rather than about a
    # checkpoint that did not carry it.
    stranded = [name for name, tensor
                in list(model.named_parameters(remove_duplicate=False))
                + list(model.named_buffers(remove_duplicate=False))
                if tensor.is_meta]
    if stranded:
        raise Sf3dError(
            "%d tensor(s) are still on the meta device after loading, so the "
            "checkpoint did not carry them: %s"
            % (len(stranded), ", ".join(stranded[:6])))
    # The non-persistent buffers were built for real on the CPU and still have
    # to reach the device. A few MB.
    model.to(device)
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    if fp16:
        model.half()
        model.decoder.float()         # the isosurface grid is float32 by design
    model.eval()

    info = {
        "device": str(device),
        "dtype": "float16" if fp16 else "float32",
        "load_s": round(time.time() - started, 2),
        "relinked_shared_tensors": relinked,
        "isosurface_resolution": int(config.isosurface_resolution),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    if device != "cpu":
        import torch as _torch
        info["vram_after_load_mb"] = int(
            _torch.cuda.memory_allocated() / (1024 * 1024))
    return model, info


# Which rembg model cuts the figure out, and where it runs.
#
# The references from an image generator arrive as RGB with no alpha, so this is
# not optional: SF3D conditions on a masked image and a figure still glued to its
# background produces a mesh with the background in it.
#
# u2netp rather than the u2net default, on two grounds. It is 4.4 MB against
# 176 MB, and the 176 MB comes from a GitHub release that delivers to this
# machine at 64 kB/s -- thirteen minutes to reach 29% and then a read timeout.
# And the job here is the easy case for a matting model: one figure, painted, on
# a plain ground. The measured alternative is in the stage's metrics as
# ``coverage``; if a cut-out ever comes out wrong, that number moves.
#
# CPU providers deliberately. onnxruntime and torch both want to initialise CUDA
# and load their own cuDNN in one process, and this stage already has 2 GB of
# torch parameters on an 8 GB card someone else is using. Matting six 1024x1536
# images on the CPU costs a few seconds in a stage whose model load costs eighty.
REMBG_MODEL = "u2netp"
REMBG_PROVIDERS = ["CPUExecutionProvider"]


def new_rembg_session(model=REMBG_MODEL, providers=None):
    """One matting session, reused for every image in a step."""
    import rembg
    return rembg.new_session(
        model, providers=list(providers or REMBG_PROVIDERS))


def cut_out(image, rembg_session=None, ratio=0.85):
    """Remove the background and centre the figure, as SF3D's own path does."""
    from sf3d.utils import remove_background, resize_foreground

    if rembg_session is None:
        rembg_session = new_rembg_session()
    cut = remove_background(image.convert("RGBA"), rembg_session)
    return resize_foreground(cut, ratio)


def coverage(image):
    """What fraction of the crop the figure fills, from its alpha channel.

    The single most useful number for predicting whether a feature will survive:
    a face that is 3% of the conditioning image has no voxels to be built from,
    however good the model is.
    """
    alpha = image.getchannel("A")
    return sum(alpha.getdata()) / (255.0 * alpha.size[0] * alpha.size[1])


def geometry(model, image):
    """One image in, one trimesh out. Geometry only; no UVs, no texture.

    This is ``run_image``'s batch built by hand, because ``run_image`` continues
    into ``generate_mesh``. Everything up to ``triplane_to_meshes`` is shared
    with it.
    """
    import time

    import numpy as np
    import torch
    import trimesh
    from sf3d.utils import create_intrinsic_from_fov_deg, default_cond_c2w

    device = next(model.parameters()).device
    started = time.time()
    with torch.no_grad():
        mask_cond, rgb_cond = model.prepare_image(image)
        c2w_cond = default_cond_c2w(model.cfg.default_distance).to(device)
        intrinsic, intrinsic_normed_cond = create_intrinsic_from_fov_deg(
            model.cfg.default_fovy_deg,
            model.cfg.cond_image_size,
            model.cfg.cond_image_size,
        )
        batch = {
            "rgb_cond": model.image_processor(rgb_cond,
                                              model.cfg.cond_image_size),
            "mask_cond": model.image_processor(mask_cond,
                                               model.cfg.cond_image_size),
            "c2w_cond": c2w_cond.view(1, 1, 4, 4),
            "intrinsic_cond": intrinsic.to(device).view(1, 1, 3, 3),
            "intrinsic_normed_cond": intrinsic_normed_cond.to(device).view(
                1, 1, 3, 3),
        }
        # The camera helpers build float32 matrices and the parameters here are
        # float16. SF3D never notices because it runs inside torch.autocast;
        # this path does not, so the batch is cast to match the weights rather
        # than relying on a context manager to paper over it.
        weight_dtype = next(model.parameters()).dtype
        for key, value in batch.items():
            if torch.is_tensor(value) and value.is_floating_point():
                batch[key] = value.to(weight_dtype)
        scene_codes, _ = model.get_scene_codes(batch)
        meshes = model.triplane_to_meshes(scene_codes)

    raw = meshes[0]
    seconds = time.time() - started
    vertices = raw.v_pos.detach().float().cpu().numpy()
    faces = raw.t_pos_idx.detach().cpu().numpy()
    if len(vertices) == 0:
        raise Sf3dError("SF3D returned an empty mesh for this image")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # The two rotations SF3D applies on export, which bypassing generate_mesh
    # skips. Without them the figure lies on its side, and a glTF consumer --
    # which assumes Y-up -- stands it up wrong again on import.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(
        np.radians(-90), [1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(
        np.radians(90), [0, 1, 0]))
    # Marching tetrahedra here comes out wound inwards: trimesh reports a
    # negative volume and a renderer with backface culling shows the inside.
    flipped = False
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()
        flipped = True

    low, high = mesh.bounds
    degenerate = int(sum(1 for a, b, c in faces
                         if a == b or b == c or a == c))
    # Mean wall thickness, and the same as a fraction of the mesh's own size.
    #
    # ``2 * volume / area`` has units of length and is the mean thickness of the
    # material: a sphere of radius r gives 2r/3, a flat plate of thickness t
    # gives t. It is the one number that tells a solid from a surface
    # impersonating one, and it needs nothing but the two quantities already
    # computed. The obvious alternatives both fail on this project's own meshes:
    # bounding-box thickness scores a curved shell as thick because the shell
    # curves through the box, and convex-hull solidity scores the gazelle's legs
    # below every head.
    longest = float(max(high - low)) or 1.0
    wall = (2.0 * mesh.volume / mesh.area) if (mesh.is_watertight
                                               and mesh.area) else None
    metrics = {
        "seconds": round(seconds, 2),
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number),
        "volume": round(float(mesh.volume), 6) if mesh.is_watertight else None,
        "extent": [round(float(v), 4) for v in (high - low)],
        "bbox_min": [round(float(v), 4) for v in low],
        "bbox_max": [round(float(v), 4) for v in high],
        "degenerate_faces": degenerate,
        "inverted_winding": flipped,
        "surface_area": round(float(mesh.area), 6),
        "wall_thickness": round(wall, 6) if wall is not None else None,
        "wall_ratio": round(wall / longest, 4) if wall is not None else None,
    }
    if str(next(model.parameters()).device) != "cpu":
        metrics["vram_peak_mb"] = int(
            __import__("torch").cuda.max_memory_allocated() / (1024 * 1024))
    return mesh, metrics
