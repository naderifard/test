"""Wan2.1-VACE-1.3B on an 8 GB laptop GPU with 15.6 GB of RAM.

VACE is a video model that follows a control video (depth, pose, or known
frames to keep) and a reference image. The pipeline uses it twice: to paint a
character's texture from a turntable of its own shape (``character_paint``),
and later to repaint shots.

Two things make it fit this machine, neither of which Wan's own
``generate.py`` does:

* **The text encoder never shares memory with the video model.** umT5-XXL is
  10.6 GB in bf16, the video model 6.7 GB in fp32 on disk -- together more
  than the machine has. Prompts are encoded first, in a separate call, the
  embeddings cached next to the model, and the encoder freed. ``WanVace`` is
  then built with a stand-in encoder that returns the cached embeddings.
* **The video model runs in bf16** (4.3 GB), not the fp32 it is stored in.

Attention runs through PyTorch's SDPA: the vendored ``wan/modules/
attention.py`` is patched to fall back to it, because flash-attn cannot be
built here.

Masks follow VACE: 1 = generate, 0 = keep the source pixels.
"""

from __future__ import annotations

import hashlib
import os
import sys

CHECKPOINT = "wan/Wan2.1-VACE-1.3B"
TASK = "vace-1.3B"
SIZE = (480, 832)              # (width, height) portrait; also (832, 480)
FRAMES = 81
DEFAULT_SHIFT = 16.0         # generate.py's default for vace


# Fewer, larger free blocks: without it the reserved memory ran 3 GB over what
# was allocated.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "garbage_collection_threshold:0.7,max_split_size_mb:256")


def _paths():
    from pipeline import envs
    code = os.path.join(envs.VENDOR_ROOT, "Wan2.1")
    ckpt = os.path.join(envs.MODEL_ROOT, *CHECKPOINT.split("/"))
    if code not in sys.path:
        sys.path.insert(0, code)
    return code, ckpt


def available():
    """Whether the code and the checkpoint are both on this machine."""
    code, ckpt = _paths()
    return (os.path.isdir(os.path.join(code, "wan"))
            and os.path.isfile(os.path.join(ckpt, "Wan2.1_VAE.pth")))


def _config():
    _paths()
    from wan.configs import WAN_CONFIGS
    return WAN_CONFIGS[TASK]


def _stream_blocks(model, device):
    """Each block's weights on ``device`` only while it runs.

    The CPU tensors are kept and put back afterwards, so they stay memory-
    mapped from the checkpoint instead of being copied into RAM. (accelerate's
    cpu_offload leaves weights on the meta device between calls, and umT5's
    relative position embedding builds its bucket indices on its weight's
    device -- meta -- so it failed.)"""
    import torch

    def pre(module, _args):
        module._cpu = {k: t.data for k, t in
                       list(module.named_parameters()) +
                       list(module.named_buffers())}
        for k, t in list(module.named_parameters()) + list(
                module.named_buffers()):
            t.data = t.data.to(device, non_blocking=True)

    def post(module, _args, _out):
        for k, t in list(module.named_parameters()) + list(
                module.named_buffers()):
            t.data = module._cpu[k]
        del module._cpu

    parts = [m for name, m in model.named_children() if name != "blocks"]
    parts += list(model.blocks)
    for part in parts:
        part.register_forward_pre_hook(pre)
        part.register_forward_hook(post)
    torch.cuda.empty_cache()


def encode_prompts(prompts):
    """umT5 embeddings for each prompt, cached on disk by prompt text."""
    import torch
    _code, ckpt = _paths()
    cache_dir = os.path.join(ckpt, "_prompt_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cfg = _config()
    out, missing = {}, []
    for text in prompts:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
        path = os.path.join(cache_dir, key + ".pt")
        if os.path.exists(path):
            out[text] = torch.load(path, weights_only=True)
        else:
            missing.append((text, path))
    if missing:
        # Wan's T5EncoderModel builds umT5-XXL in fp32 with random weights
        # first -- 23 GB, more than this machine has, so it paged for half an
        # hour -- and only then loads the real ones. Built on the meta device
        # and given the bf16 checkpoint memory-mapped, it takes 10.6 GB.
        from wan.modules.t5 import HuggingfaceTokenizer, umt5_xxl
        model = umt5_xxl(encoder_only=True, return_tokenizer=False,
                         dtype=cfg.t5_dtype, device="meta")
        state = torch.load(os.path.join(ckpt, cfg.t5_checkpoint),
                           map_location="cpu", mmap=True, weights_only=True)
        model.load_state_dict(state, assign=True)
        model.eval().requires_grad_(False)
        # bf16 on this CPU is emulated: two prompts took 89 minutes. Each
        # block is instead moved to the GPU for its own forward pass and
        # back, so the whole encoder never has to fit in 8 GB and the
        # arithmetic runs where bf16 is native.
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            _stream_blocks(model, device)
        tokenizer = HuggingfaceTokenizer(
            name=os.path.join(ckpt, cfg.t5_tokenizer), seq_len=cfg.text_len,
            clean="whitespace")
        with torch.no_grad():
            for text, path in missing:
                ids, mask = tokenizer([text], return_mask=True,
                                      add_special_tokens=True)
                ids, mask = ids.to(device), mask.to(device)
                length = int(mask.gt(0).sum())
                embedding = model(ids, mask)[0][:length]
                embedding = embedding.to(torch.bfloat16).cpu().contiguous()
                torch.save(embedding, path)
                out[text] = embedding
        del model, state
        import gc
        gc.collect()
    return out


class _CachedEncoder:
    """Stands in for ``T5EncoderModel`` inside ``WanVace``."""

    def __init__(self, table):
        self.table = table

        class _Model:
            def to(self, *a, **k):
                return self

            def cpu(self):
                return self
        self.model = _Model()

    def __call__(self, texts, device):
        return [self.table[t].to(device) for t in texts]


_VACE = None


def load(table):
    """``WanVace`` with the cached-embedding encoder and a bf16 model."""
    global _VACE
    import torch
    _code, ckpt = _paths()
    import wan.vace as vace_module
    original = vace_module.T5EncoderModel
    vace_module.T5EncoderModel = lambda **kwargs: _CachedEncoder(table)
    try:
        cfg = _config()
        _VACE = vace_module.WanVace(config=cfg, checkpoint_dir=ckpt,
                                    device_id=0, t5_cpu=True)
    finally:
        vace_module.T5EncoderModel = original
    _VACE.model.to(torch.bfloat16)
    _VACE.text_encoder.table = table
    # Measured at 17 frames: encoding the input video peaked at 10.6 GB with
    # the 4.6 GB model sitting on the GPU unused, and the spill into shared
    # system memory made one step take minutes. The video encoder and decoder
    # now run with the model moved off the GPU and the cache emptied either
    # side; the sampling loop moves the model back itself.
    # The video autoencoder in bf16 too: its wrapper autocasts to its own
    # dtype, so only changing that dtype (and the weights) takes effect.
    _VACE.vae.dtype = torch.bfloat16
    _VACE.vae.model.to(torch.bfloat16)
    _tile_vae(_VACE.vae)
    for name in ("vace_encode_frames", "vace_encode_masks", "decode_latent"):
        setattr(_VACE, name, _without_model(_VACE, getattr(_VACE, name)))
    _VACE.model.cpu()
    torch.cuda.empty_cache()
    return _VACE


TILE_PX = 320          # pixel tile edge for the video autoencoder
OVERLAP_PX = 64        # blended overlap between tiles
LATENT_STRIDE = 8


def _ramps(length, start, size, total, overlap):
    """Blend weights along one axis for a tile: rising over the overlap it
    shares with the tile before, falling over the one it shares with the
    tile after, 1 where it is alone."""
    import torch
    w = torch.ones(size)
    if start > 0:
        n = min(overlap, size)
        w[:n] = torch.linspace(0, 1, n + 2)[1:-1]
    if start + size < total:
        n = min(overlap, size)
        w[-n:] = torch.minimum(w[-n:], torch.linspace(1, 0, n + 2)[1:-1])
    return w


def _starts(total, tile, overlap):
    if total <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, total - tile, step)) + [total - tile]
    return sorted(set(starts))


def _tile_vae(vae):
    """Encode and decode in overlapping spatial tiles, blended.

    The autoencoder's activations scale with frame area: a whole 480x832
    frame peaked at 6-7 GB and spilled. Tiles of ``TILE_PX`` with
    ``OVERLAP_PX`` blended between them bring it to about a quarter.
    Time is untouched (each tile runs the whole clip), so the causal
    temporal cache inside the model still sees every frame in order.
    """
    import torch
    model, scale = vae.model, vae.scale

    def encode(videos):
        out = []
        with torch.autocast("cuda", dtype=vae.dtype):
            for u in videos:
                _c, f, h, w = u.shape
                result, weight = None, None
                for y in _starts(h, TILE_PX, OVERLAP_PX):
                    for x in _starts(w, TILE_PX, OVERLAP_PX):
                        th, tw = min(TILE_PX, h), min(TILE_PX, w)
                        z = model.encode(u[None, :, :, y:y + th, x:x + tw],
                                         scale).float()[0]
                        ly, lx = y // LATENT_STRIDE, x // LATENT_STRIDE
                        lh, lw = z.shape[-2:]
                        wy = _ramps(0, ly, lh, h // LATENT_STRIDE,
                                    OVERLAP_PX // LATENT_STRIDE)
                        wx = _ramps(0, lx, lw, w // LATENT_STRIDE,
                                    OVERLAP_PX // LATENT_STRIDE)
                        k = (wy[:, None] * wx[None, :]).to(z.device)
                        if result is None:
                            result = torch.zeros(
                                (z.shape[0], z.shape[1], h // LATENT_STRIDE,
                                 w // LATENT_STRIDE), device=z.device)
                            weight = torch.zeros(result.shape[-2:],
                                                 device=z.device)
                        result[..., ly:ly + lh, lx:lx + lw] += z * k
                        weight[ly:ly + lh, lx:lx + lw] += k
                        del z
                        torch.cuda.empty_cache()
                out.append(result / weight.clamp_min(1e-6))
        return out

    def decode(zs):
        out = []
        lt, lo = TILE_PX // LATENT_STRIDE, OVERLAP_PX // LATENT_STRIDE
        with torch.autocast("cuda", dtype=vae.dtype):
            for z in zs:
                _c, f, h, w = z.shape
                result, weight = None, None
                for y in _starts(h, lt, lo):
                    for x in _starts(w, lt, lo):
                        th, tw = min(lt, h), min(lt, w)
                        px = model.decode(z[None, :, :, y:y + th, x:x + tw],
                                          scale).float().clamp_(-1, 1)[0]
                        py, pxx = y * LATENT_STRIDE, x * LATENT_STRIDE
                        ph, pw = px.shape[-2:]
                        H, W = h * LATENT_STRIDE, w * LATENT_STRIDE
                        wy = _ramps(0, py, ph, H, OVERLAP_PX)
                        wx = _ramps(0, pxx, pw, W, OVERLAP_PX)
                        k = (wy[:, None] * wx[None, :]).to(px.device)
                        if result is None:
                            result = torch.zeros((px.shape[0], px.shape[1],
                                                  H, W), device=px.device)
                            weight = torch.zeros((H, W), device=px.device)
                        result[..., py:py + ph, pxx:pxx + pw] += px * k
                        weight[py:py + ph, pxx:pxx + pw] += k
                        del px
                        torch.cuda.empty_cache()
                out.append(result / weight.clamp_min(1e-6))
        return out

    vae.encode = encode
    vae.decode = decode


def _without_model(vace, fn):
    import torch

    def run(*args, **kwargs):
        vace.model.cpu()
        torch.cuda.empty_cache()
        # bf16 halves the video encoder's activations: alone in fp32 it still
        # peaked at 6.6 GB, which with the desktop's share of the card
        # spilled and took five minutes for 17 frames.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = fn(*args, **kwargs)
        if isinstance(out, (list, tuple)):
            out = type(out)(o.float() if torch.is_tensor(o) else o
                            for o in out)
        elif torch.is_tensor(out):
            out = out.float()
        torch.cuda.empty_cache()
        return out
    return run


def negative_prompt():
    return _config().sample_neg_prompt


REFINE_SHIFT = 3.0


def refine(prompt, init, control, masks, ref_images, strength=0.55,
           steps=40, guide=5.0, seed=42, context_scale=1.0, shift=None):
    """Repaint a rendered video rather than generate one: the render's own
    latents, noised to ``strength`` (0 keeps it, 1 is a fresh generation),
    are denoised under VACE's control video and references.

    ``init``: the render (3, F, H, W) in [-1, 1]; ``control``: the control
    video (depth, near bright) of the same shape; ``masks``: (1, F, H, W),
    1 = control; ``ref_images``: list of (3, 1, H, W) or None. A fresh
    generation from depth alone keeps the layout and loses the faces and
    costumes; starting from the render keeps who is who and where, and the
    model adds back what a render lacks -- light, skin, cloth, fur.
    """
    import math

    import torch
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    vace = _VACE
    device = vace.device
    table = vace.text_encoder.table
    neg = negative_prompt()
    context = [table[prompt].to(device)]
    context_null = [table[neg].to(device)]
    refs = [[r.to(device) for r in ref_images]] if ref_images else [None]
    n_ref = len(ref_images) if ref_images else 0

    z0 = vace.vace_encode_frames([control.to(device)], refs,
                                 masks=[masks.to(device)])
    m0 = vace.vace_encode_masks([masks.to(device)], refs)
    z = vace.vace_latent(z0, m0)
    encode = _without_model(vace, lambda v: vace.vae.encode(v))
    render = encode([init.to(device)])[0]                # (16, T, h, w)
    print("refine: encoded", tuple(render.shape), flush=True)
    target = list(z0[0].shape)
    target[0] //= 2
    start = torch.cat([z0[0][:target[0], :n_ref], render], dim=1) \
        if n_ref else render
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    noise = torch.randn(*target, dtype=torch.float32, device=device,
                        generator=generator)
    seq_len = math.ceil((target[2] * target[3])
                        / (vace.patch_size[1] * vace.patch_size[2])
                        * target[1] / vace.sp_size) * vace.sp_size

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=vace.num_train_timesteps, shift=1,
        use_dynamic_shifting=False)
    scheduler.set_timesteps(steps, device=device,
                            shift=shift if shift is not None else REFINE_SHIFT)
    # ``strength`` is the noise level the render is taken to, not a share of
    # the steps: with a shifted schedule the noise stays high for most steps,
    # and "skip 60% of the steps" still started at 0.91 -- a fresh
    # generation that lost the throne and Keyumars's tunic.
    sigmas = [float(s) for s in scheduler.sigmas[:steps]]
    first = next((i for i, s in enumerate(sigmas) if s <= strength),
                 steps - 1)
    timesteps = scheduler.timesteps[first:]
    sigma = sigmas[first]
    print(f"refine: from noise {sigma:.3f}, {steps - first} of {steps} "
          "steps", flush=True)
    latents = [(1.0 - sigma) * start.float() + sigma * noise]
    arg_c = {"context": context, "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}
    from tqdm import tqdm
    with torch.autocast("cuda", dtype=vace.param_dtype), torch.no_grad():
        vace.model.to(device)
        for t in tqdm(timesteps):
            timestep = torch.stack([t])
            cond = vace.model(latents, t=timestep, vace_context=z,
                              vace_context_scale=context_scale, **arg_c)[0]
            uncond = vace.model(latents, t=timestep, vace_context=z,
                                vace_context_scale=context_scale,
                                **arg_null)[0]
            pred = uncond + guide * (cond - uncond)
            latents = [scheduler.step(pred.unsqueeze(0), t,
                                      latents[0].unsqueeze(0),
                                      return_dict=False,
                                      generator=generator)[0].squeeze(0)]
        vace.model.cpu()
        torch.cuda.empty_cache()
    video = vace.decode_latent(latents, refs)[0]
    torch.cuda.empty_cache()
    return video


def generate(prompt, frames, masks, ref_images, size=SIZE, steps=25,
             guide=5.0, seed=42, context_scale=1.0, shift=None):
    """Generate a video. ``frames``: float tensor (3, F, H, W) in [-1, 1];
    ``masks``: (1, F, H, W) in {0, 1}, 1 = generate; ``ref_images``: list of
    (3, 1, H, W) tensors in [-1, 1] or None. Returns (3, F, H, W) in [-1, 1].
    """
    import torch
    vace = _VACE
    device = vace.device
    table = vace.text_encoder.table
    neg = negative_prompt()
    if prompt not in table or neg not in table:
        raise KeyError("encode_prompts() the prompt and the negative prompt "
                       "before load()")
    video = vace.generate(
        prompt, [frames.to(device)], [masks.to(device)],
        [[r.to(device) for r in ref_images]] if ref_images else [None],
        size=size, frame_num=frames.shape[1], context_scale=context_scale,
        shift=shift if shift is not None else DEFAULT_SHIFT,
        sample_solver="unipc", sampling_steps=steps, guide_scale=guide,
        n_prompt=neg, seed=seed, offload_model=True)
    torch.cuda.empty_cache()
    return video
