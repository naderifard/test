"""A move's words become body motion: MoMask, text to SMPL joints.

The authored clip library (``motion_library``) is a handful of key poses with
smooth blends between them -- correct, and robotic. MoMask was trained on
HumanML3D, thousands of motion-capture takes described in English, so a line
like "raises his arm slowly in blessing" comes back with weight shift, timing
and follow-through that nobody authored. It runs in the ``motion`` env, whose
torch is the comfyui env's.

Three details:

* **Prompts are HumanML3D-shaped.** The training captions read "a person ...",
  so the move text gets that subject; the other characters' names mean nothing
  to the model and become "someone" (which way the character faces is the
  staging's job, not the motion's).
* **Several samples, the steadiest kept.** Generation is sampled and cheap (a
  second or two each), and an occasional sample slides its feet. Each prompt is
  sampled ``SAMPLES`` times and the one whose planted feet slide least wins.
* **Foot clean-up is MoMask's own.** Its joints-to-BVH converter runs an IK
  pass that pins contacts; its output joints are what this module returns.

Output joints are at MoMask's 20 fps, y-up, facing +z, floor at y=0 -- then
``motion_smpl.to_rig``'d, so callers get the rig's axes.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile

FPS = 20
MIN_S, MAX_S = 2.0, 9.6          # MoMask's HumanML3D range is 40-196 frames
SAMPLES = 3
SEED = 10107

VENDOR = "momask-codes"
CLIP_DIR = "clip"
VQ = "rvq_nq6_dc512_nc512_noshare_qdp0.2"
TRANS = "t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns"
RES = "tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw"


def prompt(text, names=()):
    """A move line as a HumanML3D caption: 'a person ...', names removed."""
    words = (text or "").strip().rstrip(".")
    for name in sorted({n for n in names if n}, key=len, reverse=True):
        pattern = r"\b(the\s+)?" + re.escape(name.replace("_", " ")) + r"\b"
        words = re.sub(pattern, "someone", words, flags=re.I)
        words = re.sub(r"\b(the\s+)?" + re.escape(name) + r"\b", "someone",
                       words, flags=re.I)
    words = re.sub(r"\b(the|a)\s+(div|demon|lion|king|prince|angel)\b",
                   "someone", words, flags=re.I)
    if not words:
        words = "stands still, breathing"
    if not re.match(r"(?i)(a|the)\s+(person|man|woman)\b", words):
        words = "a person " + words
    return words


# Metres of wandering that cost as much as a millimetre per frame of slide.
WANDER_WEIGHT = 0.01


def _wander(frames):
    """How far the pelvis strays from where it started, metres."""
    import math
    from pipeline import motion_smpl
    p0 = frames[0][motion_smpl.J["pelvis"]]
    return max(math.hypot(f[motion_smpl.J["pelvis"]][0] - p0[0],
                          f[motion_smpl.J["pelvis"]][1] - p0[1])
               for f in frames)


class Generator:
    """MoMask's three networks, loaded once for every prompt of a stage run."""

    def __init__(self, device="cuda"):
        from pipeline import envs
        root = os.path.join(envs.VENDOR_ROOT, VENDOR)
        self.root = root
        if root not in sys.path:
            sys.path.insert(0, root)
        import clip
        import numpy as np
        import torch
        clip_root = os.path.join(envs.MODEL_ROOT, CLIP_DIR)
        os.makedirs(clip_root, exist_ok=True)
        if not getattr(clip.load, "_rooted", False):
            original = clip.load

            def load(name, *args, **kwargs):
                kwargs.setdefault("download_root", clip_root)
                return original(name, *args, **kwargs)
            load._rooted = True
            clip.load = load

        cwd = os.getcwd()
        os.chdir(root)            # MoMask's opt.txt paths are repo-relative
        try:
            from gen_t2m import (load_res_model,
                                 load_trans_model, load_vq_model)
            from utils.get_opt import get_opt
            self.device = torch.device(device)
            ckpt = os.path.join("checkpoints", "t2m")
            model_opt = get_opt(os.path.join(ckpt, TRANS, "opt.txt"),
                                device=self.device)
            vq_opt = get_opt(os.path.join(ckpt, VQ, "opt.txt"),
                             device=self.device)
            vq_opt.dim_pose = 263
            self.vq, vq_opt = load_vq_model(vq_opt)
            model_opt.num_tokens = vq_opt.nb_code
            model_opt.num_quantizers = vq_opt.num_quantizers
            model_opt.code_dim = vq_opt.code_dim
            res_opt = get_opt(os.path.join(ckpt, RES, "opt.txt"),
                              device=self.device)

            class _Opt:
                pass
            opt = _Opt()
            opt.device = self.device
            opt.name = TRANS
            self.res = load_res_model(res_opt, vq_opt, opt)
            self.trans = load_trans_model(model_opt, opt, "latest.tar")
            self.mean = np.load(os.path.join(ckpt, VQ, "meta", "mean.npy"))
            self.std = np.load(os.path.join(ckpt, VQ, "meta", "std.npy"))
        finally:
            os.chdir(cwd)
        for net in (self.vq, self.res, self.trans):
            net.eval().to(self.device)
        self.torch = torch

    def _one(self, caption, frames):
        from utils.motion_process import recover_from_ric
        torch = self.torch
        tokens = torch.LongTensor([frames // 4]).to(self.device)
        with torch.no_grad():
            ids = self.trans.generate([caption], tokens, timesteps=18,
                                      cond_scale=4, temperature=1.0,
                                      topk_filter_thres=0.9, gsample=False)
            ids = self.res.generate(ids, [caption], tokens, temperature=1,
                                    cond_scale=5)
            motion = self.vq.forward_decoder(ids).detach().cpu().numpy()
        data = motion[0][: tokens.item() * 4] * self.std + self.mean
        return recover_from_ric(torch.from_numpy(data).float(), 22).numpy()

    def _foot_ik(self, joints):
        from visualization.joints2bvh import Joint2BVHConvertor
        converter = Joint2BVHConvertor()
        handle, path = tempfile.mkstemp(suffix=".bvh")
        os.close(handle)
        try:
            _, pinned = converter.convert(joints, filename=path,
                                          iterations=100)
        finally:
            os.remove(path)
        return pinned

    def generate(self, text, seconds, names=(), samples=SAMPLES, seed=SEED,
                 in_place=False, penalty=None):
        """Rig-axis joints at ``FPS`` for one move, and a report.

        ``in_place`` asks for a move on the spot and scores each sample's
        wandering alongside its foot slide; ``penalty(frames)`` adds any other
        cost to a sample's score.
        """
        from pipeline import motion_smpl
        caption = prompt(text, names)
        if in_place and "in place" not in caption:
            caption += ", staying in place"
        seconds = min(MAX_S, max(MIN_S, float(seconds)))
        frames = int(round(seconds * FPS / 4.0)) * 4
        cwd = os.getcwd()
        os.chdir(self.root)
        best = None
        try:
            for k in range(max(1, samples)):
                self.torch.manual_seed(seed + k)
                joints = self._foot_ik(self._one(caption, frames))
                rig = [motion_smpl.to_rig(f.tolist()) for f in joints]
                slide = motion_smpl.skate(rig, FPS)
                score = slide
                if in_place:
                    score += WANDER_WEIGHT * _wander(rig)
                if penalty is not None:
                    score += penalty(rig)
                if best is None or score < best[0]:
                    best = (score, rig, k, slide)
        finally:
            os.chdir(cwd)
        _, rig, k, slide = best
        return rig, {"caption": caption, "sample": k, "samples": samples,
                     "wander_m": round(_wander(rig), 3),
                     "source_fps": FPS, "source_frames": len(rig),
                     "source_skate_m_per_frame": round(slide, 5)}
