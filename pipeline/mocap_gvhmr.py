"""A recorded move becomes SMPL joints: GVHMR, video to world-grounded motion.

BlazePose (``motion_video``) reads 33 landmarks per frame from one view; it
jitters, under-reads depth (a kneel reads as half a kneel) and has no idea
where the floor is. GVHMR recovers a whole SMPL-X body per frame in a
gravity-aligned world, smooth over time and with the feet on a ground plane,
which is what a take needs. This module runs it in the ``motion`` env and
returns the 22 body joints in the rig's axes for ``motion_smpl`` to retarget.

It follows GVHMR's own demo (``tools/demo/demo.py``) step for step -- YOLO
track, ViTPose 2D keypoints, HMR2 image features, the GVHMR network -- minus
its renderer, which needs a compiled pytorch3d, and minus visual odometry: the
Moves tab asks for a still camera, and ``static_cam`` is what GVHMR itself
recommends for that. Its networks assume 30 fps, so the video is re-timed to
30 fps first rather than letting a 60 fps phone video play at half speed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

FPS = 30
VENDOR = "GVHMR"


def _root():
    from pipeline import envs
    root = os.path.join(envs.VENDOR_ROOT, VENDOR)
    if not os.path.isdir(root):
        raise RuntimeError(f"GVHMR is not on this machine: {root}")
    return root


def _retime(src, dst):
    """The video at exactly 30 fps, H.264, so frame i is time i/30."""
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-i", src,
                    "-vf", f"fps={FPS}", "-an", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "17", dst], check=True)


def _register_demo():
    """The three config groups the demo composes, and nothing else.

    GVHMR's ``register_store_gvhmr`` imports every training dataset, and the
    3DPW one imports the pytorch3d renderer, which this machine cannot build.
    Registration is a side effect of importing the module that defines each
    group, so importing just these is the same registration.
    """
    import hmr4d.model.gvhmr.gvhmr_pl_demo  # noqa: F401  model
    import hmr4d.model.gvhmr.utils.endecoder  # noqa: F401  endecoder
    import hmr4d.network.gvhmr.relative_transformer  # noqa: F401  network


def track(video, work_dir):
    """Rig-axis joint frames at ``FPS`` for the performer in ``video``."""
    import torch
    root = _root()
    if root not in sys.path:
        sys.path.insert(0, root)
    os.makedirs(work_dir, exist_ok=True)
    clip = os.path.abspath(os.path.join(work_dir, "input_30fps.mp4"))
    _retime(video, clip)

    cwd = os.getcwd()
    os.chdir(root)                    # checkpoint paths are repo-relative
    try:
        import hydra
        from hydra import compose, initialize_config_module

        from hmr4d.utils.geo.hmr_cam import estimate_K, get_bbx_xys_from_xyxy
        from hmr4d.utils.geo_transform import compute_cam_angvel
        from hmr4d.utils.net_utils import detach_to_cpu
        from hmr4d.utils.preproc import Extractor, Tracker, VitPoseExtractor
        from hmr4d.utils.smplx_utils import make_smplx
        from hmr4d.utils.video_io_utils import get_video_lwh

        length, width, height = get_video_lwh(clip)
        tracker = Tracker()
        box = tracker.get_one_track(clip).float()
        del tracker
        bbx_xys = get_bbx_xys_from_xyxy(box, base_enlarge=1.2).float()
        vitpose = VitPoseExtractor()
        kp2d = vitpose.extract(clip, bbx_xys)
        del vitpose
        torch.cuda.empty_cache()
        extractor = Extractor()
        features = extractor.extract_video_features(clip, bbx_xys)
        del extractor
        torch.cuda.empty_cache()

        with initialize_config_module(version_base="1.3",
                                      config_module="hmr4d.configs"):
            _register_demo()
            cfg = compose(config_name="demo", overrides=[
                "video_name=take", "static_cam=True", "use_dpvo=False",
                f"output_root={os.path.abspath(work_dir)}"])
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        data = {
            "length": torch.tensor(length),
            "bbx_xys": bbx_xys,
            "kp2d": kp2d,
            "K_fullimg": estimate_K(width, height).repeat(length, 1, 1),
            "cam_angvel": compute_cam_angvel(
                torch.eye(3).repeat(length, 1, 1)),
            "f_imgseq": features,
        }
        with torch.no_grad():
            pred = detach_to_cpu(model.predict(data, static_cam=True))
        del model
        torch.cuda.empty_cache()

        body = make_smplx("supermotion")
        with torch.no_grad():
            out = body(**pred["smpl_params_global"])
        joints = out.joints[:, :22].numpy()
    finally:
        os.chdir(cwd)

    from pipeline import motion_smpl
    frames = [motion_smpl.to_rig(f.tolist()) for f in joints]
    confidence = kp2d[..., 2].mean().item() if kp2d.ndim == 3 else None
    return frames, {"source_fps": FPS, "source_frames": len(frames),
                    "video_size": [int(width), int(height)],
                    "keypoint_confidence": (round(confidence, 3)
                                            if confidence is not None
                                            else None),
                    "source_skate_m_per_frame": round(
                        motion_smpl.skate(frames, FPS), 5)}
