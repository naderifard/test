"""Body landmarks from an image or a video: MediaPipe BlazePose, via ONNX.

Two stages want the same thing from a picture of a person -- where the joints
are -- for two different reasons:

* ``c_asset`` needs the joints of the *reference photo*, because the photo is
  exactly aligned with the SF3D mesh built from it (the conditioning camera is
  known), so a joint found in the image can be pushed through the mesh to find
  where the bone belongs inside it. That is how the skeleton is placed.
* ``e_motion`` needs the joints of *every frame of a user's video*, in metric 3D,
  to retarget a real performance onto the rig.

**Why ONNX through OpenCV rather than the ``mediapipe`` package.** The package's
task files (``pose_landmarker_*.task``) are served from a Google bucket that
answers this machine with ``AccessDenied: not available in your location``. The
OpenCV model zoo publishes the same BlazePose network (person detector plus
landmark model, March 2023) as ONNX on Hugging Face, with the pre- and
post-processing as plain Python. Those two ``mp_*.py`` files are vendored in the
model directory next to the weights, unmodified, and imported from there.

The landmark model returns 39 points; the first 33 are BlazePose's documented
topology (``LANDMARKS`` below), the rest are auxiliary. Image landmarks are in
pixels of the input image; world landmarks are metres in a hip-centred frame
with +y *down* the body, which is the frame the retargeter reads.

Runs in any interpreter with ``opencv-python`` and ``numpy``: the ``pose`` and
``ml`` environments both qualify.
"""

from __future__ import annotations

import os
import sys

LANDMARKS = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
)
INDEX = {name: i for i, name in enumerate(LANDMARKS)}

MODEL_SUBDIR = "mediapipe"
DETECTOR = "person_detection_mediapipe_2023mar.onnx"
LANDMARKER = "pose_estimation_mediapipe_2023mar.onnx"


class PoseError(RuntimeError):
    pass


class Estimator:
    """One detector and one landmarker, reused for every image or frame."""

    def __init__(self, model_dir=None, score_threshold=0.5):
        if model_dir is None:
            from pipeline import envs
            model_dir = envs.model_dir(MODEL_SUBDIR)
        for name in (DETECTOR, LANDMARKER, "mp_pose.py", "mp_persondet.py"):
            if not os.path.exists(os.path.join(model_dir, name)):
                raise PoseError(f"pose model file missing: "
                                f"{os.path.join(model_dir, name)}")
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from mp_persondet import MPPersonDet
        from mp_pose import MPPose
        self.detector = MPPersonDet(
            modelPath=os.path.join(model_dir, DETECTOR), nmsThreshold=0.3,
            scoreThreshold=score_threshold, topK=5000)
        self.landmarker = MPPose(
            modelPath=os.path.join(model_dir, LANDMARKER), confThreshold=0.5)

    def detect(self, bgr, figure_box=None):
        """The most confident person in one BGR frame, or None.

        Returns ``{"image": [[x, y, visibility], ...33], "world": [[x, y, z],
        ...33], "confidence": c, "seeded": bool}``; image points in pixels.

        ``figure_box`` (x1, y1, x2, y2) is the silhouette of a figure already
        known to be there. BlazePose's person detector keys on a human face, so
        a wolf-headed div is never *detected* -- but the landmark model only
        needs a hip centre and a body extent to crop by, and a silhouette gives
        both. The box is used when the detector finds nobody, and the result is
        marked ``seeded`` so a caller can hold it to a higher bar.
        """
        import numpy as np

        people = self.detector.infer(bgr)
        seeded = False
        if (people is None or len(people) == 0) and figure_box is not None:
            x1, y1, x2, y2 = (float(v) for v in figure_box)
            height = y2 - y1
            cx = (x1 + x2) / 2.0
            hip = (cx, y1 + 0.53 * height)
            top = (cx, hip[1] - 0.58 * height)
            people = np.array([[x1, y1, x2, y2, hip[0], hip[1], top[0], top[1],
                                cx, y1 + 0.25 * height, cx, y1, 1.0]],
                              dtype=np.float32)
            seeded = True
        if people is None or len(people) == 0:
            return None
        best = None
        for person in people:
            result = self.landmarker.infer(bgr, person)
            if result is None:
                continue
            _bbox, image_pts, world_pts, _mask, _heat, conf = result
            if best is None or conf > best["confidence"]:
                best = {
                    "image": [[float(p[0]), float(p[1]), float(p[4])
                               if len(p) > 4 else 1.0]
                              for p in image_pts[:33]],
                    "world": [[float(v) for v in p[:3]]
                              for p in world_pts[:33]],
                    "confidence": float(conf),
                    "seeded": seeded,
                }
        return best


def detect_image(path, estimator=None):
    """Landmarks for one image file, or None when no person is found."""
    import cv2
    import numpy as np

    data = np.fromfile(path, dtype=np.uint8)       # survives non-ASCII paths
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise PoseError(f"could not read image {path}")
    return (estimator or Estimator()).detect(bgr)


def named(points):
    """``[[x, y, ...], ...]`` as ``{"left_wrist": [x, y, ...], ...}``."""
    return {name: points[i] for i, name in enumerate(LANDMARKS)
            if i < len(points)}
