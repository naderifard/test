"""A speaking face gets a real mouth: MuseTalk over the rendered frames.

The 3D mouth is shape keys on a photo-textured shell (``photo_body.visemes``):
the chin moves, the lips never part, there is nothing behind them. MuseTalk
(v1.5) is a single-step latent inpainting network conditioned on Whisper
features of the audio: given a 256x256 face crop with its lower half masked,
plus the same crop unmasked as a reference, it paints the lower face saying
this frame's sound -- lips parting, teeth, tongue, jaw. So after the shot is
rendered, each speaker's face is repainted frame by frame from the converted
line, and the upper face, the body and the set stay exactly as rendered.

What is ours rather than MuseTalk's:

* **The face box comes from the scene, not from detectors** (``face_track``):
  the crop MuseTalk was trained on is built from 68 facial landmarks, and it is
  rebuilt here from the six the character was made with -- the jaw's span from
  the eye distance, the chin from the nose-to-mouth line, the top as far above
  landmark 29 (the nose bridge) as the chin is below it, as MuseTalk's
  ``get_landmark_and_bbox`` does.
* **Only where it can work.** Frames where the face is smaller than
  ``MIN_FACE_PX`` tall, turned more than ``MAX_TURN_DEG`` from the camera or
  off screen keep the rendered mouth; so do frames outside the speaker's line.
* **No pops.** Each painted run fades in and out over ``FADE_FRAMES``.
* **One speaker's voice per speaker.** The audio MuseTalk hears is that
  speaker's own lines placed on the shot's timeline, silence elsewhere, so two
  characters in a dialogue do not both mouth every line.
"""

from __future__ import annotations

import os
import sys

MIN_FACE_PX = 44
MAX_TURN_DEG = 60.0
FADE_FRAMES = 3
PAD_FRAMES = 2              # painted either side of a line, for the lead-in
BATCH = 8
SAMPLE_RATE = 16000

VENDOR = "MuseTalk"


def face_box(points, width, height):
    """MuseTalk's crop box, in pixels, from six landmarks (see module doc)."""
    def px(name):
        u, v = points[name]
        return (u * width, v * height)
    le, re_, nose, mouth = px("left_eye"), px("right_eye"), px("nose"), \
        px("mouth")
    eyes = ((le[0] + re_[0]) / 2.0, (le[1] + re_[1]) / 2.0)
    ipd = max(1.0, ((re_[0] - le[0]) ** 2 + (re_[1] - le[1]) ** 2) ** 0.5)
    chin = (mouth[0] + 2.0 * (mouth[0] - nose[0]),
            mouth[1] + 2.0 * (mouth[1] - nose[1]))
    # Landmark 29 (the mask line) from the nose and mouth alone: a
    # reconstructed face's eye landmarks can sit on the brows (Soroush's did),
    # which lifted the line to under the eyes and had MuseTalk repaint half
    # the nose. Nose tip to mouth is the steadier ruler.
    bridge = (nose[0] - 0.65 * (mouth[0] - nose[0]),
              nose[1] - 0.65 * (mouth[1] - nose[1]))
    y2 = chin[1]
    y1 = bridge[1] - (y2 - bridge[1])
    cx = (eyes[0] + mouth[0]) / 2.0
    half = 1.15 * ipd
    # MuseTalk v1.5 extends the box below the chin by a margin (10 px on its
    # training resolution); scaled to the box here.
    y2 += 0.05 * (y2 - y1)
    # Not clamped to the frame: a face at the frame's edge keeps the crop
    # geometry MuseTalk was trained on, and the frame is padded instead
    # (``padded``). Clamping the top of a head cut by the frame edge moved the
    # mask line from the nose down to the upper lip, and the mouth never
    # opened.
    return [int(round(cx - half)), int(round(y1)), int(round(cx + half)),
            int(round(y2))]


def padded(image, box):
    """The frame padded so ``box`` fits inside it: ``(image, box, pad)``."""
    import cv2
    h, w = image.shape[:2]
    pad = max(0, -box[0], -box[1], box[2] - w, box[3] - h)
    if pad == 0:
        return image, list(box), 0
    pad += 8
    big = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return big, [v + pad for v in box], pad


def unpadded(image, pad):
    if pad == 0:
        return image
    return image[pad:-pad, pad:-pad]


def speaker_audio(tracks, shot, fps, path):
    """One speaker's lines on the shot's timeline, silence elsewhere, 16 kHz.

    Placement is ``render_shot._encode``'s: shot frame F plays file time
    ``(F - start_frame + offset) / fps`` for the line's duration, where
    ``offset`` is the take's lead-in rounded to frames.
    """
    import librosa
    import numpy as np
    import soundfile
    frames = shot["frame_end"] - shot["frame_start"] + 1
    out = np.zeros(int(round(frames / float(fps) * SAMPLE_RATE)) + 1,
                   dtype=np.float32)
    windows = []
    for track in tracks:
        data, _ = librosa.load(track["abs_path"], sr=SAMPLE_RATE, mono=True)
        offset = int(round(track["trim_start_s"] * fps))
        first = track["start_frame"] - shot["frame_start"]
        count = max(1, int(round(track["duration_s"] * fps)))
        src0 = int(round(offset / float(fps) * SAMPLE_RATE))
        dst0 = int(round(first / float(fps) * SAMPLE_RATE))
        n = min(int(round(count / float(fps) * SAMPLE_RATE)),
                len(data) - src0, len(out) - dst0)
        if n > 0:
            out[dst0:dst0 + n] += data[src0:src0 + n]
        windows.append((first, first + count))
    soundfile.write(path, out, SAMPLE_RATE)
    return windows


class Painter:
    """MuseTalk's VAE, UNet, Whisper and face parser, loaded once."""

    def __init__(self):
        from pipeline import envs
        root = os.path.join(envs.VENDOR_ROOT, VENDOR)
        if root not in sys.path:
            sys.path.insert(0, root)
        self.root = root
        import torch
        from transformers import WhisperModel
        cwd = os.getcwd()
        os.chdir(root)                         # MuseTalk's paths are relative
        try:
            from musetalk.utils.audio_processor import AudioProcessor
            from musetalk.utils.face_parsing import FaceParsing
            from musetalk.utils.utils import load_all_model
            self.device = torch.device("cuda")
            self.vae, self.unet, self.pe = load_all_model(device=self.device)
            self.pe = self.pe.half().to(self.device)
            self.vae.vae = self.vae.vae.half().to(self.device)
            self.unet.model = self.unet.model.half().to(self.device)
            self.audio = AudioProcessor(
                feature_extractor_path=os.path.join("models", "whisper"))
            self.whisper = WhisperModel.from_pretrained(
                os.path.join("models", "whisper")).to(
                    device=self.device, dtype=torch.float16).eval()
            self.whisper.requires_grad_(False)
            self.parser = FaceParsing(left_cheek_width=90,
                                      right_cheek_width=90)
        finally:
            os.chdir(cwd)
        self.torch = torch

    def chunks(self, wav, fps):
        features, length = self.audio.get_audio_feature(
            wav, weight_dtype=self.torch.float16)
        return self.audio.get_whisper_chunk(
            features, self.device, self.torch.float16, self.whisper, length,
            fps=fps, audio_padding_length_left=2,
            audio_padding_length_right=2)

    def paint(self, frames, boxes, chunks):
        """Repainted crops for ``frames`` (BGR arrays) at ``boxes``, in order,
        using ``chunks`` (one Whisper chunk per frame)."""
        import cv2
        import numpy as np
        torch = self.torch
        timestep = torch.tensor([0], device=self.device)
        out = []
        for start in range(0, len(frames), BATCH):
            latents, audio = [], []
            for frame, box, chunk in zip(frames[start:start + BATCH],
                                         boxes[start:start + BATCH],
                                         chunks[start:start + BATCH]):
                x1, y1, x2, y2 = box
                crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256),
                                  interpolation=cv2.INTER_LANCZOS4)
                latents.append(self.vae.get_latents_for_unet(crop))
                audio.append(chunk)
            with torch.no_grad():
                feats = self.pe(torch.stack(audio).to(self.device))
                lat = torch.cat(latents, 0).to(self.device,
                                               dtype=self.unet.model.dtype)
                pred = self.unet.model(lat, timestep,
                                       encoder_hidden_states=feats).sample
                recon = self.vae.decode_latents(pred)
            out.extend(np.asarray(r) for r in recon)
        return out

    def blend(self, frame, face, box, mode="jaw"):
        import cv2
        from musetalk.utils.blending import get_image
        x1, y1, x2, y2 = box
        face = cv2.resize(face.astype("uint8"), (x2 - x1, y2 - y1))
        face = match_colour(face, frame[y1:y2, x1:x2])
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            return get_image(frame, face, [x1, y1, x2, y2], mode=mode,
                             fp=self.parser)
        finally:
            os.chdir(cwd)


def match_colour(painted, original):
    """The painted face's colours matched to the render's, channel by channel.

    MuseTalk's VAE returns a painted (non-photographic) face a shade off in
    colour and contrast, and the blend's edge then shows as a band across the
    cheeks. Matching mean and spread on the upper half -- which MuseTalk was
    told to keep -- corrects the whole crop without touching the mouth shape.
    """
    import numpy as np
    h = painted.shape[0] // 2
    out = painted.astype(np.float32)
    ref = original.astype(np.float32)
    for c in range(3):
        pm, ps = out[:h, :, c].mean(), out[:h, :, c].std() + 1e-3
        rm, rs = ref[:h, :, c].mean(), ref[:h, :, c].std() + 1e-3
        out[:, :, c] = (out[:, :, c] - pm) * (rs / ps) + rm
    return np.clip(out, 0, 255).astype(np.uint8)


def usable(row, box, height_px):
    """Whether this frame's face can be repainted, and why not."""
    if not row.get("visible"):
        return False, "off screen"
    if row.get("turn_deg", 180.0) > MAX_TURN_DEG:
        return False, "turned away"
    if box[3] - box[1] < MIN_FACE_PX or box[2] - box[0] < MIN_FACE_PX * 0.6:
        return False, "too small"
    # A face mostly outside the frame has too little left to repaint.
    inside = max(0, min(box[3], height_px) - max(box[1], 0))
    if inside < 0.6 * (box[3] - box[1]):
        return False, "cut by the frame"
    return True, ""


# Measured on the test film: good repaints kept 0.83-0.96 of the rendered
# mouth's detail (median over a shot); the smeared ones 0.30 (Siamak's
# upturned, bearded face), 0.59 (the div's muzzle), 0.66 (Keyumars's vow).
DETAIL_KEEP = 0.70


def detail(image, box):
    """Sharpness of the lower half of the face box (the mouth and chin): the
    variance of the Laplacian, the usual focus measure."""
    import cv2
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h, w = image.shape[:2]
    y_mid = (y1 + y2) // 2
    x1, x2 = max(0, x1), min(w, x2)
    y_mid, y2 = max(0, y_mid), min(h, y2)
    if x2 - x1 < 4 or y2 - y_mid < 4:
        return 0.0
    grey = cv2.cvtColor(image[y_mid:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def kept_detail(before, after):
    """Median share of the rendered mouth's detail the repaint kept, over a
    run's frames (``before``/``after``: detail values). MuseTalk is trained
    on real, level, frontal faces; on a bearded face tipped back it wiped the
    beard and chin into a flat smear, and a smear is worse than a mouth that
    does not quite match the words."""
    ratios = sorted(a / b for a, b in zip(after, before) if b > 1e-6)
    return ratios[len(ratios) // 2] if ratios else 1.0


def runs(flags):
    """Contiguous True runs as ``(first, last)`` index pairs."""
    out, start = [], None
    for i, flag in enumerate(flags + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def fade(index, run):
    first, last = run
    edge = min(index - first, last - index) + 1
    return min(1.0, edge / float(FADE_FRAMES + 1))
