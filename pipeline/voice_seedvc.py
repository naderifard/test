"""Seed-VC for ``f_voice``: the user's recording in the character's voice.

Section 7-F chose Seed-VC because it is *conversion*: the user performs the
line, and the model changes whose voice it is while keeping what was said, how
it was acted and -- the part the pipeline depends on -- exactly when each sound
happens. The mouth curves are computed from the converted audio, the shot
length from its speech span, so a converter that moved the timing would
silently desynchronise the film.

This is ``inference.py``'s non-F0 path from the vendored checkout (the 22 kHz
speech model: DiT small, Whisper-small content encoder, CAM++ speaker encoder,
BigVGAN vocoder), loaded once per process and reused for every line. Three
changes from upstream, each for this pipeline:

* **No ``hf_utils``.** Upstream downloads into ``./checkpoints`` relative to
  the working directory. Here the weights come from the Hugging Face cache that
  ``envs.step_env`` points at the model drive, where they already are.
* **Exact length.** The mel front-end runs with ``center=False``, so the output
  is up to one analysis window (~35 ms) short of the input. ``f_voice`` fails a
  line whose duration moved by more than 10 ms, because the lip sync is keyed
  to the audio. The frames themselves keep their timing, so the output is
  padded (or trimmed) at the end to the input's exact sample count.
* **Level.** The output is scaled to the input's RMS, so a converted line sits
  in the mix where the performance put it.
"""

from __future__ import annotations

import os
import sys
import time

REPO = "Plachta/Seed-VC"
CHECKPOINT = "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"
CONFIG = "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
DIFFUSION_STEPS = 30
CFG_RATE = 0.7
REFERENCE_MAX_S = 25.0

_LOADED = None


def _vendor():
    """The Seed-VC checkout and the shims directory on sys.path.

    The shims satisfy Seed-VC's import of descript-audio-codec, which this
    model never calls (see vendor/shims/dac).
    """
    from pipeline import envs
    path = os.path.join(envs.VENDOR_ROOT, "seed-vc")
    if not os.path.isdir(path):
        raise RuntimeError(f"Seed-VC source is not at {path}")
    for entry in (os.path.join(envs.VENDOR_ROOT, "shims"), path):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return path


def load(device=None):
    """Everything the conversion needs, loaded once."""
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    _vendor()
    import torch
    import yaml
    from huggingface_hub import hf_hub_download
    from modules.audio import mel_spectrogram
    from modules.bigvgan import bigvgan
    from modules.campplus.DTDNN import CAMPPlus
    from modules.commons import build_model, load_checkpoint, recursive_munch
    from transformers import AutoFeatureExtractor, WhisperModel

    started = time.time()
    device = torch.device(device or ("cuda" if torch.cuda.is_available()
                                     else "cpu"))
    checkpoint = hf_hub_download(REPO, CHECKPOINT)
    with open(hf_hub_download(REPO, CONFIG), "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    params = recursive_munch(config["model_params"])
    params.dit_type = "DiT"
    model = build_model(params, stage="DiT")
    model, _, _, _ = load_checkpoint(model, None, checkpoint,
                                     load_only_params=True, ignore_modules=[],
                                     is_distributed=False)
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(torch.load(
        hf_hub_download("funasr/campplus", "campplus_cn_common.bin"),
        map_location="cpu"))
    campplus.eval().to(device)

    vocoder = bigvgan.BigVGAN.from_pretrained(params.vocoder.name,
                                              use_cuda_kernel=False)
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to(device)

    whisper_name = params.speech_tokenizer.name
    whisper = WhisperModel.from_pretrained(whisper_name,
                                           torch_dtype=torch.float16).to(device)
    del whisper.decoder
    extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

    spect = config["preprocess_params"]["spect_params"]
    sr = config["preprocess_params"]["sr"]
    mel_args = {
        "n_fft": spect["n_fft"], "win_size": spect["win_length"],
        "hop_size": spect["hop_length"], "num_mels": spect["n_mels"],
        "sampling_rate": sr, "fmin": spect.get("fmin", 0),
        "fmax": None if spect.get("fmax", "None") == "None" else 8000,
        "center": False,
    }

    def semantic(waves_16k):
        inputs = extractor([waves_16k.squeeze(0).cpu().numpy()],
                           return_tensors="pt", return_attention_mask=True)
        features = whisper._mask_input_features(
            inputs.input_features, attention_mask=inputs.attention_mask
        ).to(device)
        with torch.no_grad():
            out = whisper.encoder(features.to(whisper.encoder.dtype),
                                  head_mask=None, output_attentions=False,
                                  output_hidden_states=False, return_dict=True)
        hidden = out.last_hidden_state.to(torch.float32)
        return hidden[:, :waves_16k.size(-1) // 320 + 1]

    _LOADED = {
        "model": model, "campplus": campplus, "vocoder": vocoder,
        "semantic": semantic, "sr": sr, "hop": spect["hop_length"],
        "mel": lambda x: mel_spectrogram(x, **mel_args), "device": device,
        "load_s": round(time.time() - started, 2),
    }
    return _LOADED


def _embedding(parts, wave, sr):
    import torch
    import torchaudio
    with torch.no_grad():
        tensor = torch.tensor(wave).unsqueeze(0).float().to(parts["device"])
        wave_16k = torchaudio.functional.resample(tensor, sr, 16000)
        fbank = torchaudio.compliance.kaldi.fbank(
            wave_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        vector = parts["campplus"](fbank.unsqueeze(0))[0]
    return vector / vector.norm()


def _similarity(parts, output, reference, source, sr):
    """Cosine similarity to the voice sample: converted, and unconverted."""
    target = _embedding(parts, reference, sr)
    converted = float((_embedding(parts, output, sr) * target).sum())
    original = float((_embedding(parts, source, sr) * target).sum())
    return round(converted, 4), round(original, 4)


def _semantic_long(semantic, waves_16k):
    """Whisper takes 30 s at a time; longer lines go in overlapping pieces."""
    import torch
    if waves_16k.size(-1) <= 16000 * 30:
        return semantic(waves_16k)
    overlap = 5
    pieces, buffer, done = [], None, 0
    while done < waves_16k.size(-1):
        if buffer is None:
            chunk = waves_16k[:, done:done + 16000 * 30]
        else:
            chunk = torch.cat([buffer, waves_16k[:, done:done + 16000
                                                 * (30 - overlap)]], dim=-1)
        features = semantic(chunk)
        pieces.append(features if done == 0 else features[:, 50 * overlap:])
        buffer = chunk[:, -16000 * overlap:]
        done += 30 * 16000 if done == 0 else chunk.size(-1) - 16000 * overlap
    return torch.cat(pieces, dim=1)


def convert(source_path, reference_path, target_path,
            steps=DIFFUSION_STEPS, cfg_rate=CFG_RATE):
    """Convert one recording. Returns measurements for voice.json."""
    import librosa
    import numpy as np
    import soundfile
    import torch
    import torchaudio

    parts = load()
    sr, device = parts["sr"], parts["device"]
    model = parts["model"]
    started = time.time()
    source = librosa.load(source_path, sr=sr)[0]
    reference = librosa.load(reference_path, sr=sr)[0][:int(sr * REFERENCE_MAX_S)]

    with torch.no_grad():
        src = torch.tensor(source).unsqueeze(0).float().to(device)
        ref = torch.tensor(reference).unsqueeze(0).float().to(device)
        src_16k = torchaudio.functional.resample(src, sr, 16000)
        ref_16k = torchaudio.functional.resample(ref, sr, 16000)
        content = _semantic_long(parts["semantic"], src_16k)
        prompt_content = parts["semantic"](ref_16k)
        mel = parts["mel"](src)
        mel_ref = parts["mel"](ref)
        lengths = torch.LongTensor([mel.size(2)]).to(device)
        ref_lengths = torch.LongTensor([mel_ref.size(2)]).to(device)
        fbank = torchaudio.compliance.kaldi.fbank(
            ref_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        style = parts["campplus"](fbank.unsqueeze(0))
        cond = model.length_regulator(content, ylens=lengths,
                                      n_quantizers=3, f0=None)[0]
        prompt = model.length_regulator(prompt_content, ylens=ref_lengths,
                                        n_quantizers=3, f0=None)[0]

        window = sr // parts["hop"] * 30 - mel_ref.size(2)
        overlap_frames = 16
        overlap_wave = overlap_frames * parts["hop"]
        chunks, done, previous = [], 0, None
        while done < cond.size(1):
            piece = cond[:, done:done + window]
            last = done + window >= cond.size(1)
            both = torch.cat([prompt, piece], dim=1)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                target = model.cfm.inference(
                    both, torch.LongTensor([both.size(1)]).to(device),
                    mel_ref, style, None, steps, inference_cfg_rate=cfg_rate)
                target = target[:, :, mel_ref.size(-1):]
            wave = parts["vocoder"](target.float()).squeeze()[None, :]
            wave = wave[0].cpu().numpy()
            if previous is None and last:
                chunks.append(wave)
                break
            if previous is not None:
                fade_out = np.cos(np.linspace(0, np.pi / 2, overlap_wave)) ** 2
                fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap_wave)) ** 2
                wave[:overlap_wave] = (wave[:overlap_wave] * fade_in
                                       + previous * fade_out)
            if last:
                chunks.append(wave)
                break
            chunks.append(wave[:-overlap_wave])
            previous = wave[-overlap_wave:]
            done += target.size(2) - overlap_frames

    output = np.concatenate(chunks).astype(np.float32)
    raw_len = len(output)
    if len(output) < len(source):
        output = np.pad(output, (0, len(source) - len(output)))
    else:
        output = output[:len(source)]
    rms_in = float(np.sqrt(np.mean(source ** 2)) or 1e-9)
    rms_out = float(np.sqrt(np.mean(output ** 2)) or 1e-9)
    gain = min(8.0, rms_in / rms_out)
    output = np.clip(output * gain, -1.0, 1.0)
    soundfile.write(target_path, output, sr, subtype="PCM_16")
    similarity, baseline = _similarity(parts, output, reference, source, sr)
    return {
        # CAM++ cosine between the converted line and the voice sample, and
        # the same for the unconverted recording: the second is what the first
        # has to beat for the conversion to have done anything.
        "speaker_similarity": similarity,
        "source_similarity": baseline,
        "backend": "seedvc",
        "backend_model": f"{REPO}/{CHECKPOINT}",
        "converted": True,
        "sample_rate": sr,
        "diffusion_steps": steps,
        "length_fix_samples": len(source) - raw_len,
        "gain": round(gain, 4),
        "reference_s": round(len(reference) / float(sr), 3),
        "seconds": round(time.time() - started, 2),
        "load_s": parts["load_s"],
    }
