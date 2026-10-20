# Shahnameh Animation Pipeline — Design Plan

**Status:** design only, nothing implemented yet.
**Date:** 2026-09-03
**Goal:** turn this repo from a one-prompt scene generator into a production pipeline that
animates episodes of the Shahnameh, using generated 3D characters, the user's own body
performance, and the user's own recorded voice.

---

## 1. What this document is

The current program does one thing: a text prompt goes to an LLM, the LLM writes Blender
Python, the host renders one image or one short clip. Every run is disposable.

What is wanted instead is a *film pipeline*: a story becomes a shot list, characters become
reusable rigged assets built from reference images, motion comes from video of the user
performing, mouths come from the user's recorded dialogue, and shots accumulate into an
edit. This document specifies that pipeline, the order to build it in, and the decisions
already locked down.

Implementation is deliberately out of scope here.

---

## 2. Locked decisions

These come from the Q&A and are treated as fixed constraints for the rest of the design.

| # | Decision | Consequence for the design |
|---|----------|----------------------------|
| 1 | **All three visual looks stay selectable** — Persian miniature, stylized 3D, photoreal | Shading, lighting, camera and post-FX must live behind a swappable *style pack* layer. Assets and shot data must be style-agnostic: materials assigned by semantic role, never by hardcoded colour. See §8. |
| 2 | **Body + head pose from video; mouth always from audio** | Facial expression is *not* transferred from the performance video. Two independent, reliable signal paths. Emotion on the face comes from the dialogue audio and from explicit direction, not from monocular face tracking. See §7-E, §7-G. |
| 3 | **Rig mode is per character, hybrid by default, user-editable** | A checkbox table in the GUI, prefilled (humans → SMPL-X, creatures → rigged mesh), every row flippable. Overrides stored in a sidecar file, not in the LLM's output. See §9. |
| 4 | **First deliverable is one 10–20 s shot, end to end** | Build one complete vertical slice before deepening any stage. Milestone M1 stubs every ML model so the plumbing is proven first. See §13. |
| 5 | **Reference images are dropped into `refs/` by hand** | No image API key, no image generation code. A watcher validates, hashes and registers whatever appears. The user keeps art control and can re-roll a character until it is right. See §7-B. |
| 6 | **Stage A (story → prompts) is run manually in a strong chat model** | The host does not call a strong LLM. It defines a strict JSON Schema, the user pastes the reply in as `story/production.json`, and the host validates it hard before anything expensive runs. See §7-A. |
| 7 | **Dialogue is Persian only** | Forced alignment must work on Persian; the phoneme→viseme map is Persian-specific. Subtitles are RTL. Voice conversion is unaffected (language-agnostic). See §7-G. |
| 8 | **Personal / research use only, not commercial** | SMPL-X, FLAME, DECA/EMOCA/MICA, Hunyuan3D and research mocap checkpoints are all usable. No model is off the table. Licences are still tabulated in §14 so the constraint is visible if it ever changes. |

---

## 3. Verified environment

Everything in this table was checked on this machine, not assumed.

| Thing | Reality |
|---|---|
| GPU | NVIDIA RTX 4070 **Laptop**, **8 GB VRAM**, driver 591.86 / CUDA 13.1 |
| CPU | Intel i7-13620H |
| Blender | 5.2.0 LTS |
| Cycles compute backends | `['NONE', 'CUDA', 'OPTIX', 'HIP', 'ONEAPI']` — **OptiX and CUDA are both available and enumerate the 4070** |
| BlenderMCP | installed as a legacy addon, `scripts/addons/blender_mcp.py`; `extensions/` is empty |
| Existing weights in repo | `model.safetensors`, 4.0 GB, 1276 F32 tensors, prefixes `backbone / bbox / camera_embedder / decoder / global_estimator / image_estimator / image_tokenizer / post_processor / tokenizer` → **Stable Fast 3D**, but the download is incomplete (no `config.yaml`) |
| conda envs | `agent` (openai + PyQt6, **no torch**), `comfyui` and `fooocus` (torch/transformers/diffusers), `moviepy2` (moviepy) |
| Local image generation | ComfyUI portable exists at `D:\work\ComfyUI\...` but **every model directory is empty**. Consistent with decision 5 — not used. |
| Secrets | `config.json` holds a live NVIDIA API key in plaintext |

**Two documented claims are false and must be corrected:** the README and
`prompts.py::BLENDER_FACTS` both tell the reader (and the LLM) that *"Cycles has no GPU
backend on this machine, so it renders on the CPU and is slow"*, and
`prelude.py` hardcodes `scene.cycles.device = "CPU"`. OptiX is available. This is leaving
the single biggest quality/speed lever in the project switched off. See §15.

---

## 4. The core reframing

> **The LLM stops modelling and starts directing.**

Today the model's only way to affect the scene is `execute_blender_code`, and it builds
everything out of cubes, cylinders and spheres. That approach cannot produce Rostam. It
can produce a competent abstract composition, and nothing more.

In the new pipeline, geometry and motion no longer come from the LLM at all:

- **Geometry** comes from generative 3D models and from a parametric human body.
- **Motion** comes from capture — video for the body, audio for the mouth.
- **Appearance** comes from reference images the user approved by eye.

The LLM's remaining job is *staging*: which character stands where, which camera, which
lens, which light rig, which beat happens on which frame, which line is spoken. That is a
far smaller and far more checkable output space, and it is exactly what a modest model can
do reliably — **provided the host gives it a typed schema to fill rather than a Python
interpreter to improvise in.**

This changes the biggest failure mode of the current design. A malformed shot description
becomes a schema validation error the host can report precisely, instead of a Python
traceback the model has to guess its way out of. Free-form Python survives as an escape
hatch for one-off effects, but it is no longer the primary interface.

Everything else in this document follows from that inversion.

---

## 5. Architecture

### 5.1 A staged graph over a content-addressed artifact store

Each stage is a standalone CLI with declared inputs and outputs. It reads files, writes
files, and exits. Nothing is held in memory between stages.

```
A  story.md + seed_prompt.md ──(manual, strong chat model)──▶ story/production.json  [schema-validated]
B  production.json ──(user generates images by hand)──▶ refs/characters/*.png, refs/envs/*.png
C  refs/characters/* ──▶ assets/characters/<id>/            per-character, branch on rig mode
     C1 face identity   image ──▶ FLAME identity params
     C2 body identity   image ──▶ SMPL-X betas (+ manual sliders)
     C3 appearance      image ──▶ albedo / diffuse texture
     C4 garments        image ──▶ mesh (SF3D / Hunyuan3D) ──▶ bound to body
     C5 rigged path     image ──▶ mesh ──▶ auto-rig ──▶ skeleton + weights
D  refs/envs/*   ──▶ assets/envs/<id>/                      backdrop, HDRI, scatter props
E  performance/<take>.mp4 ──▶ motion/<take>/body.npz        SMPL-X pose sequence + head pose
F  audio/<line>/source.wav ──▶ audio/<line>/converted.wav   voice conversion to character voice
G  audio/<line>/converted.wav ──▶ align.json ──▶ visemes.json ──▶ FLAME jaw + lip curves
H  production.json + assets + motion + visemes ──▶ shots/<id>/shot.blend
I  style pack applied to shot.blend                          materials, lights, camera, post-FX
J  shot.blend ──▶ shots/<id>/frames/*.png ──▶ shot.mp4       render, then mux the audio
K  all shots ──▶ film/edit.json ──▶ film/final.mp4           concatenate, subtitles
```

Every artifact is keyed by a content hash of `(stage version, input hashes, parameters)`.
Two consequences that matter a great deal in practice:

- **Resumable.** A 40-minute mocap stage is never re-run because a later render failed.
- **Cacheable across shots.** Rostam's body, face and armour are built once and reused by
  every shot he appears in, which is also what makes him look like the same person twice.

### 5.2 The audio is the master clock

Because the body performance and the dialogue are recorded separately (decision 2), one of
them has to win when they disagree about length. **The dialogue audio wins.**

A shot's frame count is derived from its aligned dialogue duration plus handles:

```
frames = ceil((align.end - align.start + head_handle + tail_handle) * fps)
```

The body take is then fitted to that window by an explicit, per-shot `retime` policy —
`trim`, `hold_last_pose`, `loop`, or `timewarp` — with an `offset_frames` field so the user
can nudge the performance against the line. This is a small amount of design work now that
prevents a whole category of "why are the arms still moving after he stopped talking"
problems later.

The mouth curves are frame-locked to the audio by construction, so lip sync cannot drift.

### 5.3 A project is a directory

One film = one project directory. The pipeline never writes outside it, which makes
projects copyable, diffable and disposable.

```
projects/rostam_sohrab/
  project.json                  fps, resolution, style pack, engine, defaults
  story/
    story.md                    the source text
    seed_prompt.md              versioned; tells the strong LLM what to produce
    production.json             THE contract: characters, envs, shots, dialogue
    rig_overrides.json          sidecar from the checkbox table (§9)
  refs/
    characters/<id>/*.png       dropped in by hand
    envs/<id>/*.png
  performance/
    <take_id>/video.mp4         body performance, one take per shot or per beat
  audio/
    <line_id>/source.wav        the user speaking the line
    voices/<char_id>.wav        reference voice sample for that character
  assets/
    characters/<id>/            shape.npz, face.npz, albedo.png, garments/*.glb, rig.blend
    props/<id>/mesh.glb
    envs/<id>/                  backdrop.png | hdri.exr, props.json
  motion/<take_id>/             body.npz, head_pose.npz, quality.json
  shots/<shot_id>/
    shot.json                   resolved staging: who, where, camera, timing
    scene.blend
    frames/*.png
    shot.mp4
  film/
    edit.json, final.mp4, subs_fa.srt
  .cache/                       content-addressed artifact store
```

---

## 6. Data model

### 6.1 `production.json` — the contract with the strong LLM

This is the single most important file in the design. It is what the strong chat model
returns, and the only thing the pipeline trusts about the story. It is validated against a
JSON Schema before any stage runs; a missing field is a hard failure with a precise message,
never a silent default.

```jsonc
{
  "title": "رستم و سهراب — the first meeting",
  "fps": 24,
  "characters": [
    {
      "id": "rostam",
      "name_fa": "رستم",
      "kind": "human",                    // human | humanoid | creature | quadruped
      "rig_default": "smplx",             // derived from kind; user may override (§9)
      "height_m": 2.05,                   // drives SMPL-X scale + sanity gate
      "image_prompt": "...",              // prompt 2 — user pastes into their image model
      "voice_ref": "audio/voices/rostam.wav",
      "material_roles": {                 // style-agnostic: role, never colour
        "skin": "skin_weathered",
        "armour": "metal_scale",
        "robe": "cloth_heavy",
        "belt": "leather_tooled"
      }
    }
  ],
  "envs": [
    {
      "id": "helmand_bank",
      "interior": false,
      "image_prompt": "...",              // prompt 3
      "ground": "sand_dry",
      "scatter": [ { "role": "rock", "count": 40, "area_m": [30, 30], "avoid_radius_m": 1.5 } ],
      "light_intent": "late afternoon, low sun from camera left, long shadows"
    }
  ],
  "shots": [
    {
      "id": "s01",
      "env": "helmand_bank",
      "cast": [ { "character": "rostam", "take": "take_042", "retime": "hold_last_pose",
                  "offset_frames": 0, "position_m": [0, 0, 0], "facing_deg": 15 } ],
      "dialogue": [ { "line_id": "l001", "character": "rostam",
                      "text_fa": "...", "emotion": "grave" } ],
      "camera": { "position_m": [3.2, -4.5, 1.6], "aim": "rostam.head",
                  "focal_mm": 85, "dof": true },
      "beats": [ { "from_frame": 1, "to_frame": 120, "what": "slow push in, 3.2m to 2.6m" } ]
    }
  ]
}
```

Note what is *absent*: no colours, no light wattages, no Blender API names, no engine
choice. Those belong to the style pack, so the same `production.json` renders in all three
looks.

### 6.2 `rig_overrides.json` — the sidecar

The LLM's output stays pristine. The checkbox table writes here instead:

```json
{ "rostam": "smplx", "sohrab": "smplx", "div_e_sepid": "smplx", "simorgh": "rigged" }
```

Resolution order: `rig_overrides.json` → `character.rig_default` → derived from `kind`.

### 6.3 `shot.json` — the resolved shot

Where `production.json` is the *intent*, `shot.json` is the *fully resolved* shot after the
host has filled in everything it knows: real frame counts from the aligned audio, real asset
paths from the store, the chosen style pack, the resolved camera. It is what `h_assemble`
compiles into a `.blend`, and it is the thing to hand-edit when a shot is nearly right.

---

## 7. Stage designs

Each stage below lists what it consumes, what it produces, the model chosen, and why.

### A — Production (manual, then validated)

- **In:** `story/story.md`, `story/seed_prompt.md`
- **Out:** `story/production.json`
- **How:** the user pastes the story and seed prompt into a strong chat model and saves the
  reply. The host's job is *validation only*.
- **Design work:** write the JSON Schema, and write `seed_prompt.md` so that the strong model
  reliably emits conforming JSON — including the Shahnameh **style bible** (Sassanid and
  Achaemenid armour forms, manuscript palettes, canonical epithets, what Rakhsh looks like)
  so cultural detail is specified once and versioned, not re-improvised per shot.
- **Gate:** schema-valid; every `shots[].env` and `shots[].cast[].character` exists; every
  dialogue line has a matching `audio/<line_id>/source.wav`; frame ranges non-overlapping.

### B — Reference images (dropped in by hand)

- **In:** files the user saves into `refs/`
- **Out:** registered, hashed entries in the asset store
- **Design work:** a watcher/validator, not a generator. Checks resolution ≥ 1024 px, that
  the image is not blank, and — for character refs — that a face is detectable. Expects a
  small naming convention (`<char_id>_front.png`, `<char_id>_face.png`, `<char_id>_side.png`)
  and reports what is missing per character rather than failing on the first gap.

### C — Character build (the heart of it)

Branches on rig mode. **Both branches produce the same interface** — a named Blender object
with a skeleton, a material set keyed by role, and a mouth-control API — so `h_assemble`
does not care which path a character came from.

#### C-smplx: parametric path (default for humans)

The character *is* a textured SMPL-X body. This is the decision that removes retargeting
from the project entirely:

- The mocap stage outputs SMPL-X parameters, so they play on the character directly.
- **SMPL-X's head is FLAME**, so face identity params and mouth curves target the same rig.
- Identity is a vector of shape coefficients, so a character is a small file, not a mesh.
- Garments, armour and hair are *separate meshes* bound to the body with Surface Deform /
  Mesh Deform plus corrective shape keys at the shoulders. This is how Rostam gets scale
  armour without anyone solving auto-rigging.

**On face identity — use MICA, not DECA.** DECA was the user's suggestion and it is the
right family, but DECA's identity shape is entangled with whatever expression the subject
happened to be making in the photo, which is exactly wrong when the goal is a neutral
identity that the audio will later animate. MICA was built to fix that specific problem and
returns a clean neutral FLAME shape from one image. Plan: **MICA for identity, DECA/EMOCA
kept as a fallback**, and no expression is taken from the reference image at all.

**Honest weak point.** These fitters are trained on photographs. The reference images here
will be *illustrations* — possibly stylised Persian-miniature illustrations. Automatic
identity fitting will be unreliable, and this is the most likely place for the pipeline to
disappoint. Mitigation, designed in from the start rather than bolted on:

1. Attempt the automatic fit and report its landmark reprojection error.
2. Expose SMPL-X `betas` and FLAME identity as **sliders in the GUI**, with a small preset
   library (`heroic`, `slight`, `aged`, `giant`, `youth`) tuned for Shahnameh archetypes.
3. Treat the reference image primarily as the source of **appearance** (albedo, colour,
   ornament) — which is robust — rather than of proportions, which is not.

#### C-rigged: generated-mesh path (default for creatures)

- **Mesh:** SF3D for speed and clean UVs (weights already here, needs its `config.yaml`);
  Hunyuan3D 2.1 for hero assets that need real PBR maps, run with low-VRAM offloading.
- **Rig:** UniRig for predicted skeleton + skin weights, Blender Rigify as the manual
  fallback.
- **Motion:** *not* from the user's body. A human performance cannot drive a Simorgh. These
  characters are hand-keyed or procedurally animated (wing cycles, tail follow-through), and
  the plan should say so plainly rather than implying the mocap covers them.

### D — Environment build

- **In:** `refs/envs/<id>/*.png`, the env block of `production.json`
- **Out:** a backdrop strategy plus scatter props
- **Design:** three interchangeable strategies chosen by the style pack, because "the
  environment" means very different things in the three looks:
  - *miniature*: the reference image becomes a painted backdrop on stacked planes, with a
    near-orthographic camera. Cheap, beautiful, and stylistically correct.
  - *stylized*: procedural ground from `bstudio`'s existing material presets + scattered
    SF3D props + a gradient or HDRI sky.
  - *photoreal*: Poly Haven HDRI (already wired via the existing `search_textures` /
    `apply_texture` tools) + real displaced terrain + volumetric haze.
- The existing `B.scatter(...)` helper already covers the count/area/avoid-radius contract
  that `production.json` specifies, so this stage is mostly glue.

### E — Body motion from video

- **In:** `performance/<take>/video.mp4`
- **Out:** `motion/<take>/body.npz` (SMPL-X pose sequence), `head_pose.npz`, `quality.json`
- **Model:** **GVHMR** or **TRAM** — world-grounded monocular SMPL recovery with good foot
  contact, which matters because a floating or skating Rostam reads as broken instantly.
  Human3R is the newer option to evaluate later.
- **Explicitly not extracted:** facial expression. Per decision 2, only body pose and head
  orientation come from the video.
- **Post-process:** temporal smoothing, foot-lock/ground solve, NaN interpolation, and a
  velocity clamp. Raw monocular mocap is always too jittery to use directly.
- **Gate:** no NaNs; per-joint angular velocity under a limit; foot skate under ~2 cm/frame;
  root trajectory continuous.

### F — Voice conversion

The user speaks every line, then supplies a reference voice per character. **This is voice
conversion, not text-to-speech** — which is a significant simplification with three benefits:
Persian needs no TTS support, the user's own prosody and acting survive intact, and the
timing is preserved so the lip sync computed from the audio stays valid.

- **Model:** **Seed-VC** (zero-shot, diffusion transformer) as primary; **REF-VC** if the
  recordings turn out noisy.
- **Critical constraint:** the conversion must be **duration-preserving**. A model that
  resynthesises with new timing would silently invalidate every mouth curve.
- **Gate:** `abs(len(out) − len(in)) < 10 ms`; speaker similarity to the reference above a
  threshold; ASR word error rate against the source transcript below ~25% so the words are
  still intelligible after conversion.

### G — Mouth animation from audio

Two routes, and the plan deliberately keeps both because they suit different looks.

**G1 — alignment → visemes (the baseline).** Forced-align the Persian audio to get word and
phoneme timings, map Persian phonemes onto a viseme set, and emit FLAME jaw-pose and lip
expression curves directly.

- **Aligner:** WhisperX with a Persian wav2vec2 alignment model, or MFA with a Persian
  dictionary. Fallback if neither is reliable: Whisper word timestamps plus a Persian G2P,
  distributing phonemes proportionally inside each word. Less precise, still usable.
- **Why this is the baseline rather than the fancy option:** it is deterministic, debuggable,
  fully controllable, and it does not care that the language is Persian. Every curve can be
  traced back to a phoneme with a start and end time.

**G2 — NVIDIA Audio2Face-3D (the upgrade).** Audio straight to ARKit blendshapes, including
an emotion channel. Map the ARKit mouth/jaw channels onto FLAME. Better realism, but it is
trained predominantly on English and Persian phonetics differ, so it is an *option* selected
by the style pack, not the default.

**Style pack picks the route:** photoreal wants A2F-3D's subtlety; the miniature look wants
G1's small set of snappy, graphic mouth shapes, because smooth realistic co-articulation
looks wrong on a flat painted face.

**Where facial emotion comes from,** given the video deliberately does not supply it: the
`emotion` field on each dialogue line in `production.json`, expanded by the style pack into
brow/eye/cheek poses, plus A2F-3D's emotion channel when that route is active. Explicit and
directable, rather than accidental.

### H — Shot assembly

- **In:** `shot.json` + built assets + motion + mouth curves
- **Out:** `shots/<id>/scene.blend`
- **Design:** this replaces the current `build_scene` tool loop as the primary path. It is
  **deterministic host code**, not LLM code: place characters, apply motion with the retime
  policy, attach mouth curves, load the environment, set the camera, set the frame range from
  the audio. The LLM contributes the *content* of `shot.json`, never the assembly.
- `prelude.py` gains a character/mocap/audio layer: `B.character(...)`, `B.apply_motion(...)`,
  `B.apply_visemes(...)`, `B.bind_garment(...)`, `B.style(...)`.
- **Gate:** every object has a material; feet contact the ground plane; the character is
  inside the frustum; frame range matches the audio; no NaN transforms.

### J — Render and mux

- Keep the existing per-frame render over the socket for **preview** — it is what makes the
  live viewport and real progress reporting possible, and that was a good decision.
- Add a **headless batch worker** for final renders: a separate Blender process with
  `--frame-start/--frame-end`, `use_persistent_data`, and OptiX enabled. For a 360-frame shot
  in Cycles this is the difference between minutes and hours.
- Mux the converted dialogue audio into the shot's MP4. The current `encode_video` builds a
  video-only file; it needs an audio strip in the throwaway encode scene, or an ffmpeg mux
  pass.
- **Gate:** the existing `analyse()` checks (near-black, blown-out, contrast, framing) plus
  two new ones described in §11.

### K — Edit

Concatenate shot MP4s per `film/edit.json`, burn or attach RTL Persian subtitles from the
alignment data (which already contains word timings, so subtitles are free), write
`film/final.mp4`.

---

## 8. Style packs — how all three looks stay available

Decision 1 was "keep the three of them as options". That is only affordable if style is
factored out of everything else, so this is the structural centre of the design.

A style pack is data plus a small hook module:

```
blender_ai_studio/styles/
  miniature/
    style.json        engine, view transform, palette, light rig, camera policy, mouth profile
    shading.py        build_material(role, params) -> Blender node tree
    postfx.py         compositor graph: paper grain, ink edges, gold leaf, flat bloom
  stylized/
  photoreal/
```

Each pack declares:

| Field | miniature | stylized | photoreal |
|---|---|---|---|
| engine | EEVEE | Cycles / OptiX | Cycles / OptiX |
| samples | 32 | 128 | 512+ |
| view transform | Standard | AgX | AgX |
| world strategy | painted backdrop planes | gradient + props | HDRI + volumetrics |
| camera policy | near-ortho, 50–85 mm, no DOF | 24–85 mm, mild DOF | physical, DOF, motion blur |
| mouth profile | G1 visemes, snap interpolation | G1 visemes, smoothed | G2 Audio2Face-3D |
| post FX | ink edge, paper grain, gold accents | bloom, subtle grade | full grade, bloom, vignette |

The rule that makes this work: **assets and shot data reference material *roles*, never
concrete materials.** `production.json` says `"armour": "metal_scale"`; the miniature pack
turns that into flat gold with an ornament decal and an ink outline, the photoreal pack turns
it into a rough metal BSDF with scratches and an anisotropic highlight. Same asset, same
shot, two renders.

Practical payoff for the first milestone: rendering the one 10–20 s slice in all three styles
is nearly free once the pack layer exists, and it is the fastest way to decide which look is
actually worth pursuing for the Shahnameh.

---

## 9. The rig-mode checkbox table

Exactly as requested: a table in the GUI, prefilled with sensible defaults, every row
editable before generation starts.

```
 Character        SMPL-X   Rigged mesh   Consequence
 ─────────────────────────────────────────────────────────────────────────────
 Rostam             [x]        [ ]       mocap-native, FLAME face, audio lips
 Sohrab             [x]        [ ]       mocap-native, FLAME face, audio lips
 Div-e Sepid        [x]        [ ]       humanoid → driven by your performance
 Simorgh            [ ]        [x]       generated mesh + UniRig, hand-keyed
 Rakhsh             [ ]        [x]       quadruped → procedural gait
```

- Prefill from `character.kind`: `human`/`humanoid` → SMPL-X, `creature`/`quadruped` → rigged.
- The two checkboxes are an **exclusive pair per row** — clicking one clears the other —
  because a character has one body representation. Flipping a div to SMPL-X is fully
  supported and is exactly the case the user called out.
- The **Consequence** column is not decoration. It tells the user what they just chose: that
  moving a character to *rigged* means their performance no longer drives it and the motion
  becomes hand-keyed work. That is the kind of thing that should be visible at the moment of
  the decision, not discovered three stages later.
- Writes to `story/rig_overrides.json`, leaving the LLM's `production.json` untouched.

---

## 10. VRAM and process isolation

8 GB is the hard constraint that shapes the engineering. Rules:

1. **One GPU model resident at a time.** Every ML stage runs in its own subprocess and exits.
   Process exit is the only reliable way to free VRAM.
2. **One conda env per stage family.** These models have genuinely conflicting torch,
   diffusers and CUDA requirements; a single shared env will not hold. A stage declares which
   interpreter it needs and the runner invokes that `python.exe` directly. (Note: invoking
   `conda run -n <env>` failed on this machine; calling
   `~/anaconda3/envs/<env>/python.exe` directly works.)

3. **Never hold an ML model while Blender renders.** Both want the same 8 GB. The graph runs
   all ML stages to completion, frees the GPU, and only then renders.
4. **Hunyuan3D 2.1 will not fit comfortably.** Plan for its low-VRAM/offload path, and treat
   SF3D as the default with Hunyuan3D reserved for a handful of hero assets.
5. Artifacts on disk, not in memory. The graph passes file paths between stages.

---

## 11. Quality gates

The best idea in the existing codebase is that the host *measures* the result instead of
believing the model's claim that it is finished. That principle extends to every new stage.

| Stage | Metric | Threshold |
|---|---|---|
| A | schema valid, all cross-references resolve, every line has audio | hard fail |
| B | ≥1024 px, non-blank, face detectable on character refs | hard fail |
| C-face | MICA landmark reprojection error | < ~4 px on a 512 crop |
| C-body | height after scale within 1.4–2.6 m | warn |
| C-asset | tri count, UV coverage, no zero-area faces, sane bbox | warn + auto-fix |
| E | no NaN, joint velocity clamp, foot skate < 2 cm/frame | auto-smooth, then warn |
| F | duration delta < 10 ms, speaker similarity, ASR WER < 25% | hard fail |
| G | every word timed, no phoneme < 20 ms, audio coverage > 90% | hard fail |
| H | every object has a material, feet on ground, subject in frustum | hard fail |
| J | existing luminance/contrast/framing checks | pushback loop, max 2 |
| J | **temporal flicker** — mean absolute luminance delta between frames | new |
| J | **lip-sync score** — correlation of the FLAME jaw-open curve against the audio RMS envelope | new |

The two new render metrics are worth calling out because the current code has neither and
both catch failures that are invisible in a single still:

- **Temporal flicker** catches EEVEE sample noise and unstable shadows — the thing that makes
  an otherwise fine animation look cheap. A still frame can pass every existing check while
  the sequence boils.
- **Lip-sync score** is cheap and needs no model: the jaw-open parameter is something the
  pipeline itself authored, so correlating it against the audio envelope directly verifies
  that the mouth is moving when sound is happening. It catches off-by-one-frame and
  wrong-line-attached-to-wrong-shot bugs immediately, which are otherwise painful to spot.

---

## 12. Modules — what is new, what changes, what stays

**Stays as it is (good work, keep it):** `blender_link.py` (the socket protocol, buffering and
traceback harness), the live-preview thread, `analyse()` / `report()` / `auto_exposure()`, the
material and geometry helpers in `prelude.py`, the host-measures-the-result quality gate.

**Changes:**

| File | Change |
|---|---|
| `prelude.py` | remove the hardcoded `cycles.device = "CPU"`, enable OptiX; add the character/mocap/audio/style layer (`B.character`, `B.apply_motion`, `B.apply_visemes`, `B.bind_garment`, `B.style`) |
| `prompts.py` | `BLENDER_FACTS` no longer claims Cycles is CPU-only; the Director's job changes from "write a brief" to "fill the shot schema"; the Builder's tool surface shrinks toward staging |
| `tools.py` | `execute_blender_code` demoted to an escape hatch; new typed staging tools that write validated `shot.json` fragments |
| `runner.py` | replaced as the top level by the stage graph; retained as the single-shot render path |
| `session.py` | add the headless batch render worker; add audio muxing to `encode_video` |
| `gui.py` | new inputs (story, seed prompt, refs folder, performance video, dialogue audio, voice samples); the rig-mode checkbox table; a `production.json` review/edit tab; a shot list |
| `config.py` | secrets out of `config.json`; per-stage env paths; style pack selection |

**New:**

```
pipeline/
  schema/           production.schema.json, shot.schema.json, character.schema.json
  store.py          content-addressed artifact store
  graph.py          stage graph, dependency resolution, resume
  envs.py           per-stage interpreter resolution
  gates.py          the metrics in §11
  stages/           a_production  b_refs  c_face  c_body  c_texture  c_garment  c_rigged
                    d_env  e_motion  f_voice  g_align  h_assemble  j_render  k_edit
blender_ai_studio/styles/   miniature/  stylized/  photoreal/
projects/                   one directory per film
```

---

## 13. Milestones

The sequencing matters more than any individual technical choice here.

### M0 — Foundations and verified fixes (no ML at all)

OptiX enabled, the UTF-8 crash fixed, secrets moved out of `config.json`, project scaffold,
artifact store, JSON Schemas, stage graph runner. Nothing visible changes, everything after
this depends on it.

### M1 — The vertical slice, with every ML model stubbed

**This is the most important process decision in the plan.** Produce the full 10–20 s shot
end to end while *faking* every heavy stage:

- body identity: a preset SMPL-X `betas` vector, no fitting
- face identity: a hand-picked FLAME identity, no fitting
- body motion: one recorded take run through a placeholder that emits a simple walk/gesture
  cycle, or a single pose held
- voice: the user's raw recording, no conversion
- mouth: G1 visemes from alignment — the one real ML piece, because it is cheap and it is what
  makes the shot read as a character speaking
- assets: primitives from the existing `bstudio` helpers

Then assemble, apply a style pack, render, mux the audio, and watch it. The result will look
rough, and it will prove that all fourteen stages, the schema, the artifact store, the retime
policy, the frame-range-from-audio rule and the mux all actually connect.

Every stage after this becomes a *substitution* into a working pipeline rather than a step
toward a pipeline that has never run. This is the difference between a project that ships and
one that has six impressive half-finished stages.

Also renders the slice in all three styles, which answers the look question with pictures.

### M2 — Real identity

MICA face fit, SMPL-X body fit, albedo projection from the reference image, plus the slider/
preset fallback UI that the illustration problem (§7-C) makes necessary.

### M3 — Real motion

GVHMR or TRAM video → SMPL-X, the smoothing and foot-lock post-pass, and the retime policies
against the audio clock.

### M4 — Real voice and mouth

Seed-VC conversion with its duration and similarity gates; Persian forced alignment properly
evaluated; Audio2Face-3D wired in as the photoreal option.

### M5 — Assets and environments

SF3D completed (it needs its `config.yaml`), Hunyuan3D for hero pieces, garment binding with
correctives, environments from reference images per style strategy.

### M6 — Multi-shot

Shot list, asset reuse and continuity across shots, the edit stage, RTL Persian subtitles.

### M7 — Creatures

UniRig path, hand-keyed and procedural creature motion, the Simorgh and Rakhsh.

---

## 14. Model and licence table

Decision 8 makes all of these usable. Tabulated anyway so the constraint is visible if the
project ever changes intent.

| Role | Choice | Alternative | Licence note |
|---|---|---|---|
| Body model | SMPL-X | MHR (Meta, permissive) | research / non-commercial |
| Face model | FLAME | — | research / non-commercial |
| Face identity from image | **MICA** | DECA, EMOCA, SMIRK | research |
| Body identity from image | SMPLer-X, InstantHMR | SAM 3D Body (outputs MHR) | research |
| Video → motion | GVHMR / TRAM | Human3R, ProxyCap | research |
| Audio → mouth | phoneme→viseme (own code) | NVIDIA Audio2Face-3D | A2F-3D open-sourced |
| Voice conversion | Seed-VC | REF-VC, GenVC | permissive |
| Persian alignment | WhisperX + fa wav2vec2 | MFA + fa dictionary | permissive |
| Image → 3D asset | **SF3D** (weights already here) | Hunyuan3D 2.1, TRELLIS 2 | Hunyuan3D excludes EU/UK/KR |
| Auto-rig | UniRig | Rigify (manual) | research |
| Textures / HDRI | Poly Haven (already wired) | — | CC0 |
| Staging LLM | `openai/gpt-oss-120b` via NVIDIA NIM | — | free tier, rate-limited |

One consequence of MHR being the permissive option and SAM 3D Body outputting MHR: if this
ever needs to go commercial, the migration is SMPL-X → MHR, and NVIDIA's SOMA-X exists to map
between parametric body models. Worth knowing, not worth building now.

---

## 15. Bugs to fix first

All three were found by inspection or by running the code, not guessed.

**1. Cycles is pinned to CPU for no reason.** `prelude.py`:

```python
# No GPU compute backend is configured on many machines; CPU is the
# only setting guaranteed to render rather than fail.
try:
    scene.cycles.device = "CPU"
except Exception:
    pass
```

Verified on this machine: `['NONE', 'CUDA', 'OPTIX', 'HIP', 'ONEAPI']`, with the RTX 4070
enumerating under both CUDA and OPTIX. Fix: enable OptiX in the addon preferences, set
`cycles.device = "GPU"`, keep a CPU fallback if no device is found. Also correct the README
and the `BLENDER_FACTS` string that currently *tells the LLM* Cycles is CPU-only — the model
has been actively steered away from the better renderer.

**2. A non-ASCII character kills an entire run.** From `run2.log`:
`UnicodeEncodeError: 'charmap' codec can't encode character '‑'`. The LLM emitted a
non-breaking hyphen, the cp1252 Windows console could not encode it, and `print()` at
`main.py:24` took down the run from `agent.py:128`. Fix: reconfigure stdout/stderr to UTF-8 at
startup and sanitise log strings. Trivial, and it destroyed a complete run — with Persian text
about to flow through these same log paths, it is not optional.

**3. A live API key sits in plaintext in `config.json`.** It will only get worse as more
services are added. Move secrets to an ignored `.env` or the OS keyring; keep `config.json`
for non-secret settings.

**4. `encode_video` produces a silent video.** Fine today, wrong for a film. Needs an audio
strip or an ffmpeg mux pass (§7-J).

---

## 16. Risks, honestly stated

| Risk | Severity | Mitigation |
|---|---|---|
| **Identity fitting from illustrations rather than photographs** — every fitter here was trained on photos | high, and the most likely disappointment | treat the reference image as the source of appearance, not proportions; ship SMPL-X/FLAME sliders with Shahnameh archetype presets from day one; report fit error rather than hiding it |
| **Armour and robes over a parametric body** — SMPL-X is a naked-body topology | high | Surface Deform / Mesh Deform binding plus corrective shape keys at shoulders and hips; accept some clipping in the miniature style where it is invisible |
| **Persian forced alignment quality** | medium | evaluate WhisperX-fa and MFA-fa on real recordings early, in M1, while it is still cheap to change route; keep the Whisper-word-timestamps + G2P fallback |
| **8 GB VRAM** | medium | strict one-model-per-subprocess discipline; SF3D as the default asset path; Hunyuan3D only with offloading and only for hero assets |
| **Render time** — 360 frames of photoreal Cycles is hours, and a full episode is ~20k frames | medium | OptiX; headless batch worker with persistent data; the miniature style as the fast path; decide the look on the M1 slice before committing to 20k frames of anything |
| **Creatures cannot be mocap-driven** | medium | stated plainly in the Consequence column of the rig table; hand-keyed and procedural motion budgeted as real work in M7 |
| **Voice conversion altering duration** | low but silent if it happens | hard gate on duration equality in F |
| **Monocular mocap jitter and foot skate** | medium | world-grounded model (GVHMR/TRAM), smoothing and foot-lock post-pass, velocity gates |
| **Scope** — the Shahnameh is enormous | high | the vertical slice discipline of M1; one shot working beats eight stages nearly working |

---

## 17. What is needed before M1 starts

Nothing blocking — the questions that would have changed the architecture are answered. These
are inputs to collect, and they can be gathered while M0 is built:

1. **A story excerpt** for the slice: a single moment where one character speaks one line.
   Rostam at the Helmand is a good candidate.
2. **A first `seed_prompt.md`**, including the Shahnameh style bible. This is the highest-
   leverage piece of writing in the project and it is worth iterating on deliberately.
3. **One performance take** — the user on camera, static camera, full body in frame, even
   lighting, plain background.
4. **One dialogue recording** of the same line, plus **one reference voice sample** for the
   character.
5. **Reference images** for that one character and that one environment, dropped into `refs/`.
6. **SF3D's missing `config.yaml`** — the 4 GB of weights in the repo are unusable without it.

Once those exist, M0 and M1 can be built and the first Rostam can speak.

