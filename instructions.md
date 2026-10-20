# Instructions for the next session

**Written:** 2026-09-27, **extended 2026-09-28** (section 0)
**Repo:** `D:\work\Shahnameh\blender`, branch `feature/main-prompt-production`
**Supersedes** the 2026-09-09 version of this file, which described a pipeline
that built its film out of welded primitives. `PIPELINE_DESIGN.md` is still the
spec; `README.md` now leads with the flow below.

---

## 0. 2026-09-28: learned motion, real mouths, grounded staging

The user watched the first film and named three problems: mouths do not open,
movement is unnatural, placement is unnatural. They downloaded MuseTalk,
GVHMR, MoMask, SMPL and SMPL-X (Wan VACE for a final repaint pass comes
later) and asked for fix → rebuild → review loops until the film is as good as
it gets before Wan.

**Machine layout.** Code in `D:\work\Shahnameh\vendor\{MuseTalk,GVHMR,
momask-codes}`, weights in `models\{musetalk,gvhmr,momask,body_models,clip}`,
reached from the repos through junctions. One venv, `D:\work\Shahnameh\envs\
motion`, layered on the comfyui env (`--system-site-packages`); its `.pth` adds
`vendor\shims_motion` (pure-Python `pytorch3d.transforms` + a brute-force knn,
because pytorch3d does not build here) and GVHMR. Logical env `motion` in
`envs.py` (with the base env's DLL dirs on PATH). Vendor patches: MoMask's
numpy-2 incompatibilities; MuseTalk's Whisper weights converted to
safetensors (transformers refuses `torch.load` under torch 2.5).

| Piece | Module | Notes |
|---|---|---|
| SMPL joints → take | `motion_smpl.py` | shared by GVHMR and MoMask; proper rotation + left/right swap (see below); `loop()` cuts seamless cycles |
| text → motion | `motion_text.py` (MoMask) | "a person …", names → "someone", several samples, the steadiest kept; `in_place` for moves with no travel verb |
| video → motion | `mocap_gvhmr.py` | GVHMR demo minus renderer and DPVO, static camera, video re-timed to 30 fps |
| backend choice | `motion_learned.py`, `stages/e_motion.py` v3 | human actions + talk/fight cycles learned; idle and animals authored; fallbacks are named in the take's `backend` |
| mouths | `lipsync_musetalk.py`, `stages/j_mouth.py`, `blender/face_track.py` | new stage between j_render and k_edit; face box from projected mesh landmarks, not detectors |
| colours of unseen surface | `photo_body.fill_hidden` | photo colours diffused over the mesh; no more grey limbs |
| fused limbs | `photo_body.confine_limbs` | capsule rule: limb weight only within a limb radius of its bones, behind-the-arm distance ×2.5 (wings) |
| staging | `staging.FEATURED_TURN_DEG`, `h_assemble._parallax_budget`, `build_shot._follow_terrain/_flat_spot/_fit_frame/_contact_shadow` | see decisions |

**Decisions (2026-09-28).**
* **The rig's `left_*` bones are the character's anatomical right** (checked on
  Siamak's sash). Both retargets used to mirror performances; now a
  performer's right hand drives the character's right hand.
* **Image-set cameras stay near the photograph's viewpoint** (≤35% of the way
  to the stage, ±0.7 m in height) and frame with the lens. Parallax is what
  made the radial smears; a telephoto is the honest way to shoot a backdrop.
* **Featured characters (camera subject, speakers) are cheated to ≤35° from
  the camera**: an image set cannot be shot from the reverse angle.
* **Talk cycle arms damped 60% toward the hanging pose**: MoMask's talkers
  open their arms wide, which also drags fused sleeves and wings.
* **Contact shadows are dithered, not blended**: the packs' compositor
  rebuilds the frame from passes, and blended surfaces are not in them.
* **Terrain following never sinks** (max of smoothed and raw lift, over a
  25 cm band).

**Input images are the biggest lever left.** The Keyumars images have busy
backgrounds, arms touching bodies, robes hiding legs, a painted angel with side
light, a black div on dark grey. The seed prompt's character framing now asks
for an A-pose reference (arms 35° out, open hands, level chest-height camera,
flat light, plain background, white for dark characters, photographic not
painted, no weapons or flying cloth), and the Characters tab checks each image
on attach (`pipeline/image_check.py`, cached as `<image>.check.json`).

**Known limits.** SF3D hands are blobs; wide arm moves on robed or winged
characters still stretch fused cloth; MuseTalk needs ~100 px faces, so draft
renders (0.5 scale) give soft mouths — judge mouths at full resolution. GVHMR
is verified on a rendered performance (joints 9-18° median error, travel and
kneel depth close), not yet on a phone video.

---

## 1. What the user asked for, and what now exists

The user's workflow (their words, condensed): *seed prompt + Persian story →
a strong LLM → "main prompt" + image prompts → the GUI takes the main prompt →
the in-loop model analyses it → a panel of Characters (image + voice sample),
Scenes (image), Sentences (a recorded voice per line, converted to the
character's voice and lip-synced; narrator lines too, no lips, never
overlapping), Moves (text; optional video of the user doing it) → film.*
Two hard requirements: **no cubes or cylinders on screen**, and the in-loop
model is **`nvidia/nemotron-3-ultra-550b-a55b`** (NVIDIA NIM, key in `.env`),
which must work **in English** — only the story and spoken lines are Persian.

All of it is built and was run end to end on `projects/keyumars` (made from
`examples/keyumars/main_prompt.txt`):

| Piece | Module | Status |
|---|---|---|
| seed prompt v2 (main prompt + image prompts framed for 3D) | `pipeline/seed.py` | done |
| main prompt grammar + parser | `pipeline/main_prompt.py` | done, tested |
| analysis: repair + move classification by Nemotron, checked | `pipeline/analysis.py` | done, tested with a scripted model and live |
| staging: words → metres, cameras | `pipeline/staging.py` | done, tested |
| intake: panel files → project layout | `pipeline/intake.py` | done, tested |
| characters from photos (dense SF3D, photo projection, albedo colour match, BlazePose skeleton) | `sf3d_dense.py`, `skeleton.py`, `pose_mp.py`, `stages/c_asset.py` v3, `blender/photo_body.py` | done; all six Keyumars characters rig and pose |
| scenes from images (Depth-Anything-V2 → displaced set + backdrop) | `scene_depth.py`, `stages/d_env.py` v2 | done |
| move library + leg IK | `moves.py`, `motion_library.py`, `stages/e_motion.py` v2 | done |
| move video → take | `motion_video.py` | written; see §3 |
| voice conversion | `voice_seedvc.py`, `stages/f_voice.py` v2 | done: similarity 0.01 → 0.76, duration kept to 0.02 ms |
| narration | validate / h_assemble / j_render | narrator lines timed with dialogue, no mouth, no lip-sync warning |
| production window | `blender_ai_studio/production_gui.py` (`main.py` default; `--studio` for the old one) | done |

## 2. Decisions made this session (do not re-litigate)

* **Main prompt is a line grammar, not JSON** — editable by a person, parsed
  deterministically; the model repairs, it does not transcribe. A repair that
  changes any «Persian» line is rejected.
* **Staging is vocabulary** (`left, back, facing siamak`; `medium on x,
  low_angle, slow push_in`) solved by code. Characters facing upstage are
  "cheated" to ≤75° from the camera.
* **`facing_deg` convention:** CCW about +Z on a body facing +Y at rest — 0 is
  +Y, **90 is −X**, 180 faces the camera. The schema text used to say +X; the
  code never did. Fixed in the schema.
* **Rig sides are by position:** `left_*` bones are on +X (prelude's
  convention, the mirror of anatomy). `skeleton.py` names BlazePose pairs by X.
* **Rest correction** (`rig["rest_correction"]`, applied in
  `prelude._pose_locals`) covers **arms only**. Correcting legs straightened the
  div's bent legs and sank its foot 0.42 m; correcting feet drove claws into
  the ground.
* **Image sets:** ground at z=0 under the stage by construction; cameras never
  behind the image's own viewpoint; no miniature lens compression; cast dropped
  onto terrain by footprint; ground contact measured over the footprint.
* **Default look for new projects is `stylized`** — miniature's ink outline
  tore photo content into black blotches.
* **Downloads approved by the user (2026-09-23):** Seed-VC + checkpoints,
  Depth-Anything-V2-Small, MediaPipe/BlazePose. Google's MediaPipe bucket is
  geo-blocked here; BlazePose comes as ONNX from Hugging Face `opencv/*`, in
  `D:\work\Shahnameh\models\mediapipe`. The `pose` conda env was created but is
  not needed (cv2 in `comfyui` runs the ONNX).
* **Generated asset binaries are no longer tracked** (`.gitignore`): a photo
  character is ~26 MB across its stage folders. JSON manifests still are.

## 3. Open / next

* **Move-video retarget: tested on a rendered performance, not yet on a phone
  video.** Siamak rendered doing celebrate→kneel, 85 frames, all detected;
  arms within 7-8° at rest and 16-36° with arms raised; the kneel's hip drop
  reads 0.20 m against 0.42 m (BlazePose's depth under-reads folded legs).
  Recognisable, not precise.
* **Fight takes** still peak ~55-60°/frame (fast strikes); Siamak's fall
  slides his feet (11 cm/frame while planted). The skate metric now counts
  only planted feet, so steps no longer read as skating.
* **Walking characters leave the camera's aim**: the aim is the start mark.
* **Lip sync on photo bodies** is shape keys around the mouth landmark plus a
  darkening lip line; there is no mouth cavity.
* **g_align still uses the envelope fallback** (no forced aligner).
* The `shahnameh` project still holds the old production.json with props,
  which are primitives; the new flow emits no props (weapons are in the
  character images).

## 4. How to work here

* **Interpreters:** host = `C:\Users\Asus\anaconda3\python.exe` (a bare
  `python` on PATH may be a 3.13 without jsonschema); GUI = `envs\agent`;
  ml/audio = `envs\comfyui`. `envs.step_env` now puts an env's `bin` and
  `Library\bin` on PATH (activation is skipped, and nvrtc lives there).
* **Always `--force`** a stage you changed; a cache hit looks like a pass.
* **Big patches**: write a script to the scratchpad and run it; heredocs over
  ~150 lines break here.
* **Look at frames**, not just gates: every visual bug this session (cream
  characters, ink blotches, sunk feet) passed some check first.
* **Tests:** `python -m unittest discover -s tests` (273). Blender checks by
  hand: `tests/blender_character_check.py`, `tests/blender_styles_check.py`,
  `tests/blender_photo_body_check.py <project> <character> <out_dir>`.
