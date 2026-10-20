# Shahnameh Studio

A Persian story in, an animated film out. Characters and sets are built from
images you generate, voices from lines you record, and moves from the main
prompt's descriptions or from videos of you performing them. No primitives:
every character is a 3D model reconstructed from its image, every scene a 3D
set reconstructed from its image.

```bash
run.bat                     # the production window (main.py)
python main.py --studio     # the older single-shot studio
```

## The flow

1. **Seed prompt → strong model.** Press *Copy seed prompt* (or
   `python -m pipeline seed <project>`), paste it into a strong chat model
   (Claude Opus) and paste the Persian story after it. It returns two blocks:
   * the **main prompt** — production instructions in English, with the spoken
     lines in Persian inside « » (format: `pipeline/main_prompt.py`, worked
     example: `examples/keyumars/main_prompt.txt`);
   * the **image prompts** — per character a front prompt (an A-pose
     reference: full body, plain background, flat light) and a turnaround
     prompt (side and back on one canvas, made by *editing* the front image,
     so it stays the same character); one per scene (eye-level, wide).
     Generate those images yourself.
2. **Main prompt → Analyse.** Paste it into the window and press *Analyse*.
   The in-loop model, `nvidia/nemotron-3-ultra-550b-a55b` on NVIDIA's endpoint
   (key in `.env`), repairs the prompt if it does not follow the format —
   refusing any repair that changes a Persian line — and maps every free-text
   move onto the move vocabulary. Staging and cameras are solved by code
   (`pipeline/staging.py`), never by the model. Everything the model does is
   checked; see `pipeline/analysis.py`.
3. **The panel.** Five tabs:
   * **Characters** — an image (required) and a voice sample (optional) each,
     plus the narrator's sample when there is narration.
     Each image is checked as soon as it is attached (`pipeline/image_check.py`):
     arms clear of the body, legs apart, plain contrasting background, level
     camera, even light, a face large enough for the mouth pass. A red
     verdict means regenerate it before building.
   * **Scenes** — an image each.
   * **Sentences** — a recording of every line: record it in the window or
     choose a file. With the speaker's sample it is converted to their voice
     (Seed-VC); the mouth is synced to it. Narration is placed between lines,
     never over them, and moves no mouth.
   * **Moves** — each character's action. A video of someone performing it is
     optional: attached, it is retargeted onto the character (BlazePose);
     otherwise the move is built from the clips the analysis chose.
   * **Build** — look (stylized is the default), draft or final, progress, the
     film.

Attachments are filed where the pipeline's stages read them (the table is at
the top of `pipeline/intake.py`), so a GUI project builds from the command line
too: `python -m pipeline run <project> [--scale 0.5]`.

## What turns each input into film

| Input | Becomes | How |
| --- | --- | --- |
| character image | rigged, textured 3D character | Stable Fast 3D density sampled finely and meshed (`sf3d_dense.py`), the photo projected exactly through SF3D's camera, SF3D's albedo colour-matched to the photo for the unseen sides, BlazePose joints pushed through the mesh for the skeleton (`skeleton.py`), heat-weighted, mouth shape keys (`blender/photo_body.py`) |
| scene image | 3D set | Depth-Anything-V2 depth, metric from an eye-level ground fit, displaced grid with the image on it and a far backdrop (`scene_depth.py`) |
| move text | body take | clip vocabulary (`moves.py`) authored as limb directions with leg IK (`motion_library.py`) |
| move video | body take | BlazePose world landmarks per frame, retargeted as limb directions (`motion_video.py`) |
| recorded line | the character's voice, lip-synced | Seed-VC with the voice sample, length preserved to the sample (`voice_seedvc.py`), then `g_align` |

Model weights live under `D:\work\Shahnameh\models` and vendored source under
`D:\work\Shahnameh\vendor` (`pipeline/envs.py`).

---

# The single-shot studio (Blender AI Studio)

Type a prompt, pick **Image** or **Animation**, press Generate. The tool drives
Blender through the BlenderMCP addon, builds the scene, renders it, and saves
the result — while you watch it happen in the live viewport.

The scene-building model is `openai/gpt-oss-120b`, served from NVIDIA's
endpoint at `https://integrate.api.nvidia.com/v1`.

---

## Two programs live here

This repository holds two things that share a Blender helper library and very
little else.

**The studio** — `run.bat`, `main.py`, the GUI — is the one this README mostly
describes: type a prompt, an LLM writes Blender code, the host renders one image
or one short clip. Every run is disposable.

**The pipeline** — `python -m pipeline` — is a film pipeline over the same
Blender install: a staged graph of fourteen stages over a content-addressed
artifact store, which turns a story into rigged characters, aligned Persian
dialogue, assembled shots, rendered frames and a cut film. It is specified in
`PIPELINE_DESIGN.md` and its current state is recorded in
`PIPELINE_PROGRESS.md`.

The two overlap at `prelude.py`, the `bstudio` helper library, which both drive
Blender through.

---

## The pipeline

```bash
python -m pipeline doctor                       # what this machine can run
python -m pipeline init rostam_sohrab           # create a project directory
python -m pipeline seed rostam_sohrab           # write the Stage A seed prompt
python -m pipeline validate rostam_sohrab       # check production.json and inputs
python -m pipeline status rostam_sohrab         # what is cached, what is pending
python -m pipeline run rostam_sohrab            # walk the graph
```

A project is a directory under `projects/`. `projects/shahnameh/` is a worked
example: three characters, one environment, three shots and two spoken Persian
lines.

### The fourteen stages

| | Stage | What it does |
| --- | --- | --- |
| A | `a_production` | validate the hand-written `production.json`, resolve rig modes and the line index |
| B | `b_refs` | register whatever reference images were dropped into `refs/` |
| C | `c_face` `c_body` `c_texture` `c_garment` `c_rigged` | build, dress and publish each character |
| D | `d_env` | build the set and a primitive for each prop |
| E | `e_motion` | the body take for each performance |
| F | `f_voice` | convert each recorded line to its character's voice, keeping the timing |
| G | `g_align` | align the Persian audio and write the mouth curves |
| H | `h_assemble` | resolve each shot against the audio clock, compile `scene.blend` |
| J | `j_render` | render headless with OptiX, mux the dialogue, measure the sequence |
| K | `k_edit` | cut the shots together and write the RTL subtitles |

Every stage is a standalone CLI that reads files, writes files and exits. Every
artifact is keyed by a hash of the stage version, its inputs and its parameters,
so a re-run rebuilds only what actually changed — and the cache grain is the
*unit*, not the stage, so changing Rostam does not rebuild Sohrab.

### Two rules worth knowing before you use it

**The audio is the master clock.** A shot's length comes from its aligned
dialogue: `frames = ceil((speech_end − speech_start + head + tail) × fps)`. The
body take is then fitted into that window by a per-shot `retime` policy. A shot
with no dialogue has to declare `silent_duration_s` instead, because there is
otherwise nothing to derive a length from.

**A cache hit looks exactly like a passing stage.** One stage in this project
was broken for an unknown length of time behind a green run, because the
artifacts in the store had been built by older code. Before believing anything:

```bash
python -m pipeline run <project> --force all
```

### Layout

| File | Role |
| --- | --- |
| `pipeline/graph.py` | the stage graph: dependency order, caching, subprocess isolation, gates |
| `pipeline/store.py` | the content-addressed artifact store |
| `pipeline/envs.py` | which interpreter each stage runs in (§10) |
| `pipeline/gates.py` | the quality metrics and thresholds of §11 |
| `pipeline/validate.py` | JSON Schema plus the cross-references a schema cannot see |
| `pipeline/persian.py` | Persian grapheme-to-phoneme and the viseme map (§7-G) |
| `pipeline/project.py` | the project directory layout of §5.3 |
| `pipeline/schema/` | `production`, `shot`, `align`, `visemes`, `project`, `rig_overrides` |
| `pipeline/stages/` | one module per stage: the host half |
| `pipeline/blender/` | the Blender half of the stages that need one |
| `blender_ai_studio/styles/` | the three style packs of §8 |

### Useful flags

| Flag | What it does |
| --- | --- |
| `--to STAGE` / `--from STAGE` / `--only STAGE...` | bound the walk |
| `--force STAGE...` or `--force all` | rebuild even when cached |
| `--style miniature\|stylized\|photoreal` | pick the look (§8) |
| `--dry-run` | plan the work without doing it |
| `--keep-going` | do not stop at the first failing stage |

### Tests

```bash
python -m unittest discover -s tests
```

Two more checks need Blender and are **not** picked up by `unittest discover`:

```bash
blender -b --factory-startup --python tests/blender_styles_check.py
blender -b --factory-startup --python tests/blender_character_check.py
```

---

## The studio

Everything from here down is the single-shot studio.

### Quick start

1. **Get a free NVIDIA API key** at
   <https://build.nvidia.com/openai/gpt-oss-120b> (keys start with `nvapi-`).
2. **Launch:**

```bash
run.bat
```

3. Paste the key into the **NVIDIA key** field (it is saved to `.env`, which is
   gitignored — `config.json` holds only non-secret settings), type a prompt,
   choose Image or Animation, press **Generate**.

   You can also set `NVIDIA_API_KEY` in the environment instead; that takes
   precedence over `.env`. A key left in an older `config.json` is moved to
   `.env` automatically on the next start.

You do not need to start Blender yourself. If nothing is serving on the port,
the tool launches Blender, enables the BlenderMCP addon, turns on Poly Haven,
and starts the socket server. If Blender is *already* running with the addon,
the tool attaches to it instead.

#### Command line

```bash
python main.py --cli "a brass orrery on a walnut desk, morning light" --resolution 1280x720
```

```bash
python main.py --cli "a lighthouse in a storm" --mode animation --frames 96 --fps 24
```

---

### How it works

```
your prompt
    |
    v
[1] Director  -- gpt-oss-120b, high reasoning effort
    |          writes a numeric shot brief: subject, materials, lighting,
    |          camera, palette, motion beats, build order
    v
[2] Builder   -- gpt-oss-120b, tool calling
    |          builds the scene one step at a time through BlenderMCP,
    |          reading back measurements after each step
    v
[3] Quality gate -- the host re-checks the frame and pushes back
    |               (up to twice) if it is dark, blown out or badly framed
    v
[4] Auto-exposure, render, encode
    |
    v
outputs/<timestamp>_<prompt>/
```

**Why two stages.** A free 120B model given a one-line prompt produces grey
spheres on a black background. Given a brief that already specifies
"roughness 0.35, sun elevation 14°, camera at (7, -9, 3.5), 45 mm", it produces
a real image. Stage 1 exists to write that brief, and its system prompt is the
most load-bearing part of the project.

**The model cannot see.** `gpt-oss-120b` is text-only, so screenshots are for
*you*, not for it. Instead it gets:

- `inspect_scene` — every object, position, material, light and camera as text.
- `analyse_frame` — renders a 240 px thumbnail and measures it: mean and median
  luminance, contrast, fraction near-black, fraction blown out, and which
  objects are inside the camera frustum and how much of the frame they fill,
  plus plain-language hints about what to fix.

That turns "it looks wrong" into "92% of the frame is black and no object is in
the frustum", which a weak model can actually act on.

**The `bstudio` helper library.** `prelude.py` is injected into Blender at
startup as the module `bstudio`. It wraps the version-sensitive parts of the
Blender 5.x API into forgiving calls that take degrees, tuples and object
names. The model writes `B.procedural("hull", "rust")` instead of building a
seven-node shader graph, so there are far fewer ways for it to fail.

**Auto-exposure.** Before the final render the host measures the frame and
corrects the exposure in stops. This is non-destructive — it adjusts colour
management, not the lighting the model designed — and it removes the single
most common failure: a scene that is correct but three stops too bright.

---

### Output

Each run writes a folder under `outputs/`:

| File | What it is |
| --- | --- |
| `render.png` / `animation.mp4` | the final result |
| `poster.png` | a still from the middle of a clip |
| `frames/` | the individual PNG frames of an animation |
| `scene.blend` | the finished Blender project, saved before rendering |
| `scene_build.py` | just the Blender code the model wrote |
| `brief.md` | the art director's brief |
| `transcript.json` | the full model conversation |

Because `scene.blend` is saved before rendering, you can always open the scene
and re-render or keep working on it by hand.

---

### Settings

| Setting | Default | Notes |
| --- | --- | --- |
| Resolution | 1920×1080 | 960×540 is a good draft setting |
| Engine | EEVEE | Cycles renders on the GPU via **OptiX** (RTX 4070); EEVEE is still faster |
| Samples | 64 | EEVEE TAA samples, or Cycles samples |
| Frames / FPS | 96 / 24 | animation only — 96 frames is 4 seconds |
| Director effort | `high` | reasoning budget for the brief; quality matters most here |
| Builder effort | `medium` | reasoning budget per build step; `high` is better but noticeably slower across ~10 calls |
| Blender port | 9876 | must match the BlenderMCP addon's port |

Requests to NVIDIA are spaced **at least 1 second apart**, process-wide,
including retries — see `RateGate` in `llm.py`.

---

### Requirements

- Blender with the BlenderMCP addon installed and enabled
  (detected here: **Blender 5.2 LTS**, addon **v1.5**, protocol 4)
- Python with `PyQt6`, `openai`, `requests`, `pillow`
  (the `agent` conda environment has these)

The tool never modifies the BlenderMCP addon. It only speaks its socket
protocol and sets scene properties the addon already exposes.

---

### Layout

| File | Role |
| --- | --- |
| `main.py` | entry point (GUI or `--cli`) |
| `gui.py` | PyQt6 window, live viewport, worker threads |
| `runner.py` | orchestration: direct → build → expose → render → save |
| `agent.py` | the two stages and the quality gate |
| `prompts.py` | the director and builder prompts |
| `llm.py` | NVIDIA client, rate limiting, retries |
| `tools.py` | the six tools the model can call |
| `session.py` | scene inspection and the render/encode pipeline |
| `blender_link.py` | socket transport and Blender launcher |
| `prelude.py` | the `bstudio` helper library, injected into Blender |
| `bootstrap.py` | runs inside Blender at startup |

---

### Troubleshooting

**"Nothing is listening on 127.0.0.1:9876"** — open Blender, press `N` in the
3D viewport, find the **BlenderMCP** tab and click *Start Server*. Or just
close Blender and let the tool launch its own.

**"NVIDIA rejected the API key (401)"** — the key is wrong or expired. Keys
start with `nvapi-`.

**A render is black** — open `brief.md` and `scene_build.py` in the run folder
to see what was actually built. The quality gate catches most of these, but a
camera inside a wall can still slip through.

**Cycles is slower than EEVEE** — expected, but it does run on the GPU.
`B.set_engine("CYCLES")` enables OptiX on the RTX 4070 automatically. If it
prints `no GPU backend found`, check that the Cycles addon can see the card.
With 8 GB of VRAM, very heavy geometry or several 4k textures can exhaust it.
