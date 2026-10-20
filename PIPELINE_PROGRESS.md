# Pipeline Progress — state of the build

**Companion to:** `PIPELINE_DESIGN.md` (the spec; section numbers below refer to it)
**Date:** 2026-09-08 (second pass)
**Assessed by:** running the code, not by reading it. Every claim below was verified by
execution on this machine; where something was not run, it says so.

---

## 1. Headline

**M1 is complete. All fourteen stages of §12 exist, and the vertical slice runs end to
end: a Persian line is aligned, a shot is assembled against it, frames are rendered, the
dialogue is muxed in, and three shots are cut into a film with RTL subtitles.**

The measurement that matters, because §13 calls the vertical slice "the most important
process decision in the plan":

```
python -m pipeline run shahnameh --force all
14/14 stage(s) did work — 36 unit(s) built, 0 cached      (about 8 minutes)
```

Nothing was cached. Every artifact in that run was made by the code as it stands.

The slice has also been rendered in two of the three looks at full resolution and the
third at draft scale, which answers §13's look question with pictures — and, more
usefully than pictures, with a cost per frame (§3.4).

The project is now under version control, which it was not.

---

## 2. Verification performed

| What was run | Result |
|---|---|
| `python -m unittest discover -s tests` | **192 passed**, 4 s (was 24) |
| `python -m pipeline doctor` | all 5 interpreters resolve; jsonschema + Pillow present |
| `python -m pipeline status shahnameh` | **works** — all fourteen rows |
| `python -m pipeline run shahnameh --force all` | **36 units built from scratch, 0 failures** |
| `python -m pipeline run shahnameh --style stylized` | 12 built, 24 cached, 0 failures |
| `python -m pipeline run shahnameh --style photoreal --scale 0.5` | 4 built, 32 cached, 0 failures |
| `tests/blender_styles_check.py` (Blender 5.2) | 60/60 role materials; 3 compositor chains clean |
| `tests/blender_character_check.py` (Blender 5.2) | characters build, deform, speak, restyle; **all four retime policies measured** |
| OptiX probe on the built photoreal shot | `device=GPU`, `OPTIX`, RTX 4070 on, CPU off |
| MP4 container scan | `mp4a` + `soun` present in every shot and film file |

Timings are on the target machine (RTX 4070 Laptop, 8 GB; Blender 5.2; Python 3.12.4).

---

## 3. What was built this pass

### 3.1 Step 0 — the repository has a history

`git init`, and an initial commit capturing exactly the state the previous pass described,
so every change since is a diff against it. 128 files, 33k lines. `.gitignore` was already
correct. The two silent regressions the previous pass found would both have been visible
in a diff; that was the argument for doing this first and it still is.

### 3.2 `pipeline status` works, and is now the progress board

§5.3 offered two fixes and asked the next agent to choose knowingly. **Chosen: tolerate,
do not delete.** `STAGE_ORDER` is the declared shape of §12 and is worth more as a
fourteen-row plan with rows marked than as a list edited down to whatever exists this
week.

The distinction that mattered turned out not to be in either option. `ModuleNotFoundError`
means two different things — *this stage is unwritten* and *this stage is written and its
own imports are missing* — and `exc.name` is what separates them. Conflating them would
let a broken environment read as an unbuilt pipeline. `tests/test_stage_registry.py` pins
it, and pinning it found a second thing: `StageSpec` did `list(steps or [Step("host")])`,
so `steps=[]` silently became one host step.

`ModuleNotFoundError` joined the `except` clause at `cli.py`, as asked.

### 3.3 The five missing stages

| Stage | State | Notes |
|---|---|---|
| `f_voice` | built | §13's pass-through, on the same backend ladder Seed-VC will slot into |
| `g_align` | built | Persian G2P + energy alignment; new `pipeline/persian.py` |
| `h_assemble` | built | host resolves `shot.json`, Blender compiles `scene.blend` |
| `j_render` | built | headless batch, OptiX, persistent data, audio mux, both §11 metrics |
| `k_edit` | built | concat, `edit.json`, RTL `subs_fa.srt` |

Measured output of the full forced run:

```
f_voice     2 units   0.6s   passthrough, duration preserved to 0.0 ms
g_align     2 units   8.3s   sohrab_01: 5 words, 19 phones, 100% coverage, 60 frames
                             rostam_01: 8 words, 30 phones, 100% coverage, 92 frames
h_assemble  3 units  11.7s   s01 89 frames, s02 120, s03 96; all materials, feet, framing
j_render    3 units 427.6s   305 frames at 1920x1080, flicker 0.0005-0.0026
k_edit      1 unit   29.0s   305 frames, 12.71 s, 6.7 MB, 2 subtitle cues
```

`s01` is §5.2 to the letter: 2.48 s of measured speech + 0.5 head + 0.7 tail = 3.68 s =
89 frames. The audio strip is trimmed by the recording's own 0.45 s lead-in and starts on
frame 13; the mouth curves start on frame 13, computed by the same expression.

### 3.4 The three looks, and what each one costs

§13 says rendering the slice in all three styles is "nearly free once the pack layer
exists". Two of the three are. The third is not, and the number is the most decision-
relevant thing in this document.

All three films exist, from the same assets and the same `production.json`:

```
projects/shahnameh/film/miniature/final.mp4    6.7 MB   1920x1080
projects/shahnameh/film/stylized/final.mp4     4.8 MB   1920x1080
projects/shahnameh/film/photoreal/final.mp4    2.0 MB    960x540   (draft scale)
```

| Pack | Engine | Samples | Resolution | **s/frame** | The 305-frame slice |
|---|---|---|---|---|---|
| miniature | EEVEE | 24 | 1920×1080 | **1.40** | 7 min |
| stylized | EEVEE | 64 | 1920×1080 | **2.22** | 11 min |
| photoreal | Cycles / OptiX | 96 + motion blur | 960×540 | **6.67** | 34 min |
| photoreal | Cycles / OptiX | 96 + motion blur | 1920×1080 | **≈35** | **≈3 hours** |

The first three are whole-stage times divided by 305 frames, so they include the three
encodes and the measurement pass — slightly pessimistic, and the same basis throughout.
Pure render time for the photoreal draft was 4.4 s/frame.

The 1080p photoreal figure was measured over 20 frames and the run then **stopped
deliberately** rather than carried to completion: three hours for twelve seconds of film
is itself the finding. OptiX was confirmed enabled by direct probe of the built scene
(`device=GPU`, `OPTIX`, RTX 4070 on, CPU off), so this is the GPU number, not a CPU
fallback.

§16 asks the project to "decide the look on the M1 slice before committing to 20k frames
of anything". On the same basis, a 20,000-frame episode:

- **miniature: about 8 hours**
- **stylized: about 12 hours**
- **photoreal: about 194 hours** at 1080p, about 37 hours at 960×540

That is a 25× spread, and it is a constraint on the shape of the project rather than a
detail. Motion blur is a large part of it — photoreal is the only pack that declares it —
and turning it off is the first thing to try if the photoreal look is the one wanted.

`pipeline run --scale` was added for exactly this: a draft resolution that **enters the
artifact key**, so a 960×540 draft can never be served from the cache as a 1080p final.

One thing the three renders showed that no single render could: the §11 lip-sync metric
tracks the viseme profile exactly as predicted in §3.6, and passes on all three.

| Pack | viseme profile | s01 | s02 |
|---|---|---|---|
| miniature | `graphic`, 2-frame hold | r=0.34 lag +1 | r=0.24 lag −2 |
| stylized | `smooth` | — | r=0.37 lag −1 |
| photoreal | `smooth`, 2-frame filter | r=0.46 **lag 0** | r=0.37 lag −1 |

The correlation is lowest exactly where §8 asks the mouth to snap rather than
co-articulate. A fixed threshold would have failed the miniature pack for doing what the
design tells it to do; the control comparison passes all three on the same rule.

### 3.5 `apply_motion`: all four retime policies

§4.2 recorded that `shot.schema.json` declares four policies and `prelude.py` implemented
two, with `trim` and `timewarp` falling through to `hold_last_pose` in silence. All four
now exist, an unknown policy raises, and the policy is a table (`_retime_plan`) returning
one entry per output frame rather than a branch inside a loop over bones.

Measured in Blender on a 12-frame monotonic take fitted into a 24-frame window — one
number per frame, saying which source frame it came from:

```
loop            f1=0.0  f12=51.6  f13=0.0   f24=51.6   the cycle restarts
hold_last_pose  f1=0.0  f12=51.6  f18=51.6  f24=51.6   last key written at frame 24
timewarp        f1=0.0  f12=24.7  f13=26.9  f24=51.6   linear, ends on the take's own pose
trim            last key at frame 12; a 40-frame take reports 16 frames discarded
```

`timewarp` slerps between neighbouring poses. A component-wise blend of two rotations
passes through the inside of the sphere and shortens every limb on the way.

### 3.6 The two new §11 metrics, and a rebuilt one

**Temporal flicker** is wired and passing with room to spare: 0.0005 (miniature) and
0.0004 (stylized) against the miniature pack's 0.020 warn limit. The packs' film grain is
a static noise plate rather than a per-frame one, so it costs this metric nothing.

**Lip-sync score** is wired, and had to be rebuilt to be worth wiring. It fired on `s02`
at r=0.24 against the 0.30 threshold that was guessed before the metric had ever been run.
So it was measured instead of argued about:

| | r at zero lag |
|---|---|
| correct mouth, no hold | 0.35 – 0.43, argmax exactly at lag 0 |
| correct mouth, §8 snap profile (`hold_frames: 2`) | 0.24 – 0.34 |
| the **wrong line's** audio | −0.14 – 0.00 |
| the same curve **shifted four frames** | 0.04 – 0.16 |

An absolute threshold cannot separate those: a correct held mouth scores below 0.30 and a
shifted one can score above 0.15. Two things are going on and neither is a bug — a jaw
driven by *phoneme shapes* is only loosely tied to loudness, and §8's snap profile then
quantises the curve to every third frame.

So `lip_sync_score` now also correlates the jaw against eight **rotated** copies of the
same envelope — a permutation test at p<1/9 — and the real placement has to beat the best
of them. Measured margins: correct **+0.07 to +0.23**, wrong line **−0.17 to −0.44**,
shifted-by-four **−0.11 to −0.27**. Clean separation, no magic number, and it
self-calibrates across viseme profiles: the smooth-profile stylized pack scores +0.23
where the held miniature pack scores +0.07, and both pass on the same rule.

Both ends of the comparison are artifacts, which is what makes it a check on the pipeline
rather than on arithmetic done twice: the jaw is sampled per frame off the **evaluated
shape key in the built scene**, and the envelope comes off the **WAV that was muxed**.

### 3.7 §15-4 is finally exercised

`encode_video` produced silent video; the fix was written and never run because nothing
called it. Every MP4 this pipeline writes now carries an `mp4a`/`soun` track, verified by
scanning the containers. `gates.check_render` fails a shot whose spoken line is missing
from its video *or muxed on the wrong frame*; both lines landed on frame 13, which is the
frame the shot asked for.

### 3.8 Persian, as a module

`pipeline/persian.py` is new: grapheme-to-phoneme, a twelve-viseme map, per-phoneme
duration weights, and the emotion table §7-G makes the only source of facial expression.

Persian does not write its short vowels, so a G2P over the script has to guess them. Two
rules from the (C)V(C)(C) syllable template recover most of them — a word never opens with
two consonants, and a run of three is a syllable boundary — which turns the spelling
`r-s-t-m` into `r-e-s-t-e-m` and `p-h-l-v-a-n` into `p-e-h-l-e-v-a-n`. The right number of
syllables in the right places, with the vowel *qualities* wrong. That trade is the correct
one here: syllable count is what a mouth shows.

Everything from this route is flagged `approximate`, which `gates.check_align` turns into
a warning on **every run**. That noise is deliberate; §16 rates Persian alignment a medium
risk and asks for it to be evaluated early.

### 3.9 Tests: 24 → 192

The previous pass recorded that coverage stopped at the infrastructure and that the
`e_motion` bug lived in exactly that gap. That gap is now covered:

| File | What it pins |
|---|---|
| `test_m0.py` | store, keys, graph (unchanged, 24) |
| `test_stage_registry.py` | unbuilt vs unrunnable stages; the fourteen are declared once |
| `test_persian.py` | phonotactics: no onset clusters, no three-consonant runs, word-final *heh* is /e/, the names keep their syllable counts |
| `test_gates.py` | 523 previously untested lines, against synthesised audio with known answers, both sides of every threshold |
| `test_validate.py` | cross-references, pointer accuracy, rig-mode resolution order |
| `test_h_assemble.py` | §5.2 to the frame, audio placement under both lead policies, anchors across both rig topologies, camera compression |
| `test_edit_and_voice.py` | SRT timecodes, the voice backend ladder, both gates |

Four real bugs were found by writing them, listed in §5.

---

## 4. What is not done

### 4.1 The milestones after M1

M2–M7 are untouched, which is the plan: §13's whole argument is that every remaining
milestone is now a *substitution into a working pipeline* rather than a step toward one.
The seams are in place and named:

| Milestone | The seam it substitutes into |
|---|---|
| M2 real identity | `c_face` / `c_body` write the same `face.json` / `body.json` |
| M3 real motion | `e_motion` writes the same `body.json`; `axes=` already handles SMPL's frame |
| M4 real voice and mouth | `f_voice._try_seedvc`, `g_align._try_whisperx` / `_try_wav2vec2` — each raises with the reason it cannot run today |
| M5 assets and environments | `d_env` strategies; SF3D still has no `config.yaml` |
| M6 multi-shot | works already for three shots; continuity across shots is untested |
| M7 creatures | `c_rigged` builds Rakhsh; creature motion is still the idle stub |

### 4.2 Gaps and rough edges

| Item | Detail |
|---|---|
| **The frustum gate is per shot, not per character** | `gates.check_assembly` asks whether *any* cast member is in frame. In `s01` Rostam is entirely off camera and the shot passes. That may be the right shot — a two-hander with one actor out of frame is a real choice — but the gate cannot tell that from a mistake. The fix is small and shaped: have `_common.framing` return membership per object, and `build_shot` report a `characters_out_of_frame` dict that the gate **warns** on. Left undone because turning it into a failure would fail a legitimately authored shot, and which it should be is a decision to make with the shot list in front of you. |
| **`status` over-reports after a pack switch** | It plans against whatever is currently *materialised*, and g_align publishes `visemes.json` to one path per line whichever profile built it. So after `run --style photoreal`, `status` shows h_assemble, j_render and k_edit as pending for miniature, and they are not: a `run --dry-run` re-materialises the cached graphic curves and reports 36/36 cached. Use `run --dry-run` for an accurate plan. The real fix is for planning to resolve inputs from the store rather than from disk. |
| **The store still never garbage-collects** | Flagged last pass at 7.8 MB; it is now **1.9 GB**, because frames are artifacts and there are 305 of them per style. This is the point at which it starts to matter. A `pipeline gc` that keeps the newest key per (stage, unit) and drops the rest would recover most of it. |
| **Frames are published per style, `scene.blend` is not** | `j_render` writes `shots/<id>/<style>/`, extending §5.3, because the three looks would otherwise overwrite each other. `h_assemble` still writes `shots/<id>/scene.blend`, so whichever pack ran last is the one on disk. Harmless — it rebuilds in seconds and the store holds all three — but it is an inconsistency someone will trip over. |
| **Photoreal has never been rendered at full resolution** | Deliberate; see §3.4. The film exists at 960×540. Three hours of GPU time buys the same twelve seconds at 1080p, and that is a decision rather than a task. |
| **Rig checkbox table (§9) not built** | Unchanged. `gui.py` is still the old studio window; `rig_overrides.json` is honoured by `validate.resolve_rig_mode` and the resolution order works — only the UI is missing. |
| **No real aligner** | WhisperX and a Persian wav2vec2 are both unavailable here (probed: no `whisper`, no `whisperx`; `torch` and `transformers` present). The `wav2vec2` rung is deliberately unreachable from `auto` — a stage that downloads a gigabyte mid-run will do it on a machine that is offline. |
| **Missing inputs (§17)** | `refs/` empty, `performance/` empty, no `story/story.md`, no reference voices in `audio/voices/`, SF3D still has no `config.yaml`. All user-supplied inputs, not code. |
| **Blender checks still not in `unittest discover`** | Unchanged, and unchangeable without a Blender-in-CI story. Run them by hand; `blender_character_check.py` now also covers the retime policies. |

---

## 5. Defects found by running the code

All four were found by tests or by execution, none by reading.

### 5.1 `persian._spell` did not normalise — **FIXED**

`words()` folded the Arabic and Persian *yeh* to one letter; `g2p_word()`, which is public
and reachable directly, did not. So two spellings that look identical and that `FOLD`
exists to reconcile produced different phonemes. Found by `test_persian.py` on its second
assertion.

### 5.2 `_timecode` overflowed its milliseconds field — **FIXED**

`k_edit` split the seconds first and rounded the remainder afterwards, so 1.9996 s wrote
`00:00:01,1000` — four digits in a three-digit SRT field, which a player will either
reject or read as a completely different time. Now rounded to milliseconds first and
decomposed with integer division. Found by `test_edit_and_voice.py` before any subtitle
reached a player.

### 5.3 `prelude.reset()` leaves collections behind — **FIXED**

`reset()` deletes the objects but not the collections, armatures, actions and shape keys
behind them. Every stage before `h_assemble` builds from primitives, so nothing had ever
hit it; `h_assemble` *appends*, and the second shot in one Blender process could not link
an env collection the first had already linked. `_common.new_file()` reads factory
settings empty. §10 rule 1 requires one process per stage, so this had to be solved rather
than avoided.

### 5.4 `StageSpec` could not express "no steps" — **FIXED**

`list(steps or [Step("host")])` made `steps=[]` mean one host step. Only visible once
something needed to declare a stage that runs nothing.

### 5.5 A test that depended on another test's side effects — **FIXED**

`blender_character_check.check_garment` measured between frames 1 and 12 of whatever
action `check_motion` happened to leave on the rig. Adding the retime checks, which must
clear the rig's animation, made a perfectly bound garment report zero movement. It now
keys its own pose. Worth recording because the failure *looked* like a garment regression
and was not one.

---

## 6. What to do next, in order

### Step 1 — Decide the look. Half an hour, no code.

Watch `projects/shahnameh/film/miniature/final.mp4` and
`film/stylized/final.mp4`, then `film/photoreal/final.mp4` at draft scale. §3.4 gives the
cost of each. This is the decision M1 was built to enable and everything below is cheaper
once it is made.

If photoreal is the answer, the first experiment is `motion_blur: false` in its
`style.json` — it is the only pack that declares it and Cycles charges heavily for it.

### Step 2 — Garbage-collect the store. An hour.

2 GB for twelve seconds of film. A `pipeline gc` keeping the newest key per (stage, unit)
would recover most of it, and it is much easier to write now than at episode scale.

### Step 3 — Real alignment (M4's cheap half). This is the highest-value substitution.

Everything else in the film is a stub that looks like a stub. The mouth is the one place
where a stub reads as a *defect* rather than as roughness, and §13 already calls G "the
one real ML piece, because it is cheap and it is what makes the shot read as a character
speaking". The seam is written: `g_align._try_whisperx` and `_try_wav2vec2` raise with the
reason they cannot run, and `align.approximate` already tells every downstream consumer
what it is looking at.

Evaluate WhisperX-fa and MFA-fa against the two recordings that are already here. §16 asks
for this "early, in M1, while it is still cheap to change route" — it is still cheap.

### Step 4 — Then the rest of M2–M7, in §13's order.

Each is a substitution into a graph that runs. Re-run with `--force` after each one.

### Worth doing alongside

- The per-character frustum warning (§4.2). Small, shaped, and it would have told you
  Rostam is off camera in `s01` instead of leaving you to notice.
- The §9 rig checkbox table, whenever the GUI is next touched.
- A `--force all` smoke run in whatever passes for CI. A cached green run still proves
  nothing about current code, and that has not changed.

---

## 7. Summary table

| Milestone | State |
|---|---|
| **M0** Foundations, verified fixes | **Complete** |
| **M1** Vertical slice, ML stubbed | **Complete** — fourteen stages, 36 units from scratch, 0 failures; a shot renders, speaks and cuts |
| **M2** Real identity | Not started (presets in place as the M1 stub) |
| **M3** Real motion | Not started (authored idle takes in place; all four retime policies now real) |
| **M4** Real voice and mouth | Not started — **both seams written and reporting why they are empty** |
| **M5** Assets and environments | Not started (`d_env` builds all three strategies; SF3D still lacks `config.yaml`) |
| **M6** Multi-shot | Three shots cut together with subtitles; continuity across shots untested |
| **M7** Creatures | `c_rigged` path builds Rakhsh; creature motion is still the idle stub |
| **§8** Style packs | **Complete** — all three rendered, and now costed |
| **§11** Quality gates | **Complete** — every stage gated; both new metrics wired, one rebuilt against measurement |
| **§12** `prelude.py` character layer | **Complete** — six functions, four retime policies |
| **§15** The four bugs | **All four fixed and all four now exercised** |

The pipeline runs end to end and every stage is measured rather than believed. The single
biggest risk from the last pass — *a cache hit looks exactly like a passing stage* — is
unchanged and still the thing to be careful about: `--force all` before believing
anything. The new risk is different in kind and stated in §3.4: **the photoreal look costs
25× the miniature one**, and that is a choice about the shape of the project rather than a
bug to be fixed.
