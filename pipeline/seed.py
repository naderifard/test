"""The seed prompt: what the strong chat model is told before it reads the story.

The user pastes this prompt and a Persian story into a strong model (Opus). It
returns two things:

1. **The main prompt** -- the production instructions, in the line grammar of
   ``main_prompt.FORMAT``. English, with the spoken lines in Persian. This is
   the first thing the GUI takes; the in-loop model (Nemotron) analyses it.
2. **The image prompts** -- one per character and one per scene. The user
   generates those images by hand and attaches them in the panel. Each image
   becomes geometry: a character image goes through Stable Fast 3D and is
   rigged, a scene image goes through a depth model and becomes the set. So the
   image prompts are written for *that*, not for looking nice on their own --
   which is the whole reason this file dictates their framing.

**Generated, not typed.** The grammar and the vocabularies are read out of
``main_prompt`` and ``staging``, so a change there changes the prompt, and the
prompt can never ask for a word the parser does not know.

**The style bible stays.** §7-A asks for the Shahnameh's cultural detail to be
specified once rather than re-improvised per shot, and it is prose because
that is what the receiving model consumes.
"""

from __future__ import annotations

from . import main_prompt, moves, staging

STYLE_BIBLE = """\
## The style bible — Shahnameh, and what its world looks like

Fixed reference. Follow it rather than reinventing it.

**Period and dress.** The heroes are pre-Islamic Iranian: read them as Sassanid
(224–651 CE) with Achaemenid echoes for the older, mythic generations. Men of
rank wear a long belted tunic (*qaba*) over loose trousers gathered at the ankle,
a broad sash and soft boots. Warriors add lamellar or scale armour, a conical or
ridge helmet with a mail aventail, vambraces and a round shield. Kings wear a
domed crown with ribbon streamers. Women wear layered robes with a light head
veil, never a modern hijab silhouette. The first mythic kings (Keyumars and his
line) wear leopard skins, not tailored cloth.

**Weapons.** The *gorz* (ox-headed mace) is Rostam's. Also right: a slightly
curved sword, a composite recurve bow with a quiver at the hip, a long lance,
the *kamand* (lasso).

**The cast, canonically.** Rostam (رستم) is vast and bearded with a
leopard-skin over his armour and a tiger-skin helm. Sohrab (سهراب) is a
beardless, immense youth. Rakhsh (رخش) is a chestnut stallion dappled like
"rose leaves on saffron". Zal (زال) is white-haired from birth. The Simorgh
(سیمرغ) is a vast peacock-plumed bird, not an eagle. Divs are horned, shaggy
demons half again as tall as a man.

**Tone.** Ferdowsi is grave and economical. Dialogue is spoken and weighty, not
archaic for its own sake. Keep each line short enough to say in one breath: the
recorded audio sets each shot's length, and a 40-word line is a 20-second shot.
"""

IMAGE_STYLE = ("photorealistic, cinematic, highly detailed, natural colours, "
               "sharp focus")
# A character image becomes a textured 3D model, so it is a *reference*, not a
# picture: every word below exists because an image without it failed. The
# first Keyumars images taught most of them -- a low camera, hard sunlight
# baked into the skin, arms and hands touching the body (fused limbs), an oil
# painting's face that MuseTalk could not animate, wings behind the sleeves,
# a black div on a dark background.
CHARACTER_STYLE = ("photorealistic, like a studio photograph or a high-end 3D "
                   "character scan, natural colours, sharp focus, highly "
                   "detailed, not a painting, not an illustration")

CHARACTER_FRAMING = (
    "full-body character reference, front view, standing in a relaxed A-pose "
    "with both arms straight and angled about 35 degrees out from the torso, "
    "a clear gap of background between each arm and the body, hands open and "
    "relaxed with the fingers visible and touching nothing, feet "
    "shoulder-width apart and flat on the ground, both legs visible down to "
    "the ankles, facing the viewer straight on with the head level, neutral "
    "expression with the mouth closed, camera at chest height looking "
    "straight at the figure with a long lens, no perspective distortion, the "
    "whole figure centred with a margin above the head and below the feet, "
    "nothing overlapping the outline of the body, no floating cloth, "
    "ribbons or loose strands, plain seamless light grey studio background, "
    "soft flat even lighting from the front, no cast shadows, no rim light, "
    "nobody and nothing else in the image, vertical 2:3 portrait")
ANIMAL_FRAMING = (
    "full-body animal reference, facing the viewer head-on, standing square "
    "on all four legs with the legs apart and all four feet visible, mouth "
    "closed, tail hanging clear of the legs, camera at the animal's chest "
    "height looking straight at it, the whole animal centred with a margin "
    "around it, plain seamless light grey studio background, soft flat even "
    "lighting from the front, no cast shadows, nothing else in the image, "
    "vertical 2:3 portrait")
# A dark character on the grey backdrop is a silhouette the matting cannot
# find and a texture with no detail (the black div): these words replace the
# backdrop and the light for any character whose look is dark.
DARK_FRAMING = ("plain seamless white studio background, bright soft even "
                "lighting that shows every detail of the dark surface")
# And the mirror case: a white robe on the light grey backdrop is the same
# colour as the backdrop to within a level (Sorush), and the outline is gone.
# The second image of every character: its side and back, made by editing the
# front image rather than generating from words, so it is the same character.
# Both views on one canvas, because an image model keeps two views in one
# picture far more alike than two separate generations; the front stays its
# own full-resolution image, because the face and most of the texture come
# from it. {subject} and {backdrop} are filled per character.
TURNAROUND_PROMPT = (
    "Using the attached {subject} exactly as it is, create a turnaround "
    "sheet with two full-body views side by side on one wide canvas: on the "
    "left, its left side in strict profile; on the right, its back, seen "
    "straight from behind. The same {subject}: same face and head, same "
    "clothing, colours and patterns, same proportions and height, same pose "
    "as in the attached image, {backdrop}, same soft flat lighting with no "
    "shadows, camera at chest height, no perspective distortion, both "
    "figures the same size and standing on the same ground line, the whole "
    "body visible with a margin in both views. Do not add, remove or "
    "restyle anything; nothing else in the image, horizontal 3:2 canvas")
TURNAROUND_BACKDROP = "same plain light grey background"
# An animal's shape is in its length, which a head-on front image hides: the
# side profile carries most of it and gets most of the canvas, the rear view
# the rest. "Back" and "A-pose" mean nothing useful for a lion (the first
# turnaround prompt, written for people, did not work for one).
ANIMAL_TURNAROUND_PROMPT = (
    "Using the attached {subject} exactly as it is, create a turnaround "
    "sheet on one wide canvas with two full-body views of the same animal: "
    "on the left, taking about two thirds of the width, its left side in "
    "strict profile, standing square on all four legs with the legs "
    "slightly apart so all four are visible, head level and looking "
    "straight ahead, tail hanging clear of the legs; on the right, taking "
    "the remaining third, the same animal seen straight from behind, "
    "standing square with the tail hanging down. Same {subject}: same fur, "
    "colours and markings, same proportions and size, {backdrop}, same soft "
    "flat lighting with no shadows, camera at the animal's shoulder height, "
    "no perspective distortion, both views the same scale and standing on "
    "the same ground line, the whole animal visible with a margin in both "
    "views. Do not add, remove or restyle anything; nothing else in the "
    "image, horizontal 3:2 canvas")

PALE_FRAMING = ("plain seamless mid-grey studio background, darker than the "
                "clothing, soft even lighting")
SCENE_FRAMING = (
    "wide establishing view taken at human eye level (camera about 1.6 m above "
    "the ground), camera level so the horizon is straight across the middle of "
    "the frame, open ground in the lower half of the frame where people could "
    "stand, no people, no animals, no text, horizontal 16:9 landscape")


def _join(words):
    return ", ".join(f"`{w}`" for w in words)


def build_seed_prompt(project=None):
    """The whole seed prompt. ``project`` is accepted for the CLI and unused:
    the prompt no longer depends on a project's settings."""
    turnaround, backdrop = TURNAROUND_PROMPT, TURNAROUND_BACKDROP
    animal_turnaround = ANIMAL_TURNAROUND_PROMPT
    return f"""\
# Seed prompt — a Shahnameh story into a production

You are the director and production designer of a short animated film made from
a story of Ferdowsi's **Shahnameh**. The story follows this prompt, in Persian.
Turn it into **two** things, in this order and nothing else:

**Part 1 — the MAIN PROMPT.** The production instructions, inside one ```text
fence, in the exact format below. A pipeline reads it; a smaller model helps it,
so follow the format *literally*. Write it in **English**, except the spoken
lines, which are **Persian**, inside « ».

**Part 2 — the IMAGE PROMPTS.** Inside a second ```text fence: two lines per
character and one per scene, in this form:

```
character <id>: <image prompt>
turnaround <id>: <edit prompt, used with the character's front image attached>
scene <id>: <image prompt>
```

Each image is generated by a person and then turned into 3D: a character's
front image and its turnaround sheet (side and back, made by editing the front
image) are reconstructed into a rigged 3D model, a scene image into a 3D set by
estimating its depth. So the images must be framed for that, which is what the
rules in Part 2 below are for.

## Part 1 — the main prompt format

{main_prompt.FORMAT}
### How to fill it in

* **Characters.** Everyone who appears on screen, including animals. `kind`
  decides how the model is rigged: `human` and `humanoid` stand on two legs
  (a div or an angel is `humanoid`), `quadruped` walks on four (horses, lions,
  gazelles). `height_m` is real standing height — Rostam is heroic but human,
  about 2.1, not 4. `look` is the appearance only: face, age, hair, dress,
  colours, ornaments, weapons carried. No camera or lighting words in it.
* **Scenes.** Each distinct place. `look` describes the place with no people
  in it. Keep the number of scenes small; every scene is one image to make.
* **Shots.** A shot is one camera set-up in one scene. Give each an id like
  `s01_the_warning`, in film order.
  * `cast` lists everyone visible, each with a stage place and a facing.
    Places: {_join(staging.X_SLOTS.keys() - {"centre"})} across the stage;
    {_join(staging.DEPTH_SLOTS)} in depth. Give characters in one shot
    different places.
  * `camera` is `<size> on <subject>, <angle>, <move>`, e.g.
    `medium on keyumars, low_angle, slow push_in` or
    `wide on group, eye_level, static`. Sizes: {_join(staging.SIZES)}.
    Vary sizes across the film: establish wide, come closer for dialogue.
    The camera always looks into the scene from the front, so stage the
    action facing the camera or across it, never away from it.
  * `moves`: what each character *does* in the shot, in plain English, one
    item per character ("walks to the throne and kneels", "raises his mace
    and strikes the div"). Characters who only stand and listen need no move;
    characters who speak and do nothing else need no move either.
    The actions available are: {_join(moves.HUMAN_CLIPS)} for people and
    {_join(moves.QUADRUPED_CLIPS)} for animals — describe moves that can be
    built from these.
  * `lines`: every spoken line, in order, as
    `speaker [emotion]: «Persian line» (en: English gloss)`.
    The speaker is a character in the shot's cast, or `narrator` for the
    storyteller's voice over the picture. Use a narrator when the story needs
    one (set `narrator: yes` in FILM); keep narration short and let the
    picture carry the scene. Every line is recorded by a person, so write only
    the lines the story needs, each short enough to say in one breath.
  * `duration_s` only for a shot with no lines: how long it holds.
* **Fewer, better shots.** Six to twelve shots carry most stories.

## Part 2 — the image prompts

* **One per character**, starting with the character's `look` repeated
  verbatim, then: *{CHARACTER_FRAMING}* (for an animal: *{ANIMAL_FRAMING}*),
  then: *{CHARACTER_STYLE}*.
  * If the character is dark (black fur, dark robes, dark skin painted as
    shadow), replace the backdrop and lighting words with: *{DARK_FRAMING}*.
  * If the character is pale or wears white or cream, replace the backdrop
    words with: *{PALE_FRAMING}*.
  * **Leave weapons out of the image** unless the story cannot work without
    one; if it must be there, it is held upright in one hand, clear of the
    body and the legs.
  * **Wings** are folded close behind the back, clear of the arms; **capes and
    long robes** are fitted enough that the legs show through, never spread
    around the arms.
* **Write each character's `look` for this too:** fitted clothing that shows
  the body's shape, nothing flying or trailing, no cloak spread wide. What the
  character wears is theirs; how it hangs is the reference's business.
* **One turnaround per character**, written right after its `character`
  line. It is not generated from words: the person attaches the finished
  front image and asks the image model to edit it, so the prompt describes
  only what to make of the attached picture. Use exactly: *{turnaround}*
  (for an animal, which stands on four legs: *{animal_turnaround}*),
  with "{{subject}}" replaced by a two- or three-word description of the
  character ("king", "young prince", "horned demon", "lion") and
  "{{backdrop}}" replaced by the same backdrop words as that character's
  front prompt ("{backdrop}", or the dark or pale backdrop words when the
  front used them).
* **One per scene**, starting with the scene's `look`, then its `light`, then:
  *{SCENE_FRAMING}*, then: *{IMAGE_STYLE}*.
* Every image prompt is **one line**, English, self-contained, and contains
  nothing about the story or any action.

{STYLE_BIBLE}
## Before you answer, check

1. Exactly two fenced blocks: the main prompt, then the image prompts.
2. Every id is lower-case ASCII with underscores; every shot's `scene` and
   every `cast`, `moves` and `lines` name is defined above it.
3. Every spoken line is Persian inside « », with an `(en: ...)` gloss.
4. Every shot has either `lines` or `duration_s`.
5. There is one `character` and one `turnaround` line per character and one
   `scene` line per scene, each using the same id as the main prompt.

---

# The story
"""
