"""Analyse a main prompt: parse it, repair it, understand its moves, stage it.

This is the step the GUI runs when the user presses Analyse, and the only step
of the pipeline with a language model in it: ``nvidia/nemotron-3-ultra-550b-a55b``
on NVIDIA's endpoint, because nobody else is present while the pipeline runs.

The model is given two narrow jobs, both checked by code:

1. **Repair.** When the deterministic parser finds problems, the model is shown
   the grammar, the prompt and the list of problems, and asked for the corrected
   prompt. The reply is accepted only if it parses with fewer errors *and*
   every Persian «line» of the original is still in it, byte for byte -- the
   model works in English and is never trusted to carry Persian unverified.
2. **Moves.** Each free-text move ("raises his staff slowly in blessing") is
   mapped onto a short sequence of clips from ``moves.py``. The reply must be
   JSON that ``moves.check`` accepts; one retry with the errors, then the
   keyword reader in ``moves.guess`` as the floor. The panel shows which of the
   three produced each move.

Everything else -- ids, line ids, stage positions, cameras -- is deterministic.
"""

from __future__ import annotations

import json
import re
import time

from . import main_prompt, moves, staging

MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
REPAIR_ROUNDS = 2

SYSTEM = ("You are a precise assistant inside a film production pipeline. You "
          "work in English. You never translate, rewrite or re-spell text inside "
          "« » quotes: it is Persian dialogue and must be copied exactly.")


class AnalysisError(RuntimeError):
    pass


def chat_client(api_key=None, on_log=None, model=MODEL):
    """The NVIDIA client the GUI already uses, pointed at the analyst model."""
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from blender_ai_studio import config, llm
    key = config.resolve_api_key(api_key)
    return llm.Chat(key, model=model, on_log=on_log)


def _json_from(text):
    """The first JSON object in a reply, fences and chatter tolerated."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:index + 1])
    raise ValueError("unterminated JSON object in the reply")


# -- 1. repair -------------------------------------------------------------

def repair(text, problems, chat, log):
    """The model's corrected prompt, or None when no reply was acceptable."""
    originals = main_prompt.persian_lines(text)
    errors = [p for p in problems if p.severity == "error"]
    for round_index in range(REPAIR_ROUNDS):
        listing = "\n".join(f"- {p}" for p in errors)
        prompt = (
            f"{main_prompt.FORMAT}\n\n"
            f"The main prompt below does not follow this format. Problems "
            f"found:\n{listing}\n\n"
            f"Return the corrected main prompt, complete, in the format above "
            f"and nothing else. Fix only what is needed. Copy every « » "
            f"Persian line exactly as it is.\n\n"
            f"MAIN PROMPT:\n{text}")
        log(f"analysis: asking the model to repair {len(errors)} problem(s) "
            f"(round {round_index + 1})")
        reply = chat.text(SYSTEM, prompt, thinking=True, temperature=0.1,
                          max_tokens=16000)
        reply = re.sub(r"^```[a-z]*\n|\n```\s*$", "", reply.strip())
        kept = main_prompt.persian_lines(reply)
        lost = [line for line in originals if line not in kept]
        if lost:
            log(f"analysis: rejected the repair; it changed {len(lost)} "
                f"Persian line(s)")
            continue
        try:
            _doc, new_problems = main_prompt.parse(reply)
        except main_prompt.MainPromptError as exc:
            log(f"analysis: rejected the repair; {exc}")
            continue
        new_errors = [p for p in new_problems if p.severity == "error"]
        if len(new_errors) < len(errors):
            log(f"analysis: repair accepted, {len(errors)} -> "
                f"{len(new_errors)} error(s)")
            return reply
        errors = new_errors or errors
    return None


# -- 2. moves --------------------------------------------------------------

def classify_moves(shot, kinds, chat, log):
    """Clips for every move in one shot: ``{character: (clips, source)}``."""
    cast_ids = [m["character"] for m in shot["cast"]]
    targets = set(cast_ids) | {"camera"}
    wanted = shot["moves"]
    out = {}
    if not wanted:
        return out
    if chat is not None:
        menus = {kinds[m["character"]] for m in wanted}
        listing = "\n\n".join(
            f"Clips for a {k} character:\n{moves.menu(k)}" for k in sorted(menus))
        items = "\n".join(
            f"- {m['character']} ({kinds[m['character']]}): {m['text']}"
            for m in wanted)
        prompt = (
            f"A film shot. Summary: {shot.get('summary') or '(none)'}\n"
            f"Characters in the shot: {', '.join(cast_ids)}.\n\n"
            f"{listing}\n\n"
            f"Map each move below to a short sequence (1 to 3) of clips, in the "
            f"order they happen. Optional fields per clip: \"target\" (a "
            f"character id from the shot, or \"camera\"), \"direction\" (one "
            f"of {', '.join(moves.DIRECTIONS)}), \"to\" (a stage place like "
            f"\"left\" or \"center back\").\n\nMoves:\n{items}\n\n"
            f"Reply with JSON only: {{\"moves\": [{{\"character\": id, "
            f"\"clips\": [{{\"clip\": name, \"target\": id}}]}}]}}")
        feedback = ""
        for attempt in range(2):
            try:
                reply = chat.text(SYSTEM, prompt + feedback, thinking=False,
                                  temperature=0.1, max_tokens=3000)
                data = _json_from(reply)
            except Exception as exc:                        # noqa: BLE001
                log(f"analysis: move reply unusable ({exc})")
                feedback = "\n\nYour last reply was not valid JSON."
                continue
            bad = []
            for entry in data.get("moves") or []:
                cid = entry.get("character")
                if cid not in kinds:
                    continue
                clips = [c for c in entry.get("clips") or []
                         if isinstance(c, dict)]
                problems = moves.check(clips, kinds[cid], targets)
                if problems:
                    bad.append(f"{cid}: {'; '.join(problems)}")
                else:
                    out[cid] = (_clean(clips), "model")
            if not bad:
                break
            feedback = ("\n\nYour last reply had problems:\n"
                        + "\n".join(f"- {b}" for b in bad)
                        + "\nFix them and reply with the whole JSON again.")
    for move in wanted:
        cid = move["character"]
        if cid not in out:
            out[cid] = (moves.guess(move["text"], kinds[cid]), "keywords")
    return out


def _clean(clips):
    keep = ("clip", "target", "direction", "to")
    return [{k: c[k] for k in keep if c.get(k)} for c in clips]


# -- the whole analysis ------------------------------------------------------

def analyse(text, chat=None, log=print):
    """Main prompt text -> the analysis the panel shows and the plan is made of.

    ``chat`` is an ``llm.Chat`` (see :func:`chat_client`), or None to run on the
    deterministic parts alone.
    """
    started = time.time()
    notes, llm_used = [], []
    doc, problems = main_prompt.parse(text)
    errors = [p for p in problems if p.severity == "error"]
    if errors and chat is not None:
        fixed = repair(text, problems, chat, log)
        if fixed is not None:
            text = fixed
            doc, problems = main_prompt.parse(text)
            llm_used.append("repair")
    errors = [p for p in problems if p.severity == "error"]

    kinds = {c["id"]: c["kind"] for c in doc["characters"]}
    bodies = {c["id"]: {"height_m": c["height_m"], "kind": c["kind"]}
              for c in doc["characters"]}

    for shot in doc["shots"]:
        members = [dict(m, **bodies.get(m["character"], {}))
                   for m in shot["cast"]]
        placed, stage_notes = staging.place_cast(members, _featured(shot))
        notes.extend(f"{shot['id']}: {n}" for n in stage_notes)
        classified = classify_moves(shot, kinds, chat, log)
        if any(source == "model" for _clips, source in classified.values()):
            llm_used.append(f"moves:{shot['id']}")
        notes.extend(f"{shot['id']}: {n}" for n in staging.engage(
            placed, {cid: clips for cid, (clips, _s) in classified.items()},
            bodies))
        speakers = {line["speaker"] for line in shot["lines"]}
        texts = {m["character"]: m["text"] for m in shot["moves"]}
        for member in shot["cast"]:
            cid = member["character"]
            member.update(placed[cid])
            if cid in classified:
                clips, source = classified[cid]
            elif cid in speakers and kinds.get(cid) not in moves.QUADRUPED_KINDS:
                clips, source = [{"clip": "talk"}], "speaks"
            else:
                clips, source = [{"clip": "idle"}], "default"
            member["move"] = {
                "text": texts.get(cid, ""),
                "clips": clips,
                "source": source,
                "take": main_prompt.take_id(shot["id"], cid),
            }
        camera, framing, camera_notes = staging.solve_camera(
            shot["camera"], placed, bodies)
        shot["camera_resolved"] = camera
        shot["framing"] = framing
        notes.extend(f"{shot['id']}: {n}" for n in camera_notes)

    return {
        "schema_version": 1,
        "model": MODEL if chat is not None else None,
        "llm_used": llm_used,
        "seconds": round(time.time() - started, 2),
        "text": text,
        "film": doc["film"],
        "characters": doc["characters"],
        "scenes": doc["scenes"],
        "shots": doc["shots"],
        "problems": [p.as_dict() for p in problems],
        "errors": len(errors),
        "notes": notes,
    }


def _featured(shot):
    """The shot's camera subject and its speakers (narration has no face)."""
    cast_ids = [m["character"] for m in shot["cast"]]
    try:
        _size, subject, *_rest = staging.parse_camera(shot.get("camera") or "",
                                                      cast_ids)[0]
    except Exception:                               # noqa: BLE001
        subject = None
    out = {line["speaker"] for line in shot.get("lines") or []
           if line.get("speaker") != main_prompt.NARRATOR}
    if subject in cast_ids:
        out.add(subject)
    return out


def restage(analysis):
    """Re-solve every position, facing and camera from a saved analysis.

    Staging is deterministic, so this needs no model: it is what the GUI runs
    after the user edits a stage position or a camera line, and what a change
    to ``staging.py`` needs to reach an existing project. Moves keep the clips
    the model gave them.
    """
    bodies = {c["id"]: {"height_m": c["height_m"], "kind": c["kind"]}
              for c in analysis["characters"]}
    notes = []
    for shot in analysis["shots"]:
        members = [dict(m, **bodies.get(m["character"], {}))
                   for m in shot["cast"]]
        placed, stage_notes = staging.place_cast(members, _featured(shot))
        stage_notes += staging.engage(
            placed, {m["character"]: (m.get("move") or {}).get("clips")
                     for m in shot["cast"]}, bodies)
        for member in shot["cast"]:
            member.update(placed[member["character"]])
        camera, framing, camera_notes = staging.solve_camera(
            shot["camera"], placed, bodies)
        shot["camera_resolved"] = camera
        shot["framing"] = framing
        notes.extend(f"{shot['id']}: {n}" for n in stage_notes + camera_notes)
    analysis["notes"] = notes
    return analysis


def to_production(analysis, fps=24):
    """The analysis as ``production.json``, schema and cross-checks included."""
    characters = []
    for c in analysis["characters"]:
        characters.append({
            "id": c["id"],
            "name_fa": c["name_fa"],
            "name_en": c["name"],
            "kind": c["kind"],
            "height_m": round(float(c["height_m"]), 3),
            "image_prompt": c["look"] or c["name"],
            # A photo-built character takes its colours from the photo; the
            # role map only tags the one material slot for the style packs.
            "material_roles": {"surface": "skin"},
        })
    envs = [{
        "id": s["id"],
        "interior": s["setting"] == "indoor",
        "image_prompt": s["look"] or s["name"],
        "ground": "ground",
        "light_intent": s["light"],
    } for s in analysis["scenes"]]
    shots = []
    for shot in analysis["shots"]:
        cast = []
        for member in shot["cast"]:
            move = member["move"]
            cast.append({
                "character": member["character"],
                "position_m": [round(v, 3) for v in member["position_m"]],
                "facing_deg": member["facing_deg"],
                "take": move["take"],
                "retime": ("loop" if all(c["clip"] in moves.CYCLIC
                                         for c in move["clips"])
                           else "hold_last_pose"),
                "action": {"text": move["text"], "clips": move["clips"],
                           "source": move["source"]},
            })
        entry = {"id": shot["id"], "env": shot["scene"], "cast": cast,
                 "camera": shot["camera_resolved"]}
        if shot.get("summary"):
            entry["beats"] = [{"from_pct": 0.0, "to_pct": 1.0,
                               "what": shot["summary"]}]
        dialogue = []
        for line in shot["lines"]:
            record = {"line_id": line["line_id"],
                      "character": line["speaker"],
                      "text_fa": line["text_fa"],
                      "emotion": line["emotion"]}
            if line["text_en"]:
                record["text_en"] = line["text_en"]
            dialogue.append(record)
        if dialogue:
            entry["dialogue"] = dialogue
        else:
            entry["silent_duration_s"] = float(shot.get("duration_s") or 4.0)
        shots.append(entry)
    production = {
        "title": analysis["film"].get("title_fa") or analysis["film"]["title"],
        "fps": int(fps),
        "notes": f"Built from the main prompt: {analysis['film']['title']}",
        "characters": characters,
        "envs": envs,
        "shots": shots,
    }
    return production


def inputs_panel(analysis):
    """What the panel asks the user for, grouped as the GUI shows it."""
    characters = [{"id": c["id"], "name": c["name"], "name_fa": c["name_fa"],
                   "kind": c["kind"], "look": c["look"], "voice": c["voice"]}
                  for c in analysis["characters"]]
    scenes = [{"id": s["id"], "name": s["name"], "look": s["look"],
               "light": s["light"]} for s in analysis["scenes"]]
    sentences, move_rows = [], []
    for shot in analysis["shots"]:
        for line in shot["lines"]:
            sentences.append({"line_id": line["line_id"], "shot": shot["id"],
                              "speaker": line["speaker"],
                              "emotion": line["emotion"],
                              "text_fa": line["text_fa"],
                              "text_en": line["text_en"],
                              "narration": line["speaker"]
                              == main_prompt.NARRATOR})
        for member in shot["cast"]:
            move = member["move"]
            move_rows.append({"take": move["take"], "shot": shot["id"],
                              "character": member["character"],
                              "text": move["text"],
                              "clips": [c["clip"] for c in move["clips"]],
                              "source": move["source"]})
    return {"characters": characters, "scenes": scenes,
            "sentences": sentences, "moves": move_rows,
            "narrator": any(s["narration"] for s in sentences)}
