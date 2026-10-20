"""The two-stage pipeline: direct the shot, then build it.

Stage 1 turns the user's one-liner into a specific brief. Stage 2 executes that
brief through tool calls, and refuses to accept "finished" until the scene
measures up.
"""

from __future__ import annotations

from . import prompts
from .llm import LLMError, message_to_dict, parse_arguments
from .tools import TOOL_SCHEMAS, ToolBox


class Cancelled(RuntimeError):
    """The user stopped the run."""


def write_brief(chat, user_prompt, mode, frames=96, fps=24, thinking=True,
                on_log=None):
    """Stage 1 - the art direction the builder will follow."""
    if on_log:
        on_log("Writing the shot brief...")
    brief = chat.text(
        system=prompts.director_system_prompt(mode, frames=frames, fps=fps),
        user=prompts.director_user_prompt(user_prompt, mode, frames=frames, fps=fps),
        thinking=thinking,
        temperature=0.85,
        max_tokens=3000,
    )
    if not brief.strip():
        raise LLMError("The director returned an empty brief.")
    return brief.strip()


def _quality_problems(analysis):
    """The bar from the system prompt, checked by the host rather than trusted."""
    if not analysis:
        return ["The scene has never been analysed - call analyse_frame."]
    problems = []
    if analysis.get("render_error"):
        return [f"The test render failed: {analysis['render_error']}"]

    dark = analysis.get("fraction_near_black", 0.0)
    blown = analysis.get("fraction_blown_out", 0.0)
    contrast = analysis.get("contrast", 0.0)
    if dark > 0.5:
        problems.append(
            f"{dark * 100:.0f}% of the frame is nearly black. Add or brighten "
            "lights, raise the world strength, or re-aim the camera."
        )
    if blown > 0.15:
        problems.append(
            f"{blown * 100:.0f}% of the frame is blown out. Reduce light energy "
            "or world strength."
        )
    median = analysis.get("median_luminance", 0.0)
    if median > 0.72:
        problems.append(
            f"The frame is washed out (median luminance {median:.2f}) - shapes and "
            "materials have stopped reading. Cut the light energy and sky/world "
            "strength, then call B.auto_exposure()."
        )
    if contrast < 0.05:
        problems.append(
            f"Contrast is only {contrast:.3f} - the frame is nearly flat. The "
            "camera may be inside geometry or facing an empty backdrop."
        )
    visible = [row for row in analysis.get("objects_in_frame", []) if row.get("in_frame")]
    if not visible:
        problems.append("Nothing is inside the camera frustum. Call B.frame_all().")
    problems.extend(analysis.get("hints", []))
    # hints and the checks above overlap; keep the first mention of each idea
    seen, unique = set(), []
    for problem in problems:
        key = problem[:40]
        if key not in seen:
            seen.add(key)
            unique.append(problem)
    return unique


def build_scene(chat, session, user_prompt, brief, mode, res, engine,
                frames=96, fps=24, thinking=False, max_steps=26,
                on_log=None, on_event=None, should_stop=None):
    """Stage 2 - run the tool loop until the scene is built and passes."""

    def log(message):
        if on_log:
            on_log(message)

    def check_stop():
        if should_stop and should_stop():
            raise Cancelled("Stopped by the user.")

    toolbox = ToolBox(session, on_log=on_log, on_event=on_event)
    messages = [
        {"role": "system", "content": prompts.builder_system_prompt()},
        {
            "role": "user",
            "content": prompts.builder_user_prompt(
                user_prompt, brief, mode, res[0], res[1], engine,
                frames=frames, fps=fps,
            ),
        },
    ]

    pushbacks = 0
    max_pushbacks = 2
    idle_turns = 0

    for step in range(max_steps):
        check_stop()
        log(f"Model turn {step + 1}/{max_steps}")
        message = chat.complete(messages, tools=TOOL_SCHEMAS, thinking=thinking,
                                temperature=0.6, max_tokens=3000)

        reasoning = getattr(message, "reasoning_content", None)
        if reasoning and on_event:
            on_event("thinking", {"text": reasoning.strip()})

        messages.append(message_to_dict(message))
        calls = getattr(message, "tool_calls", None)

        if not calls:
            # No tool call: either it is chatting, or it thinks it is done.
            idle_turns += 1
            if message.content:
                log(f"Model: {message.content.strip()[:300]}")
            if idle_turns >= 3:
                log("The model stopped calling tools; wrapping up.")
                break
            messages.append({"role": "user", "content": prompts.CONTINUE_NUDGE})
            continue

        idle_turns = 0
        for call in calls:
            check_stop()
            name = call.function.name
            try:
                arguments = parse_arguments(call.function.arguments)
            except ValueError as exc:
                result = f"Could not read your arguments: {exc}"
            else:
                if name == "finish":
                    # Verify before believing it.
                    analysis = toolbox.last_analysis
                    if analysis is None:
                        analysis = session.analyse()
                        toolbox.last_analysis = analysis
                        if on_event:
                            on_event("analysis", analysis)
                    problems = _quality_problems(analysis)
                    if problems and pushbacks < max_pushbacks:
                        pushbacks += 1
                        log(f"Quality gate: {len(problems)} issue(s) to fix "
                            f"(pushback {pushbacks}/{max_pushbacks}).")
                        result = prompts.QUALITY_PUSHBACK.format(
                            problems="\n".join(f"- {p}" for p in problems)
                        )
                        # Force another look next time round.
                        toolbox.last_analysis = None
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result,
                        })
                        continue
                    toolbox.call(name, arguments)
                    log("Model reports the scene is complete.")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": "Acknowledged.",
                    })
                    return toolbox, messages
                result = toolbox.call(name, arguments)

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            })

        if toolbox.finished:
            return toolbox, messages

    log("Reached the step limit; using the scene as it stands.")
    return toolbox, messages
