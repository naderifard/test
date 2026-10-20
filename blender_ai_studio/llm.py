"""Client for the scene-building model on NVIDIA's hosted endpoint.

Everything that reaches the network goes through :class:`RateGate`, which
enforces the required minimum spacing between requests process-wide - including
the retries, which would otherwise be exactly the moment spacing matters most.

The reasoning control is a boolean. gpt-oss took a ``reasoning_effort`` budget
of low/medium/high; nemotron takes ``enable_thinking``, on or off, so there is
no middle setting to expose and callers pass ``thinking=True|False``.
"""

from __future__ import annotations

import json
import threading
import time

from openai import OpenAI

from . import config


class RateGate:
    """Process-wide minimum spacing between outbound requests."""

    def __init__(self, min_interval=config.MIN_REQUEST_INTERVAL):
        self.min_interval = float(min_interval)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            gap = self.min_interval - (now - self._last)
            if gap > 0:
                time.sleep(gap)
            self._last = time.monotonic()


_GATE = RateGate()


class LLMError(RuntimeError):
    pass


class Chat:
    """Thin wrapper over the chat-completions call this app needs."""

    def __init__(self, api_key, model=config.MODEL, base_url=config.NVIDIA_BASE_URL,
                 on_log=None, gate=None):
        if not api_key:
            raise LLMError(
                "No NVIDIA API key. Paste one into the Settings panel, or set "
                "the NVIDIA_API_KEY environment variable. Get a free key at "
                "https://build.nvidia.com/"
            )
        self.model = model
        self.on_log = on_log
        self.gate = gate or _GATE
        # This endpoint can spend 100s+ queueing before the first token when
        # thinking is enabled, so the timeout is deliberately long.
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=240.0,
                              max_retries=0)
        # Set to False after the endpoint rejects the thinking field, so we stop
        # paying a failed round trip for it on every later call.
        self._supports_thinking = True

    def _log(self, message):
        if self.on_log:
            self.on_log(message)

    def complete(self, messages, tools=None, thinking=False, temperature=0.7,
                 max_tokens=8000, tool_choice=None, attempts=4):
        """One chat completion. Returns the assistant message object."""
        last_error = None
        for attempt in range(attempts):
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice or "auto"
            if self._supports_thinking:
                # Nemotron exposes reasoning through the chat template rather
                # than a top-level parameter.
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": bool(thinking)}
                }

            self.gate.wait()
            try:
                response = self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - provider errors vary
                text = str(exc)
                last_error = exc
                if self._supports_thinking and _is_thinking_rejection(text):
                    self._log("Endpoint rejected enable_thinking; continuing "
                              "without it.")
                    self._supports_thinking = False
                    continue
                if not _is_retryable(text) or attempt == attempts - 1:
                    raise LLMError(_friendly(text)) from exc
                backoff = min(2.0 ** attempt, 8.0)
                self._log(f"Request failed ({_short(text)}); retrying in {backoff:.0f}s.")
                time.sleep(backoff)
                continue

            if not response.choices:
                last_error = LLMError("The model returned no choices.")
                continue
            return response.choices[0].message

        raise LLMError(_friendly(str(last_error)))

    def text(self, system, user, thinking=False, temperature=0.8,
             max_tokens=4000):
        """A plain single-turn completion, returning the message text."""
        message = self.complete(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            thinking=thinking, temperature=temperature, max_tokens=max_tokens,
        )
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            self._log(f"[thinking] {reasoning.strip()[:400]}")
        return (message.content or "").strip()


def _is_thinking_rejection(text):
    lowered = text.lower()
    return "chat_template_kwargs" in lowered or "enable_thinking" in lowered


def _is_retryable(text):
    lowered = text.lower()
    markers = ("429", "500", "502", "503", "504", "timeout", "timed out",
               "connection", "overloaded", "temporarily", "rate limit")
    return any(marker in lowered for marker in markers)


def _short(text, limit=160):
    text = " ".join(str(text).split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _friendly(text):
    lowered = str(text).lower()
    if "401" in lowered or "unauthorized" in lowered or "invalid api key" in lowered:
        return ("NVIDIA rejected the API key (401). Check it in Settings - keys "
                "start with 'nvapi-'.")
    if "404" in lowered and "model" in lowered:
        return (f"NVIDIA does not expose '{config.MODEL}' on this account. "
                "Check the model is enabled at build.nvidia.com.")
    if "429" in lowered:
        return ("NVIDIA is rate limiting this key (429). Wait a minute, or "
                "raise the request spacing in Settings.")
    return f"LLM request failed: {_short(text, 400)}"


def message_to_dict(message):
    """Normalise an SDK message into the dict shape the history expects."""
    payload = {"role": "assistant", "content": message.content or ""}
    calls = getattr(message, "tool_calls", None)
    if calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments or "{}",
                },
            }
            for call in calls
        ]
    return payload


def parse_arguments(raw):
    """Tool arguments arrive as a JSON string that small models often mangle."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except ValueError:
        pass
    # Trailing commentary after the closing brace is the usual failure.
    depth, end = 0, -1
    for index, char in enumerate(raw):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end > 0:
        try:
            return json.loads(raw[:end])
        except ValueError:
            pass
    raise ValueError(f"Could not parse tool arguments: {_short(raw, 200)}")
