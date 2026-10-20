"""Settings, paths and API-key resolution."""

from __future__ import annotations

import json
import os

APP_NAME = "Blender AI Studio"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.json")

# Secrets live here, never in config.json. config.json is meant to be readable
# and shareable; an API key in it leaks the moment the project is zipped up or
# put under version control.
ENV_PATH = os.path.join(PROJECT_DIR, ".env")

SECRET_KEYS = ("api_key",)

# The endpoint is fixed by the project brief. The model is not: the brief named
# openai/gpt-oss-120b, which NVIDIA no longer serves on this endpoint (only the
# 20b variant remains). Nemotron replaces it, and its reasoning control is a
# boolean rather than gpt-oss's low/medium/high budget.
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-super-120b-a12b"

# NVIDIA's free tier is rate limited; the brief asks for at least one second
# between requests, and every LLM call in this app goes through that gate.
MIN_REQUEST_INTERVAL = 1.0

DEFAULTS = {
    "api_key": "",
    "port": 9876,
    "blender_exe": "",
    "engine": "EEVEE",
    "resolution": "1920x1080",
    "samples": 64,
    "frames": 96,
    "fps": 24,
    # Thinking on for the director (it plans the whole shot in one pass) and off
    # for the builder (it emits many short tool calls, where the latency cost of
    # reasoning is paid over and over for little gain).
    "director_thinking": True,
    "builder_thinking": False,
    "max_steps": 26,
    "preview_interval": 2.0,
}

RESOLUTIONS = [
    ("960x540 (fast draft)", 960, 540),
    ("1280x720 (HD)", 1280, 720),
    ("1920x1080 (Full HD)", 1920, 1080),
    ("2560x1440 (2K)", 2560, 1440),
]

# Settings that used to hold "low"/"medium"/"high" for gpt-oss. Read once, on
# load, so an existing config.json does not silently lose its meaning.
LEGACY_EFFORT_KEYS = {"director_effort": "director_thinking",
                      "builder_effort": "builder_thinking"}


def load_config():
    settings = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                stored = json.load(fh)
        except (OSError, ValueError):
            stored = {}
        for old, new in LEGACY_EFFORT_KEYS.items():
            if old in stored and new not in stored:
                # "low" meant "don't spend long thinking"; anything above it did.
                stored[new] = str(stored.pop(old)).lower() in ("medium", "high")
        settings.update(stored)
    return settings


def save_config(settings):
    """Persist the non-secret settings. Secrets go to :func:`save_api_key`."""
    keep = {k: v for k, v in settings.items()
            if k in DEFAULTS and k not in SECRET_KEYS}
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(keep, fh, indent=2)
    except OSError:
        pass


# -- secrets ---------------------------------------------------------------
def read_env_file(path=ENV_PATH):
    """Parse a minimal ``KEY=value`` file. Missing file is not an error."""
    values = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                values[name.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    return values


def save_api_key(key, path=ENV_PATH):
    """Write the key to .env, leaving any other entries in place."""
    values = read_env_file(path)
    key = (key or "").strip()
    if key:
        values["NVIDIA_API_KEY"] = key
    else:
        values.pop("NVIDIA_API_KEY", None)
    body = ["# Secrets. Do not commit this file.\n"]
    body += [f"{name}={value}\n" for name, value in sorted(values.items())]
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(body)
    except OSError:
        return False
    return True


def migrate_secrets():
    """Move a key left in config.json into .env, then strip it from config.json.

    Idempotent, and it never destroys the key: .env is written and verified
    before config.json is rewritten without it.
    """
    if not os.path.exists(CONFIG_PATH):
        return False
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            stored = json.load(fh)
    except (OSError, ValueError):
        return False
    legacy = str(stored.get("api_key", "")).strip()
    if not legacy:
        return False
    if read_env_file().get("NVIDIA_API_KEY", "").strip() != legacy:
        if not save_api_key(legacy):
            return False  # could not secure it, so leave config.json alone
    stored.pop("api_key", None)
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(stored, fh, indent=2)
    except OSError:
        return False
    return True


def resolve_api_key(explicit=None):
    """Explicit value, then the environment, then .env, then legacy config."""
    for candidate in (explicit,
                      os.environ.get("NVIDIA_API_KEY"),
                      os.environ.get("NVIDIA_API_TOKEN"),
                      read_env_file().get("NVIDIA_API_KEY"),
                      load_config().get("api_key")):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return ""


def parse_resolution(text):
    try:
        part = text.split()[0]
        width, height = part.lower().split("x")
        return int(width), int(height)
    except (ValueError, IndexError):
        return 1920, 1080


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR
