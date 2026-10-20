"""Make stdout survive text the model actually emits.

A Windows console defaults to a legacy code page (cp1252 here). The moment the
model returns a character outside it - a non-breaking hyphen, a curly quote, or
any Persian text at all - ``print`` raises ``UnicodeEncodeError`` and takes the
whole run down. That is not a display problem, it is a crash: run2.log ends with

    UnicodeEncodeError: 'charmap' codec can't encode character '‑'

after a complete build had already succeeded.

Every entry point should call :func:`configure` before it prints anything.
"""

from __future__ import annotations

import sys


def _reconfigure(stream):
    """Force UTF-8 on one stream, replacing anything it still cannot encode."""
    if stream is None:
        return
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        # Not a text stream we can retune (redirected, wrapped, or closed).
        pass


def _use_utf8_code_page():
    """Ask the Windows console to actually render UTF-8 rather than cp1252.

    Without this the stream no longer crashes but Persian shows up as boxes.
    Failure is harmless - the text is still written, just less legibly.
    """
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except Exception:
        pass


def configure():
    """Idempotent; safe to call from any entry point or stage CLI."""
    _use_utf8_code_page()
    _reconfigure(sys.stdout)
    _reconfigure(sys.stderr)


def safe(text, limit=None):
    """Coerce anything to a printable string, optionally truncated.

    Belt and braces for the few places that log model output directly: even
    with the stream retuned, a stray surrogate from a truncated API response
    would still raise.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.encode("utf-8", "replace").decode("utf-8")
    if limit is not None and len(text) > limit:
        text = text[:limit] + "..."
    return text
