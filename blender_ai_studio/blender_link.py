"""Transport layer: talks to the BlenderMCP addon's socket server, and can
launch a GUI Blender instance that hosts it.

Wire protocol (dictated by the installed addon, which we never modify):
  request : one JSON object, no framing - the server buffers bytes until
            ``json.loads`` succeeds, so exactly one request may be in flight
            per connection.
  response: one JSON object, no framing - we buffer until it parses.
"""

from __future__ import annotations

import base64
import glob
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

DEFAULT_PORT = 9876
DEFAULT_HOST = "127.0.0.1"


class BlenderError(RuntimeError):
    """The addon reported ``status: error`` for a command."""


class BlenderUnavailable(RuntimeError):
    """No Blender is listening (or it stopped answering)."""


def find_blender_executable():
    """Locate blender.exe, preferring the newest installed version."""
    env = os.environ.get("BSTUDIO_BLENDER_EXE")
    if env and os.path.isfile(env):
        return env

    patterns = [
        r"C:\Program Files\Blender Foundation\Blender*\blender.exe",
        r"C:\Program Files (x86)\Blender Foundation\Blender*\blender.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Blender*\blender.exe"),
        r"C:\Program Files\Blender*\blender.exe",
    ]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern))
    if found:
        # "Blender 5.2" sorts after "Blender 4.2"; good enough and stable.
        return sorted(found)[-1]

    from shutil import which

    return which("blender")


def port_is_open(host=DEFAULT_HOST, port=DEFAULT_PORT, timeout=0.6):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class BlenderLink:
    """One command-at-a-time connection to the addon's server.

    Each :class:`BlenderLink` owns its own socket, so the GUI's preview poller
    and the agent can talk to Blender concurrently without interleaving two
    requests on one connection.
    """

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT):
        self.host = host
        self.port = port
        self._sock = None
        self._lock = threading.Lock()

    # -- connection ---------------------------------------------------------
    def connect(self, timeout=6.0):
        if self._sock is not None:
            return
        try:
            self._sock = socket.create_connection((self.host, self.port), timeout=timeout)
        except OSError as exc:
            raise BlenderUnavailable(
                "Nothing is listening on {}:{}. Start Blender with the "
                "BlenderMCP addon enabled.".format(self.host, self.port)
            ) from exc

    def close(self):
        with self._lock:
            self._close_locked()

    def _close_locked(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    # -- commands -----------------------------------------------------------
    def send(self, cmd_type, params=None, timeout=120.0):
        """Send one command and return its ``result`` payload."""
        payload = json.dumps({"type": cmd_type, "params": params or {}}).encode("utf-8")

        with self._lock:
            raw = None
            last_exc = None
            for attempt in (0, 1):
                if self._sock is None:
                    self.connect()
                try:
                    self._sock.settimeout(timeout)
                    self._sock.sendall(payload)
                    raw = self._recv_json(timeout)
                    break
                except (OSError, ConnectionError) as exc:
                    # A stale socket (Blender restarted) fails on the first
                    # write; retry once on a fresh connection before giving up.
                    last_exc = exc
                    self._close_locked()
            if raw is None:
                raise BlenderUnavailable("Lost connection to Blender: {}".format(last_exc))

        if raw.get("status") == "error":
            raise BlenderError(raw.get("message", "unknown error"))
        return raw.get("result")

    def _recv_json(self, timeout):
        buffer = b""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Blender did not answer within {:.0f}s".format(timeout))
            self._sock.settimeout(min(remaining, 5.0))
            try:
                chunk = self._sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                raise ConnectionError("Blender closed the connection")
            buffer += chunk
            try:
                return json.loads(buffer.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue  # partial payload, keep reading

    # -- convenience wrappers ----------------------------------------------
    def ping(self, timeout=4.0):
        try:
            result = self.send("ping", timeout=timeout)
        except (BlenderError, BlenderUnavailable, TimeoutError):
            return False
        return bool(result and result.get("pong"))

    def run_python(self, code, timeout=180.0):
        """Execute code in Blender and return everything it printed.

        The addon's ``execute_code`` only surfaces ``str(exception)`` when code
        raises, which loses the traceback the agent needs to repair itself. So
        the payload is wrapped in a harness that catches, prints the full
        traceback, and always returns normally - the caller reads the marker.
        """
        encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
        harness = (
            "import base64, sys, traceback\n"
            "_src = base64.b64decode('" + encoded + "').decode('utf-8')\n"
            "try:\n"
            "    exec(compile(_src, '<agent>', 'exec'), globals())\n"
            "    print('__BSTUDIO_OK__')\n"
            "except Exception:\n"
            "    print('__BSTUDIO_ERROR__')\n"
            "    traceback.print_exc(file=sys.stdout)\n"
        )
        result = self.send("execute_code", {"code": harness}, timeout=timeout)
        if isinstance(result, dict):
            return result.get("result", "")
        return str(result)

    def viewport_screenshot(self, filepath, max_size=900, timeout=30.0):
        return self.send(
            "get_viewport_screenshot",
            {"filepath": filepath, "max_size": max_size, "format": "png"},
            timeout=timeout,
        )


@dataclass
class BlenderProcess:
    process: object
    executable: str
    port: int
    launched_by_us: bool


def launch_blender(port=DEFAULT_PORT, executable=None, on_log=None, ready_timeout=120.0):
    """Start a visible Blender that auto-serves BlenderMCP on ``port``.

    If something is already serving on that port, attach to it instead.
    """

    def log(msg):
        if on_log:
            on_log(msg)

    if port_is_open(port=port):
        link = BlenderLink(port=port)
        alive = link.ping()
        link.close()
        if alive:
            log("Attached to a Blender already serving on port {}.".format(port))
            return BlenderProcess(None, executable or "", port, launched_by_us=False)

    exe = executable or find_blender_executable()
    if not exe:
        raise BlenderUnavailable(
            "Could not find blender.exe. Set BSTUDIO_BLENDER_EXE to its full path."
        )

    bootstrap = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bootstrap.py")
    ready_file = os.path.join(
        tempfile.gettempdir(), "bstudio_ready_{}_{}.txt".format(port, os.getpid())
    )
    if os.path.exists(ready_file):
        os.remove(ready_file)

    env = dict(os.environ)
    env["BSTUDIO_PORT"] = str(port)
    env["BSTUDIO_READY_FILE"] = ready_file

    log("Launching Blender: {}".format(exe))
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(
        [exe, "--python", bootstrap],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise BlenderUnavailable(
                "Blender exited during startup (code {}).".format(proc.returncode)
            )
        if os.path.exists(ready_file) and port_is_open(port=port):
            link = BlenderLink(port=port)
            alive = link.ping()
            link.close()
            if alive:
                log("Blender is up and serving.")
                return BlenderProcess(proc, exe, port, launched_by_us=True)
        time.sleep(0.5)

    raise BlenderUnavailable(
        "Blender did not start serving on port {} within {:.0f}s.".format(port, ready_timeout)
    )
