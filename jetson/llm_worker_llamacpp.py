"""5557 LLM-worker protocol shim in front of ``llama-server``.

Runs **standalone on the Jetson** (``python3 jetson/llm_worker_llamacpp.py``)
and must never import ``mfw``; the only dependency is the standard library
(``urllib`` for the HTTP hop, ``socketserver`` for the TCP front). Parses
under Python 3.10 (JetPack 6).

Model: **Qwen2.5-3B-Instruct, GGUF Q4_K_M** (``qwen2.5-3b-instruct-q4_k_m.gguf``)
served by ``llama-server -ngl 99 -c 2048 -np 1 -ub 128``. The same file and
flags run on the laptop (llama.cpp Windows CUDA build) for evaluation.

Why a shim instead of pointing the laptop at llama-server directly: the
laptop's intent parser already speaks the newline-JSON protocol of
``scripts/llm_worker.py`` (``{"prompt"}`` -> ``{"ok", "text", "duration_s"}``)
and treats ``ok: false`` as "the model could not help". Keeping that contract
means the laptop's transformers worker and the Jetson's llama-server swap by
``--llm-server HOST:PORT`` alone, and a llama-server crash degrades to the
grammar rather than stopping the robot.

How the laptop uses the answer (``mfw/language/intent_parser.py``): in the
default ``--llm-mode hybrid`` the rule grammar parses first and this worker is
asked ONLY when the grammar refuses; the model can never override a rule parse
that succeeded. Its reply is then validated on the laptop (numbers re-read
from the utterance, no pronoun destinations, no invented targets).

What the shim adds on top of a plain proxy, all decided here so the laptop
side needs no change:

* the same system message ``scripts/llm_worker.py`` injects server-side;
* ``temperature 0`` and ``max_tokens 128`` (the worker's ``max_new_tokens``);
* ``response_format: json_schema`` whose ``skill`` is an enum, so the model
  cannot invent ``"grasp"``. The CLI default (``--skills hardware``) is
  :data:`HARDWARE_SKILL_NAMES` plus :data:`REFUSAL_SKILL` -- exactly the skills
  ``mfw/hardware/runtime.py::HARDWARE_SKILLS`` registers, plus a way to say
  "not a robot command". Without ``"unknown"`` in the enum a constrained model
  must pick SOME skill for "sing me a song". ``--skills all`` uses the 14-skill
  simulation registry :data:`SKILL_NAMES` (+ ``unknown``); the library default
  of :class:`LlamaCppCompleter` stays :data:`SKILL_NAMES` for existing callers.

Design choices:

* ``params`` is a free object in the schema: its shape differs per skill and
  the laptop's ``LlmIntentParser`` validates it downstream. Constraining it
  per skill would need a ``oneOf`` that small grammars handle poorly.
* One TCP connection per prompt on the laptop side, but the server still
  handles several lines per connection (a client may reuse one), hence the
  ``for line in rfile`` loop mirroring ``llm_worker.py``.
* ``ok: false`` carries ``"text": ""`` exactly as ``llm_worker.py`` does, so
  either worker's failure looks the same to the laptop.
* ``--host 0.0.0.0`` by default: the laptop connects over the LAN. The
  original worker binds loopback because it lives on the laptop.
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import socketserver
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

__all__ = [
    "DEFAULT_PORT",
    "DEFAULT_LLAMA_URL",
    "SKILL_NAMES",
    "HARDWARE_SKILL_NAMES",
    "REFUSAL_SKILL",
    "skills_for",
    "SYSTEM_MESSAGE",
    "MAX_TOKENS",
    "TEMPERATURE",
    "intent_json_schema",
    "build_chat_request",
    "extract_text",
    "LlamaCppCompleter",
    "LlmShimServer",
    "FakeLlamaServer",
    "selftest",
    "main",
]

_log = logging.getLogger("llm_worker_llamacpp")

DEFAULT_PORT = 5557
DEFAULT_LLAMA_URL = "http://127.0.0.1:8080"

#: Identical to the system message ``scripts/llm_worker.py`` prepends; the
#: prompt body (skill list, format examples, visible objects, utterance) is
#: built by the laptop's ``LlmIntentParser.PROMPT_TEMPLATE`` and arrives as
#: the user message.
SYSTEM_MESSAGE = "You are a precise robot command parser. Reply only with a JSON object."

#: ``max_new_tokens`` of ``scripts/llm_worker.py``.
MAX_TOKENS = 128
TEMPERATURE = 0.0

#: The 14 skills registered by ``mfw/skills/primitives.py`` (``ALL_SKILLS``,
#: the simulation registry), hardcoded because this file must not import
#: ``mfw``. Keep in sync: a skill missing here can never be emitted.
SKILL_NAMES: tuple[str, ...] = (
    "observe",
    "scan_scene",
    "look_at",
    "move_to",
    "move_relative",
    "rotate_wrist",
    "go_home",
    "open_gripper",
    "close_gripper",
    "pick",
    "place",
    "wait",
    "stop",
    "emergency_stop",
)


#: The skills ``mfw/hardware/runtime.py::HARDWARE_SKILLS`` registers, in its
#: order: the registry minus the sim-only ``look_at``/``rotate_wrist`` (the
#: hobby arm has no wrist camera and no wrist roll joint); ``scan_scene`` is
#: the fixed-camera scan (look, sweep the base, report). Hardcoded for the
#: same reason; tests/test_hardware_language.py pins it to the runtime.
HARDWARE_SKILL_NAMES: tuple[str, ...] = (
    "observe",
    "scan_scene",
    "move_to",
    "move_relative",
    "pick",
    "place",
    "open_gripper",
    "close_gripper",
    "go_home",
    "wait",
    "stop",
    "emergency_stop",
)

#: How the model declines. Never a registered skill; the laptop treats it as
#: "not understood" (mfw.language.intent_parser.REFUSAL_SKILL).
REFUSAL_SKILL = "unknown"


def skills_for(choice: str) -> tuple[str, ...]:
    """``--skills`` value -> the enum: ``hardware``, ``all`` or a comma list (+ ``unknown``)."""
    choice = str(choice).strip()
    if choice == "hardware":
        base = HARDWARE_SKILL_NAMES
    elif choice == "all":
        base = SKILL_NAMES
    else:
        base = tuple(s.strip() for s in choice.split(",") if s.strip())
        if not base:
            raise ValueError(f"--skills needs 'hardware', 'all' or a comma list, got {choice!r}")
    return tuple(dict.fromkeys((*base, REFUSAL_SKILL)))


def intent_json_schema(skills: tuple[str, ...] = SKILL_NAMES) -> dict[str, Any]:
    """JSON schema for one intent: ``{"skill": <enum>, "params": {...}, "confidence": 0..1}``."""
    return {
        "type": "object",
        "properties": {
            "skill": {"type": "string", "enum": list(skills)},
            "params": {"type": "object"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["skill", "params", "confidence"],
    }


def build_chat_request(prompt: str, skills: tuple[str, ...] = SKILL_NAMES) -> dict[str, Any]:
    """The ``/v1/chat/completions`` body sent for one laptop prompt."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": str(prompt)},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "robot_intent", "strict": True, "schema": intent_json_schema(skills)},
        },
    }


def extract_text(completion: Any) -> str:
    """``choices[0].message.content`` of an OpenAI-style reply, stripped."""
    try:
        return str(completion["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"llama-server reply lacks choices[0].message.content: {completion!r}") from exc


class LlamaCppCompleter:
    """POSTs one chat request per prompt to ``<llama_url>/v1/chat/completions``."""

    def __init__(self, llama_url: str = DEFAULT_LLAMA_URL, timeout_s: float = 30.0,
                 skills: tuple[str, ...] = SKILL_NAMES) -> None:
        self.llama_url = str(llama_url).rstrip("/")
        self.timeout_s = float(timeout_s)
        self.skills = tuple(skills)

    @property
    def endpoint(self) -> str:
        return f"{self.llama_url}/v1/chat/completions"

    def complete(self, prompt: str) -> str:
        """Return the model text; raises on transport, HTTP or malformed replies."""
        body = json.dumps(build_chat_request(prompt, self.skills)).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint, data=body, method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            raise RuntimeError(f"llama-server HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"llama-server unreachable at {self.endpoint}: {exc.reason}") from exc
        return extract_text(payload)

    def ping(self) -> bool:
        """``GET /health`` -> True when llama-server reports it is ready."""
        try:
            with urllib.request.urlopen(f"{self.llama_url}/health", timeout=self.timeout_s) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError):
            return False


class _ShimHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server: LlmShimServer = self.server  # type: ignore[assignment]
        for raw_line in self.rfile:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            reply = server.answer(line)
            self.wfile.write((json.dumps(reply) + "\n").encode("utf-8"))
            self.wfile.flush()


class LlmShimServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """TCP front speaking the ``scripts/llm_worker.py`` newline-JSON protocol."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, host: str, port: int, completer: LlamaCppCompleter) -> None:
        self.completer = completer
        self._thread: threading.Thread | None = None
        self.requests_served = 0
        super().__init__((host, int(port)), _ShimHandler)

    @property
    def bound_port(self) -> int:
        return int(self.server_address[1])

    def answer(self, line: str) -> dict[str, Any]:
        """One request line -> one reply dict (never raises)."""
        t0 = time.time()
        try:
            data = json.loads(line)
            prompt = data.get("prompt", "") if isinstance(data, dict) else ""
            if not isinstance(prompt, str) or not prompt:
                raise ValueError('request must be {"prompt": "<non-empty string>"}')
            text = self.completer.complete(prompt)
            self.requests_served += 1
            return {"ok": True, "text": text, "duration_s": time.time() - t0}
        except Exception as exc:
            _log.warning("completion failed: %s: %s", type(exc).__name__, exc)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "text": ""}

    def serve_in_thread(self) -> tuple["LlmShimServer", int]:
        """Serve in a daemon thread; returns ``(self, bound_port)``."""
        # 0.1 s poll so stop() returns promptly (the default 0.5 s costs a second per test).
        self._thread = threading.Thread(
            target=self.serve_forever, kwargs={"poll_interval": 0.1}, name="llm-shim", daemon=True
        )
        self._thread.start()
        return self, self.bound_port

    def stop(self) -> None:
        """Shut down and close the socket. Idempotent."""
        if self._thread is not None:
            self.shutdown()
            self._thread.join(timeout=5.0)
            self._thread = None
        self.server_close()


# ---------------------------------------------------------------------------
# Self-test double: an in-thread llama-server that answers a canned intent.
# ---------------------------------------------------------------------------


class FakeLlamaServer(http.server.ThreadingHTTPServer):
    """Minimal ``/v1/chat/completions`` + ``/health`` that echoes a fixed intent.

    Records every request body so a test can assert the schema, temperature
    and system message actually crossed the wire.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0,
                 reply_text: str = '{"skill": "pick", "params": {"target": "marker"}, "confidence": 0.9}') -> None:
        self.reply_text = reply_text
        self.requests: list[dict[str, Any]] = []
        self._thread: threading.Thread | None = None
        super().__init__((host, int(port)), _FakeLlamaHandler)

    @property
    def url(self) -> str:
        return f"http://{self.server_address[0]}:{self.server_address[1]}"

    def serve_in_thread(self) -> "FakeLlamaServer":
        self._thread = threading.Thread(
            target=self.serve_forever, kwargs={"poll_interval": 0.1}, name="fake-llama", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is not None:
            self.shutdown()
            self._thread.join(timeout=5.0)
            self._thread = None
        self.server_close()


class _FakeLlamaHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # silence the default stderr log
        _log.debug("fake llama: " + fmt, *args)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        server: FakeLlamaServer = self.server  # type: ignore[assignment]
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except ValueError:
            self._send_json(400, {"error": "bad json"})
            return
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return
        server.requests.append(body)
        self._send_json(200, {
            "id": "fake", "object": "chat.completion", "model": "fake-qwen",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": server.reply_text}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })


def selftest() -> int:
    """Fake llama + shim in-thread, one round trip over TCP. 0 on success."""
    import socket

    fake = FakeLlamaServer().serve_in_thread()
    shim, port = LlmShimServer("127.0.0.1", 0, LlamaCppCompleter(fake.url, timeout_s=5.0)).serve_in_thread()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5.0) as sock:
            sock.sendall(json.dumps({"prompt": "Command: pick up the marker\nJSON:"}).encode("utf-8") + b"\n")
            reply = json.loads(sock.makefile("r", encoding="utf-8").readline())
        request = fake.requests[-1]
        ok = (
            reply.get("ok") is True
            and json.loads(reply["text"])["skill"] == "pick"
            and request["messages"][0]["content"] == SYSTEM_MESSAGE
            and request["temperature"] == TEMPERATURE
            and request["max_tokens"] == MAX_TOKENS
            and request["response_format"]["json_schema"]["schema"]["properties"]["skill"]["enum"] == list(SKILL_NAMES)
        )
        print(f"selftest {'OK' if ok else 'FAILED'}: reply={reply}")
        return 0 if ok else 1
    finally:
        shim.stop()
        fake.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="5557 LLM-worker shim in front of llama-server")
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default 0.0.0.0: the laptop connects over the LAN)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--llama-url", default=DEFAULT_LLAMA_URL, help="llama-server base URL")
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds to wait for one completion")
    parser.add_argument(
        "--skills", default="hardware",
        help="skill enum the model may emit: 'hardware' (default: the hardware runtime's skills), "
             "'all' (the 14-skill simulation registry) or a comma list; 'unknown' is always added",
    )
    parser.add_argument("--selftest", action="store_true", help="run against an in-thread fake llama-server and exit")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.selftest:
        return selftest()

    completer = LlamaCppCompleter(args.llama_url, timeout_s=args.timeout, skills=skills_for(args.skills))
    _log.info("skill enum: %s", ", ".join(completer.skills))
    if not completer.ping():
        _log.warning("llama-server not answering at %s/health yet; serving anyway (replies are ok:false until it is up)",
                     completer.llama_url)
    server = LlmShimServer(args.host, args.port, completer)
    _log.info("LLM shim serving at %s:%d -> %s", args.host, server.bound_port, completer.endpoint)
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log.info("Shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
