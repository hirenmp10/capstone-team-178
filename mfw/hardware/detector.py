"""TCP newline-JSON client for the detector service (port 5558).

Pure stdlib + NumPy at import. This module must never import Isaac Sim, torch
or transformers; ``cv2`` is imported lazily only when a frame is sent.

Protocol (shared by ``jetson/detector_service.py`` on the Jetson and
``scripts/serve_detector.py`` on the laptop, so they swap by host/port):

    {"cmd": "ping"} -> {"ok": true, "backend": "scripted|nanoowl|florence"}
    {"cmd": "detect", "labels": [...], "min_score": 0.1[, "jpeg": b64]}
        -> {"ok": true, "t", "width", "height",
            "objects": [{"label", "confidence", "bbox_px": [x0, y0, x1, y1]}]}

One connection per request, like the LLM client in ``run_assistant.py``. A
persistent socket would save a millisecond on the LAN and cost a reconnect
state machine; a detector round trip is dominated by inference anyway.

``detect`` returns the wire objects unchanged (pixel boxes, no geometry). The
pixel -> table mapping lives in :mod:`mfw.hardware.perception`, so the
detector stays a pure vision service and the geometry moves with the
orchestrator at Stage 4.

Two failures that used to share one message (review P3): a refused connection
means the service is not running; a *timeout* on a service that answered
``ping`` means it is running and busy -- almost always loading its model on
the first ``detect`` (NanoOWL / Florence load lazily, 10-40 s). Telling a
student to "start it first" for the second case sends them to restart a
service that was seconds from ready, which resets the load and reproduces the
error. So ``socket.timeout`` is reported as "answered ping but did not finish a
detect within N s; it may still be loading its model", and the first
``detect`` after :meth:`RemoteDetector.connect` is given
``first_call_timeout_s`` (60 s by default) instead of ``timeout_s``.
"""

from __future__ import annotations

import base64
import json
import socket
import time
from typing import Any, Sequence

from mfw.core.errors import PerceptionError
from mfw.core.interfaces import IObjectDetector
from mfw.core.types import CameraFrame
from mfw.utils.logging import get_logger

__all__ = ["RemoteDetector", "DetectorError", "DetectorTimeout"]

_log = get_logger("hardware.detector")


class DetectorError(PerceptionError):
    """The detector service is unreachable or answered with an error."""


class DetectorTimeout(DetectorError):
    """The service is up (it answered ping) but a request did not finish in time."""


class RemoteDetector(IObjectDetector):
    """Sends a vocabulary, receives pixel boxes."""

    def __init__(
        self,
        host: str,
        port: int,
        labels: Sequence[str] = (),
        min_score: float = 0.1,
        timeout_s: float = 5.0,
        first_call_timeout_s: float = 60.0,
    ) -> None:
        if not (0 < int(port) < 65536):
            raise ValueError(f"port out of range: {port}")
        if not 0.0 <= float(min_score) <= 1.0:
            raise ValueError("min_score must be in [0, 1]")
        if float(timeout_s) <= 0.0 or float(first_call_timeout_s) <= 0.0:
            raise ValueError("timeout_s and first_call_timeout_s must be > 0")
        self.host = str(host)
        self.port = int(port)
        self.labels: tuple[str, ...] = tuple(str(l).strip().lower() for l in labels if str(l).strip())
        self.min_score = float(min_score)
        self.timeout_s = float(timeout_s)
        self.first_call_timeout_s = max(float(first_call_timeout_s), self.timeout_s)
        """Receive timeout for the first ``detect`` after :meth:`connect`: the
        backend may be loading its model behind the server's lock."""
        self._detects_since_connect = 0
        self.backend: str | None = None
        self.last_reply: dict[str, Any] | None = None
        self.last_image_size: tuple[int, int] | None = None

    # ------------------------------------------------------------------

    @property
    def endpoint(self) -> str:
        """``host:port``."""
        return f"{self.host}:{self.port}"

    def _request(self, payload: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
        """One request/reply on a fresh connection.

        The connect and the reply have separate budgets on purpose: a connect
        that times out (host down, wrong IP) means *unreachable* and is
        reported with the start command, while a reply that times out on an
        accepted connection means *busy* and is :class:`DetectorTimeout`. A
        single ``create_connection(timeout=...)`` conflates the two, since both
        raise ``socket.timeout``.
        """
        timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        connect_timeout = min(timeout, self.timeout_s)
        try:
            sock = socket.create_connection((self.host, self.port), timeout=connect_timeout)
        except OSError as exc:  # includes socket.timeout during connect and ConnectionRefusedError
            raise DetectorError(
                f"detector service at {self.endpoint} unreachable: {exc}\n"
                "Start it first (scripted for the laptop lane):\n"
                f"    py -3.12 jetson/detector_service.py --backend scripted --port {self.port}"
            ) from exc
        try:
            with sock:
                sock.settimeout(timeout)
                sock.sendall(json.dumps(payload).encode("utf-8") + b"\n")
                reader = sock.makefile("rb")
                line = reader.readline()
        except socket.timeout as exc:
            # socket.timeout is an OSError; catch it first. The connection was
            # accepted, so the service is up: this is a slow (loading) backend,
            # not a missing one. Restarting it is the wrong advice.
            raise DetectorTimeout(
                f"detector service at {self.endpoint} answered ping but did not finish a "
                f"{payload.get('cmd', 'request')} within {timeout:.0f} s; it may still be "
                "loading its model (NanoOWL / Florence load on the first detect, 10-40 s). "
                "Do NOT restart it; wait and retry, or watch its log for 'ready'."
            ) from exc
        except OSError as exc:
            raise DetectorError(
                f"detector service at {self.endpoint} dropped the connection mid-request: {exc}"
            ) from exc
        if not line:
            raise DetectorError(f"detector service at {self.endpoint} closed the connection")
        try:
            reply = json.loads(line.decode("utf-8"))
        except ValueError as exc:
            raise DetectorError(f"detector service sent malformed JSON: {exc}") from exc
        if not isinstance(reply, dict):
            raise DetectorError("detector service reply is not a JSON object")
        if not reply.get("ok", False):
            raise DetectorError(f"detector service error: {reply.get('error', 'unknown')}")
        return reply

    def connect(self, retries: int = 3, backoff_s: float = 0.5) -> dict[str, Any]:
        """Ping until the service answers; records its backend name."""
        last: Exception | None = None
        for attempt in range(1, max(1, retries) + 1):
            try:
                reply = self._request({"cmd": "ping"})
                self.backend = str(reply.get("backend", "unknown"))
                self._detects_since_connect = 0
                _log.info("Connected to detector %s (backend=%s)", self.endpoint, self.backend)
                return reply
            except DetectorError as exc:
                last = exc
                if attempt < retries:
                    time.sleep(backoff_s * attempt)
        raise DetectorError(str(last))

    def is_ready(self) -> bool:
        """Whether :meth:`connect` has succeeded."""
        return self.backend is not None

    def close(self) -> None:
        """Nothing is held open between requests; kept for lifecycle symmetry."""
        self.backend = None

    # ------------------------------------------------------------------

    def detect(
        self,
        frame: CameraFrame | None = None,
        labels: Sequence[str] | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Detections as ``{label, confidence, bbox_px}`` dicts.

        With ``frame=None`` the service detects on its own latest webcam frame
        (the NanoOWL loop and the scripted backend); with a frame, its RGB is
        JPEG-encoded and sent. ``labels``/``min_score`` default to the
        constructor's vocabulary.
        """
        vocabulary = self.labels if labels is None else tuple(str(l) for l in labels)
        payload: dict[str, Any] = {
            "cmd": "detect",
            "labels": list(vocabulary),
            "min_score": self.min_score if min_score is None else float(min_score),
        }
        timeout = self.timeout_s
        if frame is not None:
            payload["jpeg"] = base64.b64encode(_encode_jpeg(frame)).decode("ascii")
            timeout = max(self.timeout_s, 30.0)  # a Florence fallback is slow
        if self._detects_since_connect == 0:
            # The backend may be loading its model behind the server's lock.
            timeout = max(timeout, self.first_call_timeout_s)
        reply = self._request(payload, timeout_s=timeout)
        self._detects_since_connect += 1
        self.last_reply = reply
        width, height = int(reply.get("width", 0)), int(reply.get("height", 0))
        if width > 0 and height > 0:
            self.last_image_size = (width, height)
        objects = reply.get("objects") or []
        out: list[dict[str, Any]] = []
        for obj in objects:
            try:
                box = [float(v) for v in obj["bbox_px"]]
                if len(box) != 4:
                    raise ValueError("bbox_px needs 4 numbers")
                out.append(
                    {
                        "label": str(obj.get("label", "")).strip().lower(),
                        "confidence": float(obj.get("confidence", 0.0)),
                        "bbox_px": box,
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                _log.warning("dropping malformed detection %r: %s", obj, exc)
        return out


def _encode_jpeg(frame: CameraFrame, quality: int = 85) -> bytes:
    try:
        import cv2
    except ImportError as exc:
        raise DetectorError(
            "opencv-python is required to send frames to the detector; "
            "`py -3.12 -m pip install opencv-python`"
        ) from exc
    import numpy as np

    bgr = np.ascontiguousarray(np.asarray(frame.rgb)[:, :, ::-1])
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise DetectorError("JPEG encoding failed")
    return buf.tobytes()
