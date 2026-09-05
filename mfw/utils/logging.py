"""Structured logging.

Pure stdlib + NumPy. This module must never import Isaac Sim.

Two sinks, deliberately separate:

* A human-readable console stream for watching a run.
* A machine-readable JSONL event stream, one object per line, capturing
  detections, chosen grasps, motion plans, joint states, contacts, errors,
  timings and planner decisions.

The JSONL stream is the one that matters. Manipulation failures are rarely
reproducible by staring at a terminal; they are diagnosed by replaying what the
robot actually perceived and decided, which requires structured records rather
than formatted strings.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

__all__ = ["EventLogger", "get_logger", "configure_logging", "JsonSafeEncoder"]

_CONFIGURED = False


class JsonSafeEncoder(json.JSONEncoder):
    """JSON encoder that understands NumPy and falls back to ``repr``.

    Every value in this framework's logs originates from NumPy at some point, so
    without this the event stream dies on the first ``float32``. Never raising is
    intentional: a logging failure must not take down a running robot.
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        if isinstance(o, Path):
            return str(o)
        if hasattr(o, "to_log"):
            return o.to_log()
        return repr(o)


def configure_logging(level: str = "INFO", console: bool = True) -> None:
    """Set up the root console handler exactly once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger("mfw")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False
    if console:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the ``mfw`` root."""
    return logging.getLogger(f"mfw.{name}" if not name.startswith("mfw") else name)


class EventLogger:
    """Append-only JSONL event sink.

    Thread-safe because perception, control and the policy client may all emit
    from different threads. Writes are line-buffered and flushed according to
    config so a crash still leaves a usable trace on disk.
    """

    def __init__(
        self,
        log_dir: str | Path = "logs",
        filename: str = "events.jsonl",
        run_id: str | None = None,
        flush_every: int = 1,
        console_level: str = "INFO",
        console: bool = True,
    ) -> None:
        configure_logging(console_level, console)
        self._log = get_logger("events")
        self.run_id = run_id or f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        self._dir = Path(log_dir) / self.run_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / filename

        self._lock = threading.Lock()
        self._flush_every = max(1, int(flush_every))
        self._since_flush = 0
        self._closed = False
        self._seq = 0

        self._fh = self._path.open("a", encoding="utf-8")
        self._log.info("Event log: %s", self._path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def directory(self) -> Path:
        """Run-specific directory; image dumps and replays go alongside the JSONL."""
        return self._dir

    def emit(self, event_type: str, payload: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        """Write one event.

        Swallows its own errors by design: a malformed payload must never
        propagate into the control loop.
        """
        if self._closed:
            return
        record = {
            "seq": 0,
            "t_wall": time.time(),
            "run_id": self.run_id,
            "event": event_type,
        }
        if payload:
            record.update(payload)
        if kwargs:
            record.update(kwargs)

        try:
            with self._lock:
                self._seq += 1
                record["seq"] = self._seq
                self._fh.write(json.dumps(record, cls=JsonSafeEncoder) + "\n")
                self._since_flush += 1
                if self._since_flush >= self._flush_every:
                    self._fh.flush()
                    self._since_flush = 0
        except Exception as exc:  # pragma: no cover - defensive
            self._log.warning("Failed to write event %r: %s", event_type, exc)

    @contextmanager
    def timed(self, event_type: str, **fields: Any) -> Iterator[dict[str, Any]]:
        """Time a block and emit its duration, including on failure.

        Yields a mutable dict so the body can attach results that are only known
        partway through, e.g. how many waypoints a planner produced.
        """
        extra: dict[str, Any] = {}
        start = time.perf_counter()
        try:
            yield extra
        except Exception as exc:
            self.emit(
                event_type,
                {
                    **fields,
                    **extra,
                    "duration_s": time.perf_counter() - start,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
        else:
            self.emit(
                event_type,
                {**fields, **extra, "duration_s": time.perf_counter() - start, "ok": True},
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:  # pragma: no cover - defensive
                pass

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
