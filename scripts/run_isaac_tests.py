"""Run the Isaac-dependent test suites inside the Isaac Sim interpreter.

Isaac Sim replaces parts of the process's stdio and can terminate the
interpreter during shutdown, which makes ``python.bat -m pytest`` lose its
output. Invoking ``pytest.main`` in-process and writing the report to a file
keeps the results regardless of how the app tears down.

Usage:
    python.bat scripts/run_isaac_tests.py [pytest args...]
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT = REPO_ROOT / "logs" / "isaac_test_report.txt"


def main() -> int:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:] or ["tests/", "-m", "isaac"]
    args = [*args, "-v", "--no-header", "-p", "no:cacheprovider"]

    # Isaac Sim parses sys.argv when SimulationApp is constructed and rejects
    # anything it does not recognise -- pytest's own flags included. Left in
    # place, "-m isaac" reaches omni.kit as "Ill formed parameter: -m", which
    # tears the app down mid-fixture and leaves a truncated report and a
    # misleading exit code 0. pytest.main() takes its arguments explicitly, so
    # clearing argv costs nothing.
    sys.argv = [sys.argv[0]]

    with REPORT.open("w", encoding="utf-8") as fh:
        original_stdout, original_stderr = sys.stdout, sys.stderr

        class Tee:
            """Mirror output to the report file and the console."""

            def __init__(self, *streams):
                self.streams = streams

            def write(self, data):
                for stream in self.streams:
                    try:
                        stream.write(data)
                        stream.flush()
                    except Exception:
                        pass
                return len(data)

            def flush(self):
                for stream in self.streams:
                    try:
                        stream.flush()
                    except Exception:
                        pass

            # pytest's terminal reporter probes the stream for these; report
            # non-tty so it emits plain output rather than ANSI control codes,
            # which keeps the report file readable.
            def isatty(self):
                return False

            def fileno(self):
                return self.streams[0].fileno()

            @property
            def encoding(self):
                return getattr(self.streams[0], "encoding", "utf-8")

            def writelines(self, lines):
                for line in lines:
                    self.write(line)

        sys.stdout = Tee(original_stdout, fh)
        sys.stderr = Tee(original_stderr, fh)

        class IncrementalReporter:
            """Write each failure to the report the moment it occurs.

            pytest emits its FAILURES section only at the end of the session,
            after fixture teardown. Tearing down a session-scoped Isaac fixture
            shuts down the app, which can terminate the process -- taking the
            entire summary with it and leaving a report that shows FAILED with
            no explanation. Recording eagerly means a crash during shutdown can
            no longer hide why a test failed.
            """

            def __init__(self, stream):
                self.stream = stream
                self.failures = []

            def pytest_runtest_logreport(self, report):
                if report.failed:
                    self.failures.append(report.nodeid)
                    self.stream.write(
                        f"\n{'=' * 70}\nFAILURE: {report.nodeid} ({report.when})\n"
                        f"{'=' * 70}\n{report.longreprtext}\n"
                    )
                    self.stream.flush()

        try:
            import pytest

            reporter = IncrementalReporter(fh)
            code = pytest.main(
                [str(REPO_ROOT / a) if a.startswith("tests/") else a for a in args],
                plugins=[reporter],
            )
            fh.write(f"\nPYTEST_EXIT_CODE={int(code)}\n")
            fh.write(f"FAILED_TESTS={len(reporter.failures)}\n")
            for nodeid in reporter.failures:
                fh.write(f"  FAILED {nodeid}\n")
        except BaseException:
            fh.write("\nRUNNER CRASHED\n" + traceback.format_exc())
            code = 99
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr

    print(f"\nReport written to {REPORT}")
    # os._exit avoids Isaac Sim's atexit teardown, which can abort the process
    # and mask the exit code we actually care about.
    sys.stdout.flush()
    os._exit(int(code))


if __name__ == "__main__":
    main()
