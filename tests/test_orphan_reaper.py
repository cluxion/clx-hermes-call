from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time


def test_sigterm_reaps_hermes_process_group(tmp_path) -> None:
    """Killing hermes-call must also kill the detached hermes child group."""
    fake_hermes = tmp_path / "hermes"
    pid_file = tmp_path / "child.pid"
    fake_hermes.write_text(f"#!/bin/sh\necho $$ > {pid_file}\nsleep 300\n", encoding="utf-8")
    fake_hermes.chmod(0o755)

    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from cluxion_hermes_call.core import CallOptions, run_call

        options = CallOptions(prompt="hi", hermes_bin=sys.argv[1], timeout_seconds=120, keep_session=True)
        run_call(options)
        """
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", script, str(fake_hermes)],
        cwd=str(tmp_path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        child_pid = None
        while time.monotonic() < deadline and child_pid is None:
            if pid_file.exists() and pid_file.read_text().strip():
                child_pid = int(pid_file.read_text().strip())
            else:
                time.sleep(0.1)
        assert child_pid is not None, "fake hermes child never started"
        assert child_pid != parent.pid

        parent.send_signal(signal.SIGTERM)
        parent.wait(timeout=10)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        os.kill(child_pid, signal.SIGKILL)
        raise AssertionError(f"hermes child {child_pid} survived parent SIGTERM")
    finally:
        if parent.poll() is None:
            parent.kill()


def test_sigterm_reaps_session_snapshot_process_group(tmp_path) -> None:
    """Killing hermes-call during session snapshot must also kill `hermes sessions list`."""
    fake_hermes = tmp_path / "hermes"
    pid_file = tmp_path / "session-list.pid"
    fake_hermes.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, time\n"
        f"open({str(pid_file)!r}, 'w', encoding='utf-8').write(str(__import__('os').getpid()))\n"
        "time.sleep(300)\n",
        encoding="utf-8",
    )
    fake_hermes.chmod(0o755)

    script = textwrap.dedent(
        """
        import sys
        from cluxion_hermes_call.sessions import capture_session_ids

        capture_session_ids(hermes_bin=sys.argv[1])
        """
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", script, str(fake_hermes)],
        cwd=str(tmp_path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        child_pid = None
        while time.monotonic() < deadline and child_pid is None:
            if pid_file.exists() and pid_file.read_text().strip():
                child_pid = int(pid_file.read_text().strip())
            else:
                time.sleep(0.1)
        assert child_pid is not None, "fake hermes sessions list never started"
        assert child_pid != parent.pid

        parent.send_signal(signal.SIGTERM)
        parent.wait(timeout=10)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        os.kill(child_pid, signal.SIGKILL)
        raise AssertionError(f"hermes sessions list child {child_pid} survived parent SIGTERM")
    finally:
        if parent.poll() is None:
            parent.kill()
