from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from unittest.mock import patch

from cluxion_hermes_call import core


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


def _reset_child_registry() -> tuple[set[int], set[int], bool, bool]:
    with core._live_processes_lock:
        saved_pids = set(core._live_processes)
        saved_installed = set(core._installed_signal_hooks)
        saved_atexit = core._atexit_registered
        saved_active = core._signal_cleanup_active
        core._live_processes.clear()
        core._installed_signal_hooks.clear()
        core._atexit_registered = False
        core._signal_cleanup_active = False
    return saved_pids, saved_installed, saved_atexit, saved_active


def _restore_child_registry(
    saved_pids: set[int],
    saved_installed: set[int],
    saved_atexit: bool,
    saved_active: bool,
) -> None:
    with core._live_processes_lock:
        core._live_processes.clear()
        core._live_processes.update(saved_pids)
        core._installed_signal_hooks.clear()
        core._installed_signal_hooks.update(saved_installed)
        core._atexit_registered = saved_atexit
        core._signal_cleanup_active = saved_active


def test_worker_thread_registration_does_not_mark_hooks_when_signal_fails() -> None:
    saved = _reset_child_registry()
    try:
        with (
            patch.object(core.atexit, "register") as register,
            patch.object(core.signal, "signal", side_effect=ValueError("signal only works in main thread")),
        ):
            worker = threading.Thread(target=core._register_child, args=(4242,))
            worker.start()
            worker.join(timeout=2)
            assert not worker.is_alive()
        register.assert_called_once_with(core._reap_live_processes)
        with core._live_processes_lock:
            assert 4242 in core._live_processes
            assert core._atexit_registered is True
            assert core._installed_signal_hooks == set()
    finally:
        _restore_child_registry(*saved)


def test_partial_signal_install_retries_only_missing() -> None:
    """SIGINT success + one-shot SIGTERM failure must retry only SIGTERM next time."""
    saved = _reset_child_registry()
    handlers: dict[int, object] = {}
    signal_calls: list[int] = []
    term_failures_left = 1

    def _install(signum: int, handler: object) -> None:
        nonlocal term_failures_left
        signal_calls.append(signum)
        if signum == signal.SIGTERM and term_failures_left > 0:
            term_failures_left -= 1
            raise ValueError("transient SIGTERM install failure")
        handlers[signum] = handler

    try:
        with (
            patch.object(core.atexit, "register") as register,
            patch.object(core.signal, "getsignal", return_value=signal.SIG_DFL),
            patch.object(core.signal, "signal", side_effect=_install),
        ):
            core._register_child(1001)
            with core._live_processes_lock:
                assert core._installed_signal_hooks == {signal.SIGINT}
            assert signal.SIGINT in handlers
            assert signal.SIGTERM not in handlers

            calls_after_first = list(signal_calls)
            core._register_child(1002)
            with core._live_processes_lock:
                assert core._installed_signal_hooks == {signal.SIGINT, signal.SIGTERM}
            assert signal.SIGTERM in handlers
            # Second registration retries only the missing signal.
            assert signal_calls[len(calls_after_first) :] == [signal.SIGTERM]
        register.assert_called_once_with(core._reap_live_processes)
    finally:
        _restore_child_registry(*saved)


def test_worker_thread_child_registration_adds_one_atexit_hook() -> None:
    saved = _reset_child_registry()
    try:
        with patch.object(core.atexit, "register") as register:
            for pid in (111, 222, 333):
                worker = threading.Thread(target=core._register_child, args=(pid,))
                worker.start()
                worker.join(timeout=2)
                assert not worker.is_alive()
        register.assert_called_once_with(core._reap_live_processes)
    finally:
        _restore_child_registry(*saved)


def test_signal_hook_preserves_previous_sig_ign() -> None:
    handlers: dict[int, object] = {}
    saved = _reset_child_registry()

    def _capture_handler(signum, handler):
        handlers[signum] = handler

    try:
        with (
            patch.object(core.atexit, "register"),
            patch.object(core.signal, "getsignal", return_value=signal.SIG_IGN),
            patch.object(core.signal, "signal", side_effect=_capture_handler),
        ):
            core._register_child(12345)
            handler = handlers[signal.SIGTERM]
            assert callable(handler)
            with (
                patch.object(core, "_reap_live_processes", return_value=True) as reap,
                patch.object(core.os, "kill") as kill,
            ):
                handler(signal.SIGTERM, None)
            reap.assert_called_once_with()
            kill.assert_not_called()
    finally:
        _restore_child_registry(*saved)


def test_signal_hook_skips_propagation_on_reentry() -> None:
    """Nested SIGTERM during active cleanup must not re-send default signal."""
    handlers: dict[int, object] = {}
    previous_calls: list[tuple[int, object]] = []

    def _previous(signo: int, frame: object) -> None:
        previous_calls.append((signo, frame))

    def _capture_handler(signum, handler):
        handlers[signum] = handler

    saved = _reset_child_registry()
    try:
        with (
            patch.object(core.atexit, "register"),
            patch.object(core.signal, "getsignal", return_value=_previous),
            patch.object(core.signal, "signal", side_effect=_capture_handler),
        ):
            core._register_child(54321)
            handler = handlers[signal.SIGTERM]
            assert callable(handler)
            with (
                patch.object(core, "_reap_live_processes", return_value=False) as reap,
                patch.object(core.os, "kill") as kill,
            ):
                handler(signal.SIGTERM, None)
            reap.assert_called_once_with()
            kill.assert_not_called()
            assert previous_calls == []
    finally:
        _restore_child_registry(*saved)


def test_term_ignoring_child_is_killed_and_reaped(tmp_path) -> None:
    child = tmp_path / "ignore_term.py"
    ready = tmp_path / "ready"
    child.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pathlib.Path(sys.argv[1]).write_text('ready', encoding='utf-8')\n"
        "time.sleep(300)\n",
        encoding="utf-8",
    )
    child.chmod(0o755)
    proc = subprocess.Popen([sys.executable, str(child), str(ready)], start_new_session=True)
    core._register_child(proc.pid)
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not ready.exists():
            time.sleep(0.01)
        assert ready.exists(), "child did not install SIGTERM handler"
        assert core._reap_live_processes() is True
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.05)
        assert proc.poll() is not None
        try:
            os.kill(proc.pid, 0)
            raise AssertionError("child still alive after reap")
        except ProcessLookupError:
            pass
        with core._live_processes_lock:
            assert proc.pid not in core._live_processes
    finally:
        with core._live_processes_lock:
            core._live_processes.discard(proc.pid)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=2)


def test_double_sigterm_during_grace_does_not_orphan_term_ignoring_child(tmp_path) -> None:
    """Regression: nested SIGTERM must not kill parent before owner reaps SIG_IGN child."""
    ready = tmp_path / "ready"
    child_pid_file = tmp_path / "child.pid"
    parent_script = textwrap.dedent(
        f"""
        import os
        import signal
        import subprocess
        import sys
        import time
        from pathlib import Path

        from cluxion_hermes_call import core

        ready = Path({str(ready)!r})
        child_pid_file = Path({str(child_pid_file)!r})

        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os, signal, sys, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "sys.stdout.write(str(os.getpid()) + chr(10)); "
                "sys.stdout.flush(); "
                "time.sleep(300)",
            ],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        assert child.stdout is not None
        line = child.stdout.readline().strip()
        child_pid = int(line)
        child_pid_file.write_text(str(child_pid), encoding="utf-8")
        core._register_child(child.pid)
        ready.write_text("ready", encoding="utf-8")
        # Stay alive so the installed SIGTERM handler owns cleanup.
        while True:
            time.sleep(1)
        """
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_script],
        cwd=str(tmp_path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    child_pid: int | None = None
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not ready.exists():
            time.sleep(0.01)
        assert ready.exists(), "parent did not signal readiness"
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if child_pid_file.exists() and child_pid_file.read_text(encoding="utf-8").strip():
                child_pid = int(child_pid_file.read_text(encoding="utf-8").strip())
                break
            time.sleep(0.01)
        assert child_pid is not None, "child pid never reported"
        assert child_pid != parent.pid

        parent.send_signal(signal.SIGTERM)
        time.sleep(0.05)
        if parent.poll() is None:
            parent.send_signal(signal.SIGTERM)

        # Owner cleanup uses ~0.5s TERM grace + drain; allow bounded headroom.
        parent.wait(timeout=5)
        # Parent must exit via SIGTERM (default propagation by cleanup owner).
        assert parent.returncode is not None
        assert parent.returncode == -signal.SIGTERM or parent.returncode == 128 + signal.SIGTERM

        cleanup_deadline = time.monotonic() + 3
        while time.monotonic() < cleanup_deadline:
            child_gone = False
            group_gone = False
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                child_gone = True
            try:
                os.killpg(child_pid, 0)
            except ProcessLookupError:
                group_gone = True
            except PermissionError:
                group_gone = False
            if child_gone and group_gone:
                return
            time.sleep(0.05)
        raise AssertionError(f"term-ignoring child {child_pid} or its process group survived double SIGTERM")
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=2)
        if child_pid is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(child_pid, signal.SIGKILL)
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.kill(child_pid, signal.SIGKILL)


def test_multiple_children_share_one_global_term_window() -> None:
    clock = 0.0
    sleep_calls: list[float] = []
    pids = {111, 222, 333}

    def _monotonic() -> float:
        return clock

    def _sleep(seconds: float) -> None:
        nonlocal clock
        sleep_calls.append(seconds)
        clock += seconds

    with core._live_processes_lock:
        saved_pids = set(core._live_processes)
        core._live_processes.clear()
        core._live_processes.update(pids)
    try:
        with (
            patch.object(core, "_TERM_POLL_TIMEOUT_SECONDS", 0.05),
            patch.object(core.time, "monotonic", side_effect=_monotonic),
            patch.object(core.time, "sleep", side_effect=_sleep),
            patch.object(core, "_process_group_alive", return_value=True),
            patch.object(core, "_pid_alive", return_value=False),
            patch.object(core.os, "waitpid", side_effect=ChildProcessError),
            patch.object(core.os, "killpg") as killpg,
        ):
            assert core._reap_live_processes() is True

        assert 0.05 <= clock < 0.06
        assert len(sleep_calls) <= 6
        for pid in pids:
            killpg.assert_any_call(pid, signal.SIGTERM)
            killpg.assert_any_call(pid, signal.SIGKILL)
    finally:
        with core._live_processes_lock:
            core._live_processes.clear()
            core._live_processes.update(saved_pids)


def test_reap_preserves_pids_registered_after_snapshot() -> None:
    late_pid = 424242
    with core._live_processes_lock:
        core._live_processes.clear()
        core._live_processes.add(111)

    def _killpg(pid, sig):
        if pid == 111 and sig == signal.SIGTERM:
            with core._live_processes_lock:
                core._live_processes.add(late_pid)
            raise ProcessLookupError
        if pid == late_pid:
            raise AssertionError("late pid should not be in snapshot cleanup")
        raise ProcessLookupError

    with (
        patch.object(core.os, "killpg", side_effect=_killpg),
        patch.object(core, "_process_group_alive", return_value=False),
        patch.object(core.os, "waitpid", side_effect=ChildProcessError),
    ):
        assert core._reap_live_processes() is True
    with core._live_processes_lock:
        assert late_pid in core._live_processes
        assert 111 not in core._live_processes
        core._live_processes.discard(late_pid)


def test_reap_empty_snapshot_returns_true_for_owner() -> None:
    with core._live_processes_lock:
        saved_pids = set(core._live_processes)
        core._live_processes.clear()
    try:
        assert core._reap_live_processes() is True
        with core._live_processes_lock:
            assert core._signal_cleanup_active is False
    finally:
        with core._live_processes_lock:
            core._live_processes.clear()
            core._live_processes.update(saved_pids)


def test_reap_reentry_returns_immediately() -> None:
    entered = threading.Event()
    release = threading.Event()
    results: list[object] = []

    def _blocking_killpg(pid, sig):
        entered.set()
        release.wait(timeout=2)
        raise ProcessLookupError

    with core._live_processes_lock:
        core._live_processes.add(999001)

    def worker():
        with (
            patch.object(core.os, "killpg", side_effect=_blocking_killpg),
            patch.object(core, "_process_group_alive", return_value=False),
            patch.object(core.os, "waitpid", side_effect=ChildProcessError),
        ):
            results.append(("owner", core._reap_live_processes()))

    t = threading.Thread(target=worker)
    t.start()
    assert entered.wait(timeout=2)
    results.append(("reentry", core._reap_live_processes()))
    release.set()
    t.join(timeout=3)
    assert ("owner", True) in results
    assert ("reentry", False) in results
    with core._live_processes_lock:
        core._live_processes.discard(999001)
        assert core._signal_cleanup_active is False
