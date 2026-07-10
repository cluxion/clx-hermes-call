from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from cluxion_hermes_call import core, jobs
from cluxion_hermes_call.core import CallOptions, HermesProcessResult, run_call
from cluxion_hermes_call.jobs import MARKER_FILE, create_job, gc_jobs
from cluxion_hermes_call.sessions import SessionCleanupReport, SessionSnapshot


def test_env_override_wins(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(jobs.HOME_ENV, str(tmp_path / "custom"))
    root = jobs.resolve_jobs_root()
    assert root == tmp_path / "custom" / "jobs"
    assert root.is_dir()


def test_falls_back_to_workspace_when_home_unwritable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(jobs.HOME_ENV, raising=False)
    monkeypatch.setattr(jobs, "DEFAULT_JOBS_ROOT", Path("/dev/null/impossible/jobs"))
    monkeypatch.chdir(tmp_path)
    root = jobs.resolve_jobs_root()
    assert root == tmp_path / ".hermes-call" / "jobs"


def test_raises_typed_error_with_hint_when_nothing_writable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(jobs.HOME_ENV, "/dev/null/impossible")
    with pytest.raises(jobs.JobRootUnwritableError) as exc:
        jobs.resolve_jobs_root()
    assert jobs.HOME_ENV in exc.value.hint


def _fake_ok_popen(monkeypatch) -> None:
    class FakeProcess:
        pid = 999991
        returncode = 0

        def communicate(self, timeout=None):
            return "ok", ""

    monkeypatch.setattr(core.subprocess, "Popen", lambda *a, **k: FakeProcess())


def test_run_call_sandbox_cleanup_uses_job_parent_under_env_root(tmp_path: Path, monkeypatch) -> None:
    """Successful run must delete job under env override root, not DEFAULT_JOBS_ROOT."""
    monkeypatch.setenv(jobs.HOME_ENV, str(tmp_path / "custom-home"))
    monkeypatch.setattr(jobs, "DEFAULT_JOBS_ROOT", tmp_path / "default-jobs")
    _fake_ok_popen(monkeypatch)

    result = run_call(CallOptions(prompt="hi", sandbox=True, keep_session=True))

    assert result.ok is True
    assert result.job_deleted is True
    assert result.job_dir is not None
    assert not Path(result.job_dir).exists()
    assert str(tmp_path / "custom-home" / "jobs") in result.job_dir


def test_until_done_sandbox_cleanup_uses_job_parent_under_env_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(jobs.HOME_ENV, str(tmp_path / "custom-home"))
    monkeypatch.setattr(jobs, "DEFAULT_JOBS_ROOT", tmp_path / "default-jobs")
    monkeypatch.setattr(core, "capture_session_ids", lambda **kwargs: SessionSnapshot(frozenset()))
    monkeypatch.setattr(
        core,
        "identify_created_session",
        lambda *a, **k: SessionCleanupReport(cleaned=False, reason="no_new_session"),
    )

    def fake_run(options, *, cwd, prompt, resume_session_id=None, timeout_seconds=None):
        return HermesProcessResult(stdout="done\nTASK_COMPLETE\n", stderr="", returncode=0, timed_out=False)

    monkeypatch.setattr(core, "_run_hermes_process_with_prompt", fake_run)

    result = run_call(CallOptions(prompt="hi", sandbox=True, until_done=True, keep_session=True))

    assert result.ok is True
    assert result.status == "complete"
    assert result.job_deleted is True
    assert result.job_dir is not None
    assert not Path(result.job_dir).exists()
    assert str(tmp_path / "custom-home" / "jobs") in result.job_dir


def test_gc_jobs_omitted_root_uses_env_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(jobs.HOME_ENV, str(tmp_path / "custom-home"))
    monkeypatch.setattr(jobs, "DEFAULT_JOBS_ROOT", tmp_path / "default-jobs")
    root = jobs.resolve_jobs_root()
    old = create_job(jobs_root=root)
    old_time = time.time() - 25 * 60 * 60
    os.utime(old.root / MARKER_FILE, (old_time, old_time))

    removed, kept = gc_jobs()

    assert removed == 1
    assert kept == 0
    assert not old.root.exists()


def test_gc_jobs_omitted_root_uses_workspace_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(jobs.HOME_ENV, raising=False)
    monkeypatch.setattr(jobs, "DEFAULT_JOBS_ROOT", Path("/dev/null/impossible/jobs"))
    monkeypatch.chdir(tmp_path)
    root = jobs.resolve_jobs_root()
    assert root == tmp_path / ".hermes-call" / "jobs"
    old = create_job(jobs_root=root)
    old_time = time.time() - 25 * 60 * 60
    os.utime(old.root / MARKER_FILE, (old_time, old_time))

    removed, _kept = gc_jobs()

    assert removed == 1
    assert not old.root.exists()


def test_gc_jobs_omitted_root_returns_zeros_when_root_unwritable(monkeypatch) -> None:
    monkeypatch.setenv(jobs.HOME_ENV, "/dev/null/impossible")
    removed, kept = gc_jobs()
    assert (removed, kept) == (0, 0)
