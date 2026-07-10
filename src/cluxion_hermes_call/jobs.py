"""Sandbox job lifecycle and fail-closed deletion gates."""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MARKER_FILE = ".cluxion_hermes_job"
HOME_ENV = "CLUXION_HERMES_CALL_HOME"
DEFAULT_JOBS_ROOT = Path.home() / ".cluxion_hermes" / "jobs"
GC_AGE_SECONDS = 24 * 60 * 60


class JobRootUnwritableError(OSError):
    """No writable location for sandbox jobs; carries a fix-it hint."""

    hint = (
        f"set {HOME_ENV} to a writable directory, or run without --sandbox; "
        "sandboxed hosts often block writes outside the workspace"
    )


def resolve_jobs_root() -> Path:
    """Pick a writable jobs root: env override, then home, then workspace-local."""
    override = os.environ.get(HOME_ENV, "").strip()
    candidates = (
        [Path(override).expanduser() / "jobs"]
        if override
        else [DEFAULT_JOBS_ROOT, Path.cwd() / ".hermes-call" / "jobs"]
    )
    errors: list[str] = []
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / f".probe-{os.getpid()}"
            probe.touch()
            probe.unlink()
            return candidate
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise JobRootUnwritableError("no writable sandbox job root: " + "; ".join(errors))


@dataclass(frozen=True)
class Job:
    """A sandbox job directory and its work directory."""

    job_id: str
    root: Path
    work: Path


@dataclass(frozen=True)
class DeleteDecision:
    """Deletion-gate decision for a job directory."""

    allowed: bool
    reason: str


def create_job(*, jobs_root: Path = DEFAULT_JOBS_ROOT) -> Job:
    """Create a marked sandbox job directory."""
    job_id = str(uuid.uuid4())
    root = jobs_root.expanduser() / job_id
    work = root / "work"
    work.mkdir(parents=True, exist_ok=False)
    marker = {
        "job_id": job_id,
        "pid": os.getpid(),
        "pid_create_time": time.time(),
    }
    (root / MARKER_FILE).write_text(json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8")
    return Job(job_id=job_id, root=root, work=work)


def delete_job_dir(job_dir: Path, *, jobs_root: Path = DEFAULT_JOBS_ROOT) -> DeleteDecision:
    """Delete a job directory only when all fail-closed safety checks pass."""
    decision = can_delete_job_dir(job_dir, jobs_root=jobs_root)
    if not decision.allowed:
        return decision
    shutil.rmtree(job_dir)
    return DeleteDecision(allowed=True, reason="deleted")


def can_delete_job_dir(job_dir: Path, *, jobs_root: Path = DEFAULT_JOBS_ROOT) -> DeleteDecision:
    """Return whether a job directory passes the deletion gate."""
    root = jobs_root.expanduser()
    try:
        root_real = root.resolve(strict=False)
        job_real = job_dir.resolve(strict=False)
    except OSError as exc:
        return DeleteDecision(False, f"resolve_failed:{exc}")

    if job_dir.is_symlink():
        return DeleteDecision(False, "job_dir_is_symlink")
    if job_real == root_real or not _is_strictly_under(job_real, root_real):
        return DeleteDecision(False, "path_escape")
    if not job_dir.is_dir():
        return DeleteDecision(False, "not_directory")

    marker_path = job_dir / MARKER_FILE
    if not marker_path.is_file():
        return DeleteDecision(False, "missing_marker")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return DeleteDecision(False, f"bad_marker:{exc}")

    if str(marker.get("job_id") or "") != job_dir.name:
        return DeleteDecision(False, "marker_job_id_mismatch")

    pid = _coerce_pid(marker.get("pid"))
    if pid is None:
        return DeleteDecision(False, "bad_marker_pid")
    if pid == os.getpid():
        return DeleteDecision(True, "current_process")
    if _pid_is_alive(pid):
        return DeleteDecision(False, "pid_alive")
    return DeleteDecision(True, "pid_dead")


def gc_jobs(*, jobs_root: Path | None = None, now: float | None = None) -> tuple[int, int]:
    """Prune marked job directories older than 24 hours.

    When ``jobs_root`` is omitted, resolve the live writable root. If root
    preparation raises ``JobRootUnwritableError``, return ``(0, 0)`` without crashing.
    """
    now = time.time() if now is None else now
    if jobs_root is None:
        try:
            root = resolve_jobs_root()
        except JobRootUnwritableError:
            return 0, 0
    else:
        root = jobs_root.expanduser()
    if not root.exists():
        return 0, 0

    removed = 0
    kept = 0
    for child in root.iterdir():
        marker = child / MARKER_FILE
        if not marker.exists():
            kept += 1
            continue
        try:
            marker_mtime = marker.stat().st_mtime
        except OSError:
            kept += 1
            continue
        if now - marker_mtime < GC_AGE_SECONDS:
            kept += 1
            continue
        decision = delete_job_dir(child, jobs_root=root)
        if decision.allowed:
            removed += 1
        else:
            kept += 1
    return removed, kept


def _is_strictly_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return path != parent


def _coerce_pid(value: Any) -> int | None:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
