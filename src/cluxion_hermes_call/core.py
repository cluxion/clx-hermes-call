"""Core subprocess wrapper around the public Hermes CLI."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from cluxion_hermes_call.jobs import Job, create_job, delete_job_dir
from cluxion_hermes_call.sessions import (
    SessionCleanupReport,
    SessionSnapshot,
    capture_session_ids,
    cleanup_created_session,
)

ASK_TOOLSETS = "context_engine"
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|secret|password)(\s*[=:]\s*)(\S+)"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"),
]


@dataclass(frozen=True)
class CallOptions:
    """User-facing hermes-call options."""

    prompt: str
    ask: bool = False
    cwd: Path | None = None
    sandbox: bool = False
    json_mode: bool = False
    timeout_seconds: float = 600.0
    keep_session: bool = False
    keep_job: bool = False
    toolsets: str | None = None
    hermes_bin: str = "hermes"


@dataclass(frozen=True)
class HermesProcessResult:
    """Raw subprocess result."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


@dataclass(frozen=True)
class CallResult:
    """Final wrapper result."""

    ok: bool
    answer: str
    model: str | None
    duration_ms: int
    session_cleaned: bool
    exit_code: int
    session_cleanup_reason: str | None = None
    session_id: str | None = None
    job_dir: str | None = None
    job_deleted: bool | None = None

    def to_json_object(self) -> dict[str, object]:
        """Return the stable JSON result object."""
        payload: dict[str, object] = {
            "ok": self.ok,
            "answer": self.answer,
            "model": self.model,
            "duration_ms": self.duration_ms,
            "session_cleaned": self.session_cleaned,
            "exit_code": self.exit_code,
        }
        if not self.session_cleaned and self.session_cleanup_reason is not None:
            payload["session_cleanup_reason"] = self.session_cleanup_reason
        return payload


def run_call(options: CallOptions) -> CallResult:
    """Run Hermes once, clean up its session and sandbox when safe."""
    start = time.monotonic()
    job: Job | None = None
    cwd = options.cwd

    if options.sandbox:
        job = create_job()
        cwd = job.work
    elif cwd is None:
        cwd = Path.cwd()
    cwd = cwd.expanduser().resolve(strict=False)

    before = SessionSnapshot(ids=frozenset(), ok=True)
    if not options.keep_session:
        before = capture_session_ids(hermes_bin=options.hermes_bin)

    process_result = _run_hermes_process(options, cwd=cwd)

    cleanup_report = SessionCleanupReport(cleaned=False, reason="keep_session" if options.keep_session else None)
    if not options.keep_session:
        after = capture_session_ids(hermes_bin=options.hermes_bin)
        cleanup_report = cleanup_created_session(before, after, hermes_bin=options.hermes_bin, expected_cwd=cwd)

    ok = process_result.returncode == 0 and not process_result.timed_out
    exit_code = _map_exit_code(process_result)
    job_deleted: bool | None = None
    if job is not None:
        if ok and not options.keep_job:
            decision = delete_job_dir(job.root)
            job_deleted = decision.allowed
            if not decision.allowed:
                _diagnose(f"sandbox cleanup skipped: {decision.reason}; job_dir={job.root}")
        elif ok and options.keep_job:
            job_deleted = False
        else:
            job_deleted = False
            _diagnose(f"sandbox kept after failure: {job.root}")

    duration_ms = int((time.monotonic() - start) * 1000)
    _emit_diagnostics(
        options=options,
        process_result=process_result,
        cleanup_report=cleanup_report,
        exit_code=exit_code,
    )

    return CallResult(
        ok=ok,
        answer=process_result.stdout,
        model=cleanup_report.model,
        duration_ms=duration_ms,
        session_cleaned=cleanup_report.cleaned,
        exit_code=exit_code,
        session_cleanup_reason=cleanup_report.reason,
        session_id=cleanup_report.session_id,
        job_dir=str(job.root) if job is not None else None,
        job_deleted=job_deleted,
    )


def _run_hermes_process(options: CallOptions, *, cwd: Path) -> HermesProcessResult:
    command = _build_hermes_command(options)
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=_child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        return HermesProcessResult(stdout="", stderr=f"failed to start hermes: {exc}", returncode=1, timed_out=False)

    try:
        stdout, stderr = process.communicate(timeout=options.timeout_seconds)
        return HermesProcessResult(stdout=stdout, stderr=stderr, returncode=process.returncode or 0, timed_out=False)
    except subprocess.TimeoutExpired:
        stdout, stderr = _terminate_process_group(process)
        return HermesProcessResult(stdout=stdout, stderr=stderr, returncode=124, timed_out=True)


def _build_hermes_command(options: CallOptions) -> list[str]:
    command = [options.hermes_bin, "-z", options.prompt]
    if options.ask:
        command.extend(["-t", ASK_TOOLSETS])
    elif options.toolsets is not None:
        command.extend(["-t", options.toolsets])
    return command


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    if env.get("CLUXION_HERMES_CALL_LIVE") == "1":
        env.pop("PYTEST_CURRENT_TEST", None)
    return env


def _terminate_process_group(process: subprocess.Popen[str]) -> tuple[str, str]:
    stderr_chunks: list[str] = []
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError as exc:
        stderr_chunks.append(f"failed to terminate hermes process group {process.pid}: {exc}")
    try:
        stdout, stderr = process.communicate(timeout=5)
        return stdout or "", _join_stderr(stderr, stderr_chunks)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            stderr_chunks.append(f"failed to kill hermes process group {process.pid}: {exc}")
        stdout, stderr = process.communicate()
        return stdout or "", _join_stderr(stderr, stderr_chunks)


def _join_stderr(stderr: str | None, chunks: list[str]) -> str:
    parts = [part for part in [stderr or "", *chunks] if part]
    return "\n".join(parts)


def _map_exit_code(process_result: HermesProcessResult) -> int:
    if process_result.timed_out:
        return 124
    if process_result.returncode == 0:
        return 0
    return 1


def _emit_diagnostics(
    *,
    options: CallOptions,
    process_result: HermesProcessResult,
    cleanup_report: SessionCleanupReport,
    exit_code: int,
) -> None:
    if process_result.timed_out:
        _diagnose(f"hermes timed out after {options.timeout_seconds:g}s; child process group was terminated")
    elif exit_code != 0:
        _diagnose(f"hermes exited with code {process_result.returncode}")

    if process_result.stderr and (process_result.timed_out or exit_code != 0):
        _diagnose(sanitize_diagnostic(process_result.stderr, prompt=options.prompt))

    if not cleanup_report.cleaned and cleanup_report.reason not in {None, "keep_session", "no_new_session"}:
        _diagnose(f"session cleanup skipped: {cleanup_report.reason}")


def sanitize_diagnostic(text: str, *, prompt: str) -> str:
    """Redact obvious secrets and the exact prompt from diagnostics."""
    redacted = text.replace(prompt, "[prompt omitted]") if prompt else text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(
            lambda match: f"{match.group(1)}{match.group(2) if len(match.groups()) > 1 else ''}[redacted]", redacted
        )
    redacted = redacted.strip()
    if len(redacted) > 4000:
        return redacted[:3997] + "..."
    return redacted


def _diagnose(message: str) -> None:
    if message:
        print(message, file=sys.stderr)
