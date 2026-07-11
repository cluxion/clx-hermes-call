"""Core subprocess wrapper around the public Hermes CLI."""

from __future__ import annotations

import atexit
import contextlib
import difflib
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from cluxion_hermes_call.jobs import Job, JobRootUnwritableError, create_job, delete_job_dir, resolve_jobs_root
from cluxion_hermes_call.sessions import (
    SessionCleanupReport,
    SessionSnapshot,
    capture_session_ids,
    cleanup_created_session,
    delete_session,
    fetch_session_metadata,
    identify_created_session,
)

ASK_TOOLSETS = "context_engine"
ASK_MODE_PREFACE = "[ASK MODE] You have NO file, terminal, or write tools — only reasoning and read-only context retrieval. Never claim you created, edited, ran, or deleted anything. If the request requires tools you do not have, say so explicitly and stop.\n\n"
TASK_COMPLETE_MARKER = "TASK_COMPLETE"
WORK_REMAINS_PREFIX = "WORK_REMAINS:"
COMPLETION_CONTRACT = """

---
Completion contract for hermes-call --until-done:
End your reply with a final line exactly `TASK_COMPLETE` when the task is fully done.
If any work remains, end your reply with a final line `WORK_REMAINS: <what remains>`.
Do not use either marker except as the final line.
""".rstrip()
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|secret|password)(\s*[=:]\s*)(\S+)"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"),
    # labelless known-prefix keys (mirror backup secret-scan); one prefix group, body dropped by the sub-lambda; min lengths avoid over-redacting benign short strings
    re.compile(r"(sk-ant-)[A-Za-z0-9_-]{16,}"),
    re.compile(r"(gh[pousr]_)[A-Za-z0-9]{20,}"),
    re.compile(r"(xai-)[A-Za-z0-9-]{16,}"),
    re.compile(r"(hf_)[A-Za-z0-9]{20,}"),
    re.compile(r"(AKIA)[A-Z0-9]{16}"),
    re.compile(r"(xox[baprs]-)[A-Za-z0-9-]{10,}"),
]
MAX_PROMPT_BYTES = 256 * 1024
MAX_TIMEOUT_SECONDS = 86_400.0
_KILL_DRAIN_TIMEOUT_SECONDS = 0.5
_TERM_POLL_TIMEOUT_SECONDS = 0.5


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
    model: str | None = None
    until_done: bool = False
    max_iterations: int = 8
    hermes_bin: str = "hermes"
    resume_session: str | None = None


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
    status: str | None = None
    iterations: int | None = None
    work_log: tuple[str, ...] = ()
    last_work_remains: str | None = None
    error: str | None = None
    message: str | None = None
    hint: str | None = None

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
        if self.status is not None:
            payload["status"] = self.status
        if self.iterations is not None:
            payload["iterations"] = self.iterations
        # Expose session_id for until-done status or failed cleanup; never for
        # a successful one-shot solely because cleanup happened to know the id.
        if self.session_id is not None and (self.status is not None or not self.session_cleaned):
            payload["session_id"] = self.session_id
        if self.work_log:
            payload["work_log"] = list(self.work_log)
        if self.last_work_remains is not None:
            payload["last_work_remains"] = self.last_work_remains
        if self.error is not None:
            payload["error"] = self.error
        if self.message is not None:
            payload["message"] = self.message
        if self.hint is not None:
            payload["hint"] = self.hint
        return payload


def _sandbox_error_result(exc: JobRootUnwritableError) -> CallResult:
    return CallResult(
        ok=False,
        answer=f"sandbox unavailable: {exc}. Hint: {exc.hint}",
        model=None,
        duration_ms=0,
        session_cleaned=False,
        exit_code=2,
        status="sandbox_unwritable",
    )


def _structured_error_result(*, error: str, message: str, hint: str, exit_code: int = 2) -> CallResult:
    return CallResult(
        ok=False,
        answer=message,
        model=None,
        duration_ms=0,
        session_cleaned=False,
        exit_code=exit_code,
        status=error,
        error=error,
        message=message,
        hint=hint,
    )


def validate_call_options(options: CallOptions) -> CallResult | None:
    """Return a structured user error before any subprocess is started."""
    if "\0" in options.prompt:
        return _structured_error_result(
            error="invalid_prompt",
            message="PROMPT contains a null byte and cannot be passed to Hermes.",
            hint="Pass text input only; binary data must be encoded or stored in a file and summarized.",
        )
    prompt = _wrap_until_done_prompt(options.prompt) if options.until_done else options.prompt
    if options.ask:
        prompt = ASK_MODE_PREFACE + prompt
    try:
        prompt_bytes = len(prompt.encode("utf-8"))
    except UnicodeEncodeError:
        return _structured_error_result(
            error="invalid_prompt",
            message="PROMPT contains code points that cannot be encoded as UTF-8.",
            hint="Pass well-formed Unicode text; repair surrogateescape/binary input before invoking hermes-call.",
        )
    if prompt_bytes >= MAX_PROMPT_BYTES:
        return _structured_error_result(
            error="prompt_too_large",
            message=f"PROMPT is too large ({prompt_bytes} bytes).",
            hint=f"Limit is {MAX_PROMPT_BYTES} bytes because Hermes prompt passthrough uses argv; split the task or point Hermes at a file.",
        )
    try:
        # Order matters: comparisons first so -inf / +inf / huge values hit the
        # existing bound messages; NaN falls through (comparisons are False).
        if options.timeout_seconds <= 0:
            return _structured_error_result(
                error="invalid_timeout",
                message="--timeout must be greater than 0.",
                hint=f"Use a value between 0 and {int(MAX_TIMEOUT_SECONDS)} seconds.",
            )
        if options.timeout_seconds > MAX_TIMEOUT_SECONDS:
            return _structured_error_result(
                error="invalid_timeout",
                message=f"--timeout must be at most {int(MAX_TIMEOUT_SECONDS)} seconds.",
                hint="Use a bounded run and resume with --resume or --until-done if more work remains.",
            )
        if not math.isfinite(options.timeout_seconds):
            return _structured_error_result(
                error="invalid_timeout",
                message="--timeout must be a finite number within the supported range.",
                hint=f"Use a value between 0 and {int(MAX_TIMEOUT_SECONDS)} seconds.",
            )
    except (TypeError, ValueError, OverflowError):
        return _structured_error_result(
            error="invalid_timeout",
            message="--timeout must be a finite number within the supported range.",
            hint=f"Use a value between 0 and {int(MAX_TIMEOUT_SECONDS)} seconds.",
        )
    if options.max_iterations <= 0:
        return _structured_error_result(
            error="invalid_max_iterations",
            message="--max-iterations must be greater than 0.",
            hint="Use a positive integer.",
        )
    if options.resume_session is not None:
        if (
            not isinstance(options.resume_session, str)
            or not options.resume_session
            or any(character.isspace() for character in options.resume_session)
            or "\0" in options.resume_session
        ):
            return _structured_error_result(
                error="invalid_resume_session",
                message="--resume must be a non-empty session ID without whitespace or null bytes.",
                hint="Copy the exact ID from `hermes sessions list`.",
            )
        if options.sandbox:
            return _structured_error_result(
                error="resume_sandbox_conflict",
                message="--resume cannot be combined with a new sandbox workspace.",
                hint="Resume from the session's original project directory without --sandbox.",
            )
        if options.until_done:
            return _structured_error_result(
                error="resume_until_done_conflict",
                message="--resume cannot be combined with --until-done.",
                hint="Use --resume for one existing session or --until-done to let the caller own a new session.",
            )
    if options.cwd is not None and "\0" in str(options.cwd):
        return _structured_error_result(
            error="invalid_cwd",
            message="cwd contains a null byte and cannot be used as a working directory.",
            hint="Pass a filesystem path without embedded NUL bytes.",
        )
    if "\0" in options.hermes_bin:
        return _structured_error_result(
            error="invalid_hermes_bin",
            message="hermes_bin contains a null byte and cannot be executed.",
            hint="Pass a binary name or path without embedded NUL bytes.",
        )
    return None


def _resolve_call_cwd(cwd: Path | None) -> Path | CallResult:
    """Resolve cwd and require a usable directory before any process side effect."""
    try:
        if cwd is None:
            cwd = Path.cwd()
        resolved = cwd.expanduser().resolve(strict=False)
        if resolved.is_dir():
            return resolved
    except (OSError, RuntimeError, ValueError):
        pass
    return _structured_error_result(
        error="invalid_cwd",
        message="cwd cannot be resolved as a usable working directory.",
        hint="Pass an existing directory without circular symlinks or other unresolvable components.",
    )


def _resume_owner_error(options: CallOptions, cwd: Path) -> CallResult | None:
    """Fail closed unless exported resume metadata owns the requested cwd."""
    if options.resume_session is None:
        return None
    metadata = fetch_session_metadata(options.resume_session, hermes_bin=options.hermes_bin)
    if metadata.error is not None or metadata.cwd is None:
        return _structured_error_result(
            error="resume_owner_unknown",
            message="The resumed session's project directory could not be verified.",
            hint="Verify the session still exists and its original project directory is available.",
        )
    try:
        owner_cwd = Path(metadata.cwd).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return _structured_error_result(
            error="resume_owner_unknown",
            message="The resumed session's original project directory is unavailable.",
            hint="Restore the original directory or start a new session in this project.",
        )
    if not owner_cwd.is_dir():
        return _structured_error_result(
            error="resume_owner_unknown",
            message="The resumed session's original project path is not a directory.",
            hint="Restore the original directory or start a new session in this project.",
        )
    if owner_cwd != cwd:
        return _structured_error_result(
            error="resume_owner_mismatch",
            message="The resumed session belongs to a different project directory.",
            hint="Run from the exported session directory or start a new session here.",
        )
    return None


def run_call(options: CallOptions) -> CallResult:
    """Run Hermes once, clean up its session and sandbox when safe."""
    validation_error = validate_call_options(options)
    if validation_error is not None:
        return validation_error
    if options.until_done:
        return _run_until_done_call(options)

    start = time.monotonic()
    job: Job | None = None
    cwd = options.cwd

    if options.sandbox:
        try:
            job = create_job(jobs_root=resolve_jobs_root())
        except JobRootUnwritableError as exc:
            return _sandbox_error_result(exc)
        cwd = job.work
    resolved = _resolve_call_cwd(cwd)
    if isinstance(resolved, CallResult):
        return resolved
    cwd = resolved

    resume_error = _resume_owner_error(options, cwd)
    if resume_error is not None:
        return resume_error

    # A resumed session belongs to the user; hermes-call must never GC it.
    owns_session = not options.keep_session and options.resume_session is None
    before = SessionSnapshot(ids=frozenset(), ok=True)
    if owns_session:
        before = capture_session_ids(hermes_bin=options.hermes_bin)

    process_result = _run_hermes_process(options, cwd=cwd, resume_session_id=options.resume_session)

    cleanup_reason = (
        "resumed_session" if options.resume_session is not None else ("keep_session" if options.keep_session else None)
    )
    cleanup_report = SessionCleanupReport(cleaned=False, reason=cleanup_reason)
    if owns_session:
        after = capture_session_ids(hermes_bin=options.hermes_bin)
        cleanup_report = cleanup_created_session(before, after, hermes_bin=options.hermes_bin, expected_cwd=cwd)

    ok = process_result.returncode == 0 and not process_result.timed_out
    exit_code = _map_exit_code(process_result)
    job_deleted: bool | None = None
    if job is not None:
        if ok and not options.keep_job:
            decision = delete_job_dir(job.root, jobs_root=job.root.parent)
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


def _run_hermes_process(
    options: CallOptions, *, cwd: Path, resume_session_id: str | None = None
) -> HermesProcessResult:
    return _run_hermes_process_with_prompt(options, cwd=cwd, prompt=options.prompt, resume_session_id=resume_session_id)


def _run_hermes_process_with_prompt(
    options: CallOptions,
    *,
    cwd: Path,
    prompt: str,
    resume_session_id: str | None = None,
    timeout_seconds: float | None = None,
) -> HermesProcessResult:
    timeout = options.timeout_seconds if timeout_seconds is None else timeout_seconds
    if timeout <= 0:
        return HermesProcessResult(stdout="", stderr="overall timeout exceeded", returncode=124, timed_out=True)
    command = _build_hermes_command(options, prompt=prompt, resume_session_id=resume_session_id)
    try:
        process = _spawn_detached(
            command,
            cwd=str(cwd),
            env=_child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, ValueError) as exc:
        return HermesProcessResult(stdout="", stderr=f"failed to start hermes: {exc}", returncode=2, timed_out=False)

    try:
        # CPython communicate() drains stdout/stderr together, so a chatty
        # child cannot pipe-deadlock us while we wait for process exit.
        stdout, stderr = process.communicate(timeout=timeout)
        if resume_session_id is not None:
            stdout = _strip_chat_query_preamble(stdout)
        return HermesProcessResult(stdout=stdout, stderr=stderr, returncode=process.returncode or 0, timed_out=False)
    except subprocess.TimeoutExpired:
        stdout, stderr = _terminate_process_group(process, grace_seconds=_termination_grace(timeout))
        if resume_session_id is not None:
            stdout = _strip_chat_query_preamble(stdout)
        return HermesProcessResult(stdout=stdout, stderr=stderr, returncode=124, timed_out=True)
    finally:
        _unregister_child(process.pid)


def _build_hermes_command(
    options: CallOptions,
    *,
    prompt: str | None = None,
    resume_session_id: str | None = None,
) -> list[str]:
    actual_prompt = options.prompt if prompt is None else prompt
    if options.ask:
        actual_prompt = ASK_MODE_PREFACE + actual_prompt
    if resume_session_id is not None:
        command = [options.hermes_bin, "chat", "-Q", "--resume", resume_session_id]
        if options.model:
            command.extend(["-m", options.model])
        if options.ask:
            command.extend(["-t", ASK_TOOLSETS])
        elif options.toolsets is not None:
            command.extend(["-t", options.toolsets])
        command.extend(["-q", actual_prompt])
        return command

    command = [options.hermes_bin]
    if options.model:
        command.extend(["-m", options.model])
    command.extend(["-z", actual_prompt])
    if options.ask:
        command.extend(["-t", ASK_TOOLSETS])
    elif options.toolsets is not None:
        command.extend(["-t", options.toolsets])
    return command


def _run_until_done_call(options: CallOptions) -> CallResult:
    start = time.monotonic()
    deadline = start + options.timeout_seconds
    job: Job | None = None
    cwd = options.cwd

    if options.sandbox:
        try:
            job = create_job(jobs_root=resolve_jobs_root())
        except JobRootUnwritableError as exc:
            return _sandbox_error_result(exc)
        cwd = job.work
    resolved = _resolve_call_cwd(cwd)
    if isinstance(resolved, CallResult):
        return resolved
    cwd = resolved

    before = capture_session_ids(hermes_bin=options.hermes_bin)
    first_prompt = _wrap_until_done_prompt(options.prompt)
    process_results: list[HermesProcessResult] = []
    outputs: list[str] = []
    last_work_remains: str | None = None
    no_progress_abort = False
    owned_session = SessionCleanupReport(cleaned=False, reason=None)

    first_result = _run_hermes_process_with_prompt(
        options,
        cwd=cwd,
        prompt=first_prompt,
        timeout_seconds=max(0.0, deadline - time.monotonic()),
    )
    process_results.append(first_result)
    outputs.append(first_result.stdout)
    after = capture_session_ids(hermes_bin=options.hermes_bin)
    owned_session = identify_created_session(before, after, hermes_bin=options.hermes_bin, expected_cwd=cwd)

    marker = _parse_completion_marker(first_result.stdout)
    iterations = 1
    status = "complete" if marker == TASK_COMPLETE_MARKER else "incomplete"
    if marker and marker.startswith(WORK_REMAINS_PREFIX):
        last_work_remains = marker.removeprefix(WORK_REMAINS_PREFIX).strip()

    if first_result.returncode != 0 or first_result.timed_out or marker is None:
        status = "incomplete"

    while (
        status != "complete"
        and owned_session.session_id is not None
        and iterations < options.max_iterations
        and time.monotonic() < deadline
        and process_results[-1].returncode == 0
        and not process_results[-1].timed_out
        and last_work_remains is not None
    ):
        resume_prompt = _resume_until_done_prompt(last_work_remains)
        previous_work_remains = last_work_remains
        result = _run_hermes_process_with_prompt(
            options,
            cwd=cwd,
            prompt=resume_prompt,
            resume_session_id=owned_session.session_id,
            timeout_seconds=max(0.0, deadline - time.monotonic()),
        )
        process_results.append(result)
        outputs.append(result.stdout)
        iterations += 1
        marker = _parse_completion_marker(result.stdout)
        if marker == TASK_COMPLETE_MARKER:
            status = "complete"
            break
        if marker and marker.startswith(WORK_REMAINS_PREFIX):
            next_work_remains = marker.removeprefix(WORK_REMAINS_PREFIX).strip()
            last_work_remains = next_work_remains
            if _same_remaining_work(next_work_remains, previous_work_remains):
                no_progress_abort = True
                status = "incomplete"
                break
        else:
            last_work_remains = None
        if result.returncode != 0 or result.timed_out:
            status = "incomplete"
            break

    cleanup_report = SessionCleanupReport(cleaned=False, reason="keep_session" if options.keep_session else None)
    if owned_session.session_id is None:
        cleanup_report = SessionCleanupReport(cleaned=False, reason=owned_session.reason or "no_session_id")
    elif options.keep_session:
        cleanup_report = SessionCleanupReport(
            cleaned=False,
            reason="keep_session",
            session_id=owned_session.session_id,
            model=owned_session.model,
        )
    else:
        deleted = delete_session(owned_session.session_id, hermes_bin=options.hermes_bin)
        cleanup_report = SessionCleanupReport(
            cleaned=deleted.cleaned,
            reason=deleted.reason,
            session_id=owned_session.session_id,
            model=owned_session.model,
        )

    job_deleted = _cleanup_job(job, ok=status == "complete", keep_job=options.keep_job)
    duration_ms = int((time.monotonic() - start) * 1000)
    final_process = process_results[-1]
    exit_code = _until_done_exit_code(status=status, process_result=final_process)
    answer = _compose_until_done_answer(
        outputs,
        status=status,
        session_id=owned_session.session_id,
        last_work_remains=last_work_remains,
        max_iterations_reached=iterations >= options.max_iterations and status != "complete",
        timed_out=time.monotonic() >= deadline or final_process.timed_out,
        no_progress_abort=no_progress_abort,
    )

    _emit_diagnostics(
        options=options,
        process_result=final_process,
        cleanup_report=cleanup_report,
        exit_code=exit_code,
    )

    return CallResult(
        ok=status == "complete" and exit_code == 0,
        answer=answer,
        model=cleanup_report.model or options.model,
        duration_ms=duration_ms,
        session_cleaned=cleanup_report.cleaned,
        exit_code=exit_code,
        session_cleanup_reason=cleanup_report.reason,
        session_id=cleanup_report.session_id,
        job_dir=str(job.root) if job is not None else None,
        job_deleted=job_deleted,
        status=status,
        iterations=iterations,
        work_log=tuple(outputs),
        last_work_remains=last_work_remains,
    )


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    if env.get("CLUXION_HERMES_CALL_LIVE") == "1":
        env.pop("PYTEST_CURRENT_TEST", None)
    return env


def _wrap_until_done_prompt(prompt: str) -> str:
    return f"{prompt.rstrip()}{COMPLETION_CONTRACT}\n"


def _resume_until_done_prompt(last_work_remains: str | None) -> str:
    remains = last_work_remains or "the task is not complete"
    return f"Continue the remaining work. Last reported remaining work: {remains}{COMPLETION_CONTRACT}\n"


def _same_remaining_work(current: str, previous: str | None) -> bool:
    """Near-identical WORK_REMAINS across iterations means no progress.

    Models rephrase the same blocker; exact string equality misses that and
    the loop burns every iteration. difflib similarity >= 0.9 counts as same.
    """
    if previous is None:
        return False
    left = current.strip().casefold()
    right = previous.strip().casefold()
    if left == right:
        return True
    return difflib.SequenceMatcher(None, left, right).ratio() >= 0.9


def _marker_kind(line: str) -> str | None:
    """Classify a line as a completion marker, tolerating case and whitespace drift."""
    candidate = line.strip()
    if candidate.upper() == TASK_COMPLETE_MARKER:
        return TASK_COMPLETE_MARKER
    if candidate.upper().startswith(WORK_REMAINS_PREFIX.upper()):
        return f"{WORK_REMAINS_PREFIX} {candidate[len(WORK_REMAINS_PREFIX) :].strip()}"
    return None


def _parse_completion_marker(text: str) -> str | None:
    lines = [line.strip() for line in text.rstrip().splitlines() if line.strip()]
    if not lines:
        return None
    return _marker_kind(lines[-1])


def _strip_completion_marker(text: str) -> str:
    lines = [line for line in text.rstrip().splitlines() if _marker_kind(line) is None]
    return "\n".join(lines).strip()


def _compose_until_done_answer(
    outputs: list[str],
    *,
    status: str,
    session_id: str | None,
    last_work_remains: str | None,
    max_iterations_reached: bool,
    timed_out: bool,
    no_progress_abort: bool,
) -> str:
    body = "\n\n".join(part for part in (_strip_completion_marker(output) for output in outputs) if part)
    if status == "complete":
        return body

    reasons: list[str] = []
    if session_id is None:
        reasons.append("could not determine the Hermes session id, so continuation was not attempted")
    if max_iterations_reached:
        reasons.append("max iterations reached")
    if timed_out:
        reasons.append("timeout reached")
    if no_progress_abort:
        reasons.append("no progress observed")
    if last_work_remains:
        reasons.append(f"last WORK_REMAINS: {last_work_remains}")
    if not reasons:
        reasons.append("TASK_COMPLETE was not observed")
    note = "hermes-call status: incomplete (" + "; ".join(reasons) + ")"
    return f"{body}\n\n{note}".strip()


def _strip_chat_query_preamble(stdout: str) -> str:
    lines = stdout.splitlines()
    kept = [line for line in lines if not line.startswith("↻ Resumed session ") and not line.startswith("session_id: ")]
    while kept and not kept[0].strip():
        kept.pop(0)
    text = "\n".join(kept)
    if stdout.endswith("\n") and text:
        return f"{text}\n"
    return text


def _cleanup_job(job: Job | None, *, ok: bool, keep_job: bool) -> bool | None:
    if job is None:
        return None
    if ok and not keep_job:
        decision = delete_job_dir(job.root, jobs_root=job.root.parent)
        if not decision.allowed:
            _diagnose(f"sandbox cleanup skipped: {decision.reason}; job_dir={job.root}")
        return decision.allowed
    if ok and keep_job:
        return False
    _diagnose(f"sandbox kept after failure: {job.root}")
    return False


def _until_done_exit_code(*, status: str, process_result: HermesProcessResult) -> int:
    if process_result.timed_out:
        return 124
    if process_result.returncode != 0:
        return 1
    return 0 if status == "complete" else 1


_live_processes: set[int] = set()
_live_processes_lock = threading.RLock()
_atexit_registered = False
_installed_signal_hooks: set[int] = set()
_signal_cleanup_active = False
# Process-global spawn protocol: close the Popen→register gap under signals.
_pending_spawns = 0
_terminating = False
_deferred_signal: tuple[int, object] | None = None
# Test-only hook invoked after successful Popen, before PID registration.
_spawn_gap_hook = None


def _install_orphan_hooks() -> bool:
    """Install atexit + SIGINT/SIGTERM hooks (idempotent; safe to call often).

    Track installed signums separately: a transient failure on one signal must
    not prevent later attempts from installing the other. atexit is registered once.
    """
    global _atexit_registered
    with _live_processes_lock:
        if not _atexit_registered:
            atexit.register(_reap_live_processes)
            _atexit_registered = True
        missing = [signum for signum in (signal.SIGINT, signal.SIGTERM) if signum not in _installed_signal_hooks]
    if not missing:
        return True

    for signum in missing:
        previous = signal.getsignal(signum)

        def _handler(signo: int, frame: object, _previous=previous) -> None:
            global _terminating, _deferred_signal
            with _live_processes_lock:
                _terminating = True
                if _pending_spawns > 0:
                    # Defer until the last pending spawn registers (or fails).
                    if _deferred_signal is None:
                        _deferred_signal = (signo, frame)
                    return
            # Only the cleanup owner may propagate/default the signal.
            # A nested reentry during TERM grace returns False and must not
            # kill the parent before the owner finishes KILL/reap.
            if not _reap_live_processes():
                return
            if callable(_previous):
                try:
                    _previous(signo, frame)
                finally:
                    with _live_processes_lock:
                        _terminating = False
            elif _previous == signal.SIG_IGN:
                with _live_processes_lock:
                    _terminating = False
                return
            else:
                # Deferred delivery is re-signalled, so this always runs on main.
                signal.signal(signo, signal.SIG_DFL)
                os.kill(os.getpid(), signo)

        try:
            signal.signal(signum, _handler)
        except (ValueError, OSError):
            continue
        with _live_processes_lock:
            _installed_signal_hooks.add(signum)
    with _live_processes_lock:
        return {signal.SIGINT, signal.SIGTERM}.issubset(_installed_signal_hooks)


def _register_child(pid: int) -> None:
    """Track the child's process group so parent death cannot orphan it.

    start_new_session=True detaches hermes from our signals on purpose (we
    manage its lifetime), which also means Ctrl-C on hermes-call alone would
    leave the group running forever. SIGINT/SIGTERM/atexit now reap it.
    """
    _install_orphan_hooks()
    with _live_processes_lock:
        _live_processes.add(pid)


def _spawn_detached(command: list[str], **popen_kwargs: object) -> subprocess.Popen[str]:
    """Spawn a detached child under the process-global orphan-safe protocol.

    Install hooks first, bump a pending-spawn counter, Popen, register PID, then
    drop the counter. SIGTERM/SIGINT while pending>0 is deferred and re-delivered
    when the last pending spawn completes so every registered child is reaped.
    After a termination signal, further spawns are rejected.
    """
    global _pending_spawns, _deferred_signal
    popen_kwargs = dict(popen_kwargs)
    popen_kwargs["start_new_session"] = True
    with _live_processes_lock:
        if _terminating:
            raise OSError("spawn rejected: process is terminating")
    if not _install_orphan_hooks():
        raise OSError("detached spawn requires main-thread signal hook initialization")
    with _live_processes_lock:
        if _terminating:
            raise OSError("spawn rejected: process is terminating")
        _pending_spawns += 1
    try:
        process = subprocess.Popen(command, **popen_kwargs)  # type: ignore[arg-type]
        hook = _spawn_gap_hook
        if hook is not None:
            hook(process)
        _register_child(process.pid)
        return process
    finally:
        deferred: tuple[int, object] | None = None
        with _live_processes_lock:
            _pending_spawns -= 1
            if _pending_spawns == 0 and _deferred_signal is not None:
                deferred = _deferred_signal
                _deferred_signal = None
        if deferred is not None:
            _redeliver_deferred_signal(deferred[0], deferred[1])


def _redeliver_deferred_signal(signo: int, _frame: object) -> None:
    """Re-signal the process so Python dispatches the handler on the main thread."""
    os.kill(os.getpid(), signo)


def _unregister_child(pid: int) -> None:
    with _live_processes_lock:
        _live_processes.discard(pid)


def _reap_live_processes() -> bool:
    """TERM one snapshot with a single global grace window, then KILL + drain.

    Returns False only when cleanup is already active (reentrant nested call).
    Returns True for the owner on every normal path, including an empty snapshot.
    Never clears PIDs registered after the snapshot was taken. No per-child waits.
    """
    global _signal_cleanup_active
    with _live_processes_lock:
        if _signal_cleanup_active:
            return False
        _signal_cleanup_active = True
        snapshot = set(_live_processes)
    try:
        if not snapshot:
            return True
        for pid in snapshot:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(pid, signal.SIGTERM)

        term_deadline = time.monotonic() + _TERM_POLL_TIMEOUT_SECONDS
        survivors = set(snapshot)
        while survivors and time.monotonic() < term_deadline:
            survivors = {pid for pid in survivors if _process_group_alive(pid)}
            if not survivors:
                break
            time.sleep(0.01)
        survivors = {pid for pid in survivors if _process_group_alive(pid)}

        for pid in survivors:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(pid, signal.SIGKILL)

        drain_deadline = time.monotonic() + _KILL_DRAIN_TIMEOUT_SECONDS
        pending = set(snapshot)
        while pending and time.monotonic() < drain_deadline:
            for pid in list(pending):
                reaped = False
                try:
                    waited, _status = os.waitpid(pid, os.WNOHANG)
                    if waited == pid:
                        reaped = True
                except ChildProcessError:
                    reaped = True
                except ProcessLookupError:
                    reaped = True
                except OSError:
                    reaped = not _pid_alive(pid)
                if reaped or not _pid_alive(pid):
                    pending.discard(pid)
            if pending:
                time.sleep(0.01)

        with _live_processes_lock:
            _live_processes.difference_update(snapshot)
        return True
    finally:
        with _live_processes_lock:
            _signal_cleanup_active = False


def _process_group_alive(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return _pid_alive(pid)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _termination_grace(timeout_seconds: float) -> float:
    return min(5.0, max(0.5, timeout_seconds * 0.5))


def _terminate_process_group(process: subprocess.Popen[str], *, grace_seconds: float = 5.0) -> tuple[str, str]:
    stderr_chunks: list[str] = []
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError as exc:
        stderr_chunks.append(f"failed to terminate hermes process group {process.pid}: {exc}")
    try:
        stdout, stderr = process.communicate(timeout=grace_seconds)
        return stdout or "", _join_stderr(stderr, stderr_chunks)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            stderr_chunks.append(f"failed to kill hermes process group {process.pid}: {exc}")
        try:
            stdout, stderr = process.communicate(timeout=_KILL_DRAIN_TIMEOUT_SECONDS)
            return stdout or "", _join_stderr(stderr, stderr_chunks)
        except subprocess.TimeoutExpired:
            stderr_chunks.append(f"hermes process group {process.pid} did not exit after SIGKILL")
            return "", _join_stderr("", stderr_chunks)


def _join_stderr(stderr: str | None, chunks: list[str]) -> str:
    parts = [part for part in [stderr or "", *chunks] if part]
    return "\n".join(parts)


def _map_exit_code(process_result: HermesProcessResult) -> int:
    if process_result.timed_out:
        return 124
    if process_result.returncode == 0:
        return 0
    if process_result.returncode == 2:
        return 2
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
        _diagnose(
            sanitize_diagnostic(
                f"session cleanup skipped: {cleanup_report.reason}",
                prompt=options.prompt,
            )
        )


def sanitize_diagnostic(text: str, *, prompt: str) -> str:
    """Redact obvious secrets and the exact prompt from diagnostics."""
    # Span union on original coordinates so partial prompt/secret overlaps never
    # shift; emit once left-to-right. (start, end, replacement, prompt_flag)
    spans: list[tuple[int, int, str, bool]] = []
    for pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            replacement = f"{match.group(1)}{match.group(2) if len(match.groups()) > 1 else ''}[redacted]"
            spans.append((match.start(), match.end(), replacement, False))
    if prompt:
        # Non-overlapping exact occurrences (same as str.replace-all semantics).
        search_from = 0
        while True:
            idx = text.find(prompt, search_from)
            if idx < 0:
                break
            end = idx + len(prompt)
            spans.append((idx, end, "[prompt omitted]", True))
            search_from = end

    spans.sort(key=lambda s: (s[0], s[1]))
    merged: list[tuple[int, int, str, bool]] = []
    for start, end, replacement, prompt_flag in spans:
        if merged and start < merged[-1][1]:
            m_start, m_end, m_repl, m_prompt = merged[-1]
            new_end = max(m_end, end)
            if m_prompt or prompt_flag:
                # Prompt-linked: omit when the union must cover prompt text
                # (including partial overlaps that extend past a secret). A
                # prompt fully inside an existing secret span stays redacted.
                if m_prompt or new_end > m_end:
                    merged[-1] = (m_start, new_end, "[prompt omitted]", True)
                else:
                    merged[-1] = (m_start, m_end, m_repl, False)
            else:
                # Pure-secret overlap: stay redacted, keep first replacement.
                merged[-1] = (m_start, new_end, m_repl, False)
        else:
            merged.append((start, end, replacement, prompt_flag))

    parts: list[str] = []
    cursor = 0
    for start, end, replacement, _ in merged:
        if start < cursor:
            continue
        parts.append(text[cursor:start])
        parts.append(replacement)
        cursor = end
    parts.append(text[cursor:])
    redacted = "".join(parts)

    redacted = redacted.strip()
    if len(redacted) > 4000:
        return redacted[:3997] + "..."
    return redacted


def _diagnose(message: str) -> None:
    if message:
        print(message, file=sys.stderr)
