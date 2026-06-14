"""Session self-cleanup helpers for the public Hermes CLI."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SESSION_ID_AT_EOL_RE = re.compile(r"(?P<id>\d{8}_\d{6}_[0-9a-fA-F]+)\s*$")


@dataclass(frozen=True)
class SessionSnapshot:
    """A captured set of visible Hermes CLI session IDs."""

    ids: frozenset[str]
    ok: bool = True
    error: str | None = None


@dataclass(frozen=True)
class SessionCleanupReport:
    """Result of deleting the one session created by a run."""

    cleaned: bool
    reason: str | None = None
    session_id: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class SessionMetadata:
    """Exported metadata for one Hermes session."""

    session_id: str
    model: str | None = None
    cwd: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class SessionSelection:
    """Result of selecting one session from concurrent candidates."""

    session_id: str | None
    model: str | None
    reason: str | None


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def default_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a Hermes session-management command."""
    return subprocess.run(command, text=True, capture_output=True, check=False)


def capture_session_ids(
    *,
    hermes_bin: str = "hermes",
    runner: CommandRunner = default_runner,
    limit: int = 10000,
) -> SessionSnapshot:
    """Capture visible CLI session IDs through `hermes sessions list`."""
    completed = _run_session_command(
        [hermes_bin, "sessions", "list", "--source", "cli", "--limit", str(limit)],
        runner=runner,
    )
    if completed.returncode != 0:
        return SessionSnapshot(
            ids=frozenset(),
            ok=False,
            error=_short_error(completed.stderr or completed.stdout or f"exit {completed.returncode}"),
        )
    return SessionSnapshot(ids=frozenset(parse_session_ids_from_list(completed.stdout)))


def parse_session_ids_from_list(output: str) -> set[str]:
    """Parse IDs from the verified human table emitted by `hermes sessions list`."""
    ids: set[str] = set()
    for line in output.splitlines():
        match = SESSION_ID_AT_EOL_RE.search(line)
        if match:
            ids.add(match.group("id"))
    return ids


def cleanup_created_session(
    before: SessionSnapshot,
    after: SessionSnapshot,
    *,
    hermes_bin: str = "hermes",
    runner: CommandRunner = default_runner,
    expected_cwd: str | Path | None = None,
) -> SessionCleanupReport:
    """Delete exactly one newly-created session, refusing ambiguous diffs."""
    report = identify_created_session(
        before,
        after,
        hermes_bin=hermes_bin,
        runner=runner,
        expected_cwd=expected_cwd,
    )
    if report.session_id is None:
        return report

    deleted = delete_session(report.session_id, hermes_bin=hermes_bin, runner=runner)
    if deleted.cleaned:
        return SessionCleanupReport(cleaned=True, session_id=report.session_id, model=report.model)
    return SessionCleanupReport(
        cleaned=False,
        reason=deleted.reason,
        session_id=report.session_id,
        model=report.model,
    )


def identify_created_session(
    before: SessionSnapshot,
    after: SessionSnapshot,
    *,
    hermes_bin: str = "hermes",
    runner: CommandRunner = default_runner,
    expected_cwd: str | Path | None = None,
) -> SessionCleanupReport:
    """Identify exactly one newly-created session without deleting it."""
    if not before.ok:
        return SessionCleanupReport(cleaned=False, reason=f"before_list_failed:{before.error or 'unknown'}")
    if not after.ok:
        return SessionCleanupReport(cleaned=False, reason=f"after_list_failed:{after.error or 'unknown'}")

    new_ids = sorted(after.ids - before.ids)
    if not new_ids:
        return SessionCleanupReport(cleaned=False, reason="no_new_session")
    if len(new_ids) != 1:
        if expected_cwd is None:
            return SessionCleanupReport(cleaned=False, reason=f"multiple_new_sessions:{len(new_ids)}")
        selection = select_session_by_exported_cwd(
            new_ids,
            expected_cwd=expected_cwd,
            hermes_bin=hermes_bin,
            runner=runner,
        )
        if selection.session_id is None:
            reason = f"multiple_new_sessions:{len(new_ids)}"
            if selection.reason:
                reason = f"{reason};{selection.reason}"
            return SessionCleanupReport(cleaned=False, reason=reason)
        session_id = selection.session_id
        model = selection.model
    else:
        session_id = new_ids[0]
        model = fetch_session_model(session_id, hermes_bin=hermes_bin, runner=runner)

    return SessionCleanupReport(cleaned=False, reason=None, session_id=session_id, model=model)


def delete_session(
    session_id: str,
    *,
    hermes_bin: str = "hermes",
    runner: CommandRunner = default_runner,
) -> SessionCleanupReport:
    """Delete a known Hermes session id through the verified CLI command."""
    completed = _run_session_command([hermes_bin, "sessions", "delete", "--yes", session_id], runner=runner)
    combined_output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode == 0 and f"Deleted session '{session_id}'" in combined_output:
        return SessionCleanupReport(cleaned=True, session_id=session_id)
    return SessionCleanupReport(
        cleaned=False,
        reason=f"delete_failed:{_short_error(combined_output or f'exit {completed.returncode}')}",
        session_id=session_id,
    )


def fetch_session_model(
    session_id: str,
    *,
    hermes_bin: str = "hermes",
    runner: CommandRunner = default_runner,
) -> str | None:
    """Fetch a session's model using the verified JSONL export path."""
    return fetch_session_metadata(session_id, hermes_bin=hermes_bin, runner=runner).model


def fetch_session_metadata(
    session_id: str,
    *,
    hermes_bin: str = "hermes",
    runner: CommandRunner = default_runner,
) -> SessionMetadata:
    """Fetch a session's metadata using the verified JSONL export path."""
    completed = _run_session_command([hermes_bin, "sessions", "export", "-", "--session-id", session_id], runner=runner)
    if completed.returncode != 0:
        return SessionMetadata(
            session_id=session_id,
            error=_short_error(completed.stderr or completed.stdout or f"exit {completed.returncode}"),
        )
    for line in completed.stdout.splitlines():
        try:
            data: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("id") == session_id:
            model = data.get("model")
            cwd = data.get("cwd")
            return SessionMetadata(
                session_id=session_id,
                model=str(model) if model else None,
                cwd=str(cwd) if cwd else None,
            )
    return SessionMetadata(session_id=session_id, error="missing_export_record")


def select_session_by_exported_cwd(
    session_ids: list[str],
    *,
    expected_cwd: str | Path | None,
    hermes_bin: str = "hermes",
    runner: CommandRunner = default_runner,
) -> SessionSelection:
    """Select exactly one candidate whose exported cwd matches the run cwd."""
    normalized_expected = _normalize_cwd(expected_cwd)
    if normalized_expected is None:
        return SessionSelection(None, None, "no_expected_cwd")

    metadata: list[SessionMetadata] = []
    for session_id in session_ids:
        item = fetch_session_metadata(session_id, hermes_bin=hermes_bin, runner=runner)
        if item.error is not None:
            return SessionSelection(None, None, f"candidate_export_failed:{session_id}:{item.error}")
        metadata.append(item)

    matches = [item for item in metadata if _normalize_cwd(item.cwd) == normalized_expected]
    if len(matches) == 1:
        match = matches[0]
        return SessionSelection(match.session_id, match.model, None)
    return SessionSelection(None, None, f"cwd_match_count:{len(matches)}")


def _short_error(text: str, *, max_len: int = 300) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def _run_session_command(command: list[str], *, runner: CommandRunner) -> subprocess.CompletedProcess[str]:
    try:
        return runner(command)
    except OSError as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))


def _normalize_cwd(cwd: str | Path | None) -> str | None:
    if cwd is None:
        return None
    raw = str(cwd).strip()
    if not raw:
        return None
    try:
        return str(Path(raw).expanduser().resolve(strict=False))
    except OSError:
        return raw
