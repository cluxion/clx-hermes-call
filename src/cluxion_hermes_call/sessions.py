"""Session self-cleanup helpers for the public Hermes CLI."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

SESSION_ID_AT_EOL_RE = re.compile(r"(?P<id>\d{8}_\d{6}_[0-9a-fA-F]+)\s*$")
# Fixed column widths from `hermes sessions list` table header (Preview / Last Active / Src / ID).
_LIST_PREVIEW_WIDTH = 50
_LIST_LAST_ACTIVE_WIDTH = 13
_SESSION_COMMAND_TIMEOUT_ENV = "CLUXION_HERMES_CALL_SESSION_TIMEOUT"
_DEFAULT_SESSION_COMMAND_TIMEOUT = 30.0


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
    started_at: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class SessionSelection:
    """Result of selecting one session from concurrent candidates."""

    session_id: str | None
    model: str | None
    reason: str | None


@dataclass(frozen=True)
class SessionGcMetadata:
    """Metadata needed to decide whether a CLI session is safe to GC."""

    session_id: str
    source: str | None = None
    title: str | None = None
    ended_at: float | None = None
    last_active: float | None = None
    error: str | None = None


@dataclass
class SessionGcReport:
    """Summary of a session GC pass."""

    dry_run: bool
    deleted: int = 0
    kept_named: int = 0
    skipped_recent: int = 0
    skipped_unknown: int = 0
    failed: int = 0
    optimized: bool = False
    deleted_ids: list[str] = field(default_factory=list)
    kept_named_ids: list[str] = field(default_factory=list)
    skipped_recent_ids: list[str] = field(default_factory=list)
    skipped_unknown_ids: list[str] = field(default_factory=list)
    failed_ids: list[str] = field(default_factory=list)
    error: str | None = None


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def default_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a Hermes session-management command."""
    from cluxion_hermes_call.core import (
        _register_child,
        _terminate_process_group,
        _termination_grace,
        _unregister_child,
    )

    timeout = _session_command_timeout()
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    _register_child(process.pid)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(command, process.returncode or 0, stdout, stderr)
    except subprocess.TimeoutExpired:
        stdout, stderr = _terminate_process_group(process, grace_seconds=_termination_grace(timeout))
        return subprocess.CompletedProcess(command, 124, stdout, _with_timeout_message(stderr, timeout))
    finally:
        _unregister_child(process.pid)


def _session_command_timeout() -> float:
    raw = os.environ.get(_SESSION_COMMAND_TIMEOUT_ENV)
    if raw is None:
        return _DEFAULT_SESSION_COMMAND_TIMEOUT
    try:
        timeout = float(raw)
    except ValueError:
        return _DEFAULT_SESSION_COMMAND_TIMEOUT
    return timeout if timeout > 0 else _DEFAULT_SESSION_COMMAND_TIMEOUT


def _with_timeout_message(stderr: str, timeout: float) -> str:
    message = f"session command timed out after {timeout:g}s"
    return f"{stderr}\n{message}" if stderr else message


def capture_session_ids(
    *,
    hermes_bin: str = "hermes",
    runner: CommandRunner = default_runner,
    limit: int = 50,
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
                started_at=_coerce_timestamp(data.get("started_at")),
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
    started_matches = [item for item in matches if item.started_at is not None]
    if started_matches:
        match = max(started_matches, key=lambda item: (item.started_at or 0.0, item.session_id))
        return SessionSelection(match.session_id, match.model, None)
    return SessionSelection(None, None, f"cwd_match_count:{len(matches)}")


def gc_sessions(
    *,
    dry_run: bool = True,
    idle_minutes: int = 10,
    hermes_bin: str = "hermes",
    runner: CommandRunner = default_runner,
    list_limit: int = 10_000,
    optimize: bool = True,
    now: float | None = None,
) -> SessionGcReport:
    """Garbage-collect orphaned untitled CLI sessions (fail-closed, dry-run by default)."""
    report = SessionGcReport(dry_run=dry_run)
    if idle_minutes <= 0:
        report.error = "idle_minutes_must_be_positive"
        return report

    rich_by_id = _load_rich_cli_sessions(hermes_bin=hermes_bin, runner=runner, list_limit=list_limit)
    list_row_by_id: dict[str, dict[str, str]] = {}
    if rich_by_id:
        session_ids = sorted(rich_by_id.keys())
    else:
        completed = _run_session_command(
            [hermes_bin, "sessions", "list", "--source", "cli", "--limit", str(list_limit)],
            runner=runner,
        )
        if completed.returncode != 0:
            report.error = (
                f"list_failed:{_short_error(completed.stderr or completed.stdout or f'exit {completed.returncode}')}"
            )
            return report

        session_ids = sorted(parse_session_ids_from_list(completed.stdout))
        if not session_ids:
            return report

        list_rows = parse_session_list_rows(completed.stdout)
        list_row_by_id = {row["id"]: row for row in list_rows}

    if not session_ids:
        return report
    current_time = time.time() if now is None else now
    idle_seconds = idle_minutes * 60

    # Fast path: the list table already proves recency. A recent session is
    # always kept, so skip it before paying a per-session export subprocess.
    remaining: list[str] = []
    for session_id in session_ids:
        row = list_row_by_id.get(session_id)
        if rich_by_id.get(session_id) is None and row is not None:
            seen = parse_relative_last_active(row.get("last_active", "?"), now=current_time)
            if seen is not None and (current_time - seen) < idle_seconds:
                report.skipped_recent += 1
                report.skipped_recent_ids.append(session_id)
                continue
        remaining.append(session_id)

    # Exports are independent subprocess calls; run them concurrently.
    metadata_by_id: dict[str, SessionGcMetadata] = {}
    if remaining:
        with ThreadPoolExecutor(max_workers=min(8, len(remaining))) as executor:
            futures = {
                executor.submit(
                    _resolve_session_gc_metadata,
                    session_id,
                    list_row=list_row_by_id.get(session_id),
                    rich=rich_by_id.get(session_id),
                    hermes_bin=hermes_bin,
                    runner=runner,
                ): session_id
                for session_id in remaining
            }
            for future in as_completed(futures):
                metadata_by_id[futures[future]] = future.result()

    for session_id in remaining:
        metadata = metadata_by_id[session_id]
        decision = _classify_session_for_gc(metadata, idle_seconds=idle_seconds, now=current_time)
        if decision == "kept_named":
            report.kept_named += 1
            report.kept_named_ids.append(session_id)
            continue
        if decision == "skipped_recent":
            report.skipped_recent += 1
            report.skipped_recent_ids.append(session_id)
            continue
        if decision == "skipped_unknown":
            report.skipped_unknown += 1
            report.skipped_unknown_ids.append(session_id)
            continue

        if dry_run:
            report.deleted += 1
            report.deleted_ids.append(session_id)
            continue

        deleted = delete_session(session_id, hermes_bin=hermes_bin, runner=runner)
        if deleted.cleaned:
            report.deleted += 1
            report.deleted_ids.append(session_id)
        else:
            report.failed += 1
            report.failed_ids.append(session_id)

    if not dry_run and report.deleted > 0 and optimize:
        report.optimized = optimize_session_store(hermes_bin=hermes_bin, runner=runner)

    return report


def optimize_session_store(
    *,
    hermes_bin: str = "hermes",
    runner: CommandRunner = default_runner,
) -> bool:
    """Run `hermes sessions optimize` to reclaim disk after bulk deletion."""
    completed = _run_session_command([hermes_bin, "sessions", "optimize"], runner=runner)
    return completed.returncode == 0


def parse_session_list_rows(output: str) -> list[dict[str, str]]:
    """Parse session rows from `hermes sessions list` table output."""
    rows: list[dict[str, str]] = []
    last_active_start = _LIST_PREVIEW_WIDTH + 1
    last_active_end = last_active_start + _LIST_LAST_ACTIVE_WIDTH
    for line in output.splitlines():
        id_match = SESSION_ID_AT_EOL_RE.search(line)
        if not id_match:
            continue
        session_id = id_match.group("id")
        last_active = "?"
        if len(line) >= last_active_end:
            value = line[last_active_start:last_active_end].strip()
            if value:
                last_active = value
        source = "cli"
        between = line[last_active_end : id_match.start()].strip()
        if between:
            source = between.split()[0]
        rows.append({"id": session_id, "last_active": last_active, "source": source})
    return rows


def parse_relative_last_active(value: str, *, now: float | None = None) -> float | None:
    """Convert a `hermes sessions list` relative timestamp to a unix timestamp."""
    current_time = time.time() if now is None else now
    text = value.strip()
    if not text or text == "?":
        return None
    if text == "just now":
        return current_time
    if text == "yesterday":
        return current_time - 36 * 3600
    if text.endswith("m ago"):
        try:
            minutes = int(text[:-5])
        except ValueError:
            return None
        return current_time - minutes * 60
    if text.endswith("h ago"):
        try:
            hours = int(text[:-5])
        except ValueError:
            return None
        return current_time - hours * 3600
    if text.endswith("d ago"):
        try:
            days = int(text[:-5])
        except ValueError:
            return None
        return current_time - days * 86400
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return None
    return parsed.timestamp()


def _classify_session_for_gc(metadata: SessionGcMetadata, *, idle_seconds: float, now: float) -> str:
    if metadata.error is not None:
        return "skipped_unknown"
    if metadata.source not in (None, "cli"):
        return "skipped_unknown"
    if _session_has_title(metadata.title):
        return "kept_named"
    if metadata.ended_at is not None:
        return "delete"
    if metadata.last_active is None:
        return "skipped_unknown"
    if now - metadata.last_active < idle_seconds:
        return "skipped_recent"
    return "delete"


def _session_has_title(title: str | None) -> bool:
    return bool(title and str(title).strip())


def _resolve_session_gc_metadata(
    session_id: str,
    *,
    list_row: dict[str, str] | None,
    rich: dict[str, Any] | None,
    hermes_bin: str,
    runner: CommandRunner,
) -> SessionGcMetadata:
    if rich is not None:
        return SessionGcMetadata(
            session_id=session_id,
            source=str(rich.get("source")) if rich.get("source") else None,
            title=str(rich["title"]) if rich.get("title") else None,
            ended_at=_coerce_timestamp(rich.get("ended_at")),
            last_active=_coerce_timestamp(rich.get("last_active")),
        )

    exported = _fetch_exported_session_record(session_id, hermes_bin=hermes_bin, runner=runner)
    if exported is not None:
        last_active = _coerce_timestamp(exported.get("last_active"))
        if last_active is None:
            messages = exported.get("messages") or []
            if messages:
                timestamps = [_coerce_timestamp(item.get("timestamp")) for item in messages]
                known = [item for item in timestamps if item is not None]
                last_active = max(known) if known else None
        if last_active is None and list_row is not None:
            last_active = parse_relative_last_active(list_row.get("last_active", "?"))
        if last_active is None:
            last_active = _coerce_timestamp(exported.get("started_at"))

        title = exported.get("title")
        return SessionGcMetadata(
            session_id=session_id,
            source=str(exported.get("source")) if exported.get("source") else None,
            title=str(title) if title else None,
            ended_at=_coerce_timestamp(exported.get("ended_at")),
            last_active=last_active,
        )

    source = list_row.get("source") if list_row else None
    return SessionGcMetadata(session_id=session_id, source=source, error="missing_export_record")


def _fetch_exported_session_record(
    session_id: str,
    *,
    hermes_bin: str,
    runner: CommandRunner,
) -> dict[str, Any] | None:
    completed = _run_session_command([hermes_bin, "sessions", "export", "-", "--session-id", session_id], runner=runner)
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        try:
            data: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("id") == session_id:
            return data
    return None


def _load_rich_cli_sessions(
    *,
    hermes_bin: str,
    runner: CommandRunner,
    list_limit: int,
) -> dict[str, dict[str, Any]]:
    rich = _load_sqlite_cli_sessions(list_limit=list_limit)
    if rich is not None:
        return rich

    db = _try_open_session_db()
    if db is None:
        return {}
    try:
        sessions = db.list_sessions_rich(source="cli", limit=list_limit)
    except Exception:
        return {}
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()
    return {str(item["id"]): item for item in sessions if item.get("id")}


def _load_sqlite_cli_sessions(*, list_limit: int) -> dict[str, dict[str, Any]] | None:
    path = _session_db_path()
    if path is None or not path.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            session_columns = _table_columns(con, "sessions")
            message_columns = _table_columns(con, "messages")
            required = {"id", "source", "title", "ended_at", "started_at"}
            if not required.issubset(session_columns) or not {"session_id", "timestamp"}.issubset(message_columns):
                return None
            rows = con.execute(
                """
                select
                    s.id,
                    s.source,
                    s.title,
                    s.ended_at,
                    coalesce(max(m.timestamp), s.started_at) as last_active
                from sessions s
                left join messages m on m.session_id = s.id
                where s.source = ?
                group by s.id
                order by last_active desc, s.id desc
                limit ?
                """,
                ("cli", list_limit),
            ).fetchall()
        finally:
            con.close()
    except (OSError, sqlite3.Error):
        return None
    return {
        str(row["id"]): {
            "id": row["id"],
            "source": row["source"],
            "title": row["title"],
            "ended_at": row["ended_at"],
            "last_active": row["last_active"],
        }
        for row in rows
        if row["id"]
    }


def _session_db_path() -> Path | None:
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        return Path(hermes_home).expanduser() / "state.db"
    home = Path.home()
    if not str(home):
        return None
    return home / ".hermes" / "state.db"


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in con.execute(f"pragma table_info({table})")}


def _try_open_session_db() -> Any | None:
    try:
        from hermes_state import SessionDB
    except ImportError:
        return None
    try:
        return SessionDB()
    except Exception:
        return None


def _coerce_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            try:
                return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return None
    return None


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
