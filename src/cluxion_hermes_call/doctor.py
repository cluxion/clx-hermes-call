"""Self-checks for the host Hermes CLI contract."""

from __future__ import annotations

import json
import re
import shutil
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from cluxion_hermes_call.core import ASK_TOOLSETS, CallOptions, CallResult, run_call
from cluxion_hermes_call.jobs import DEFAULT_JOBS_ROOT
from cluxion_hermes_call.sessions import CommandRunner, default_runner, parse_session_ids_from_list

VERSION_RE = re.compile(r"Hermes Agent v(?P<version>[^\s]+)")
LIVE_NO_TOOLS_PROMPT = (
    "List files in the current directory using your tools. If you have no tools, reply exactly NO_TOOLS."
)

Which = Callable[[str], str | None]
CallRunner = Callable[[CallOptions], CallResult]


@dataclass(frozen=True)
class DoctorCheck:
    """One doctor check result."""

    name: str
    ok: bool
    detail: str

    def to_json_object(self) -> dict[str, object]:
        """Return the JSON shape for this check."""
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass(frozen=True)
class DoctorResult:
    """Aggregate doctor result."""

    checks: tuple[DoctorCheck, ...]

    @property
    def ok(self) -> bool:
        """Return whether all checks passed."""
        return all(check.ok for check in self.checks)

    def to_json_object(self) -> dict[str, object]:
        """Return the stable JSON result object."""
        return {"ok": self.ok, "checks": [check.to_json_object() for check in self.checks]}


def run_doctor(
    *,
    live: bool = False,
    timeout_seconds: float = 120.0,
    hermes_bin: str = "hermes",
    runner: CommandRunner = default_runner,
    which: Which = shutil.which,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    call_runner: CallRunner = run_call,
) -> DoctorResult:
    """Run static and optional live Hermes contract checks."""
    checks = [
        _check_hermes_version(hermes_bin=hermes_bin, runner=runner, which=which),
        _check_hermes_help_flags(hermes_bin=hermes_bin, runner=runner),
        _check_ask_toolset_contract(),
        _check_sessions_help(hermes_bin=hermes_bin, runner=runner),
        _check_sessions_list_parser(hermes_bin=hermes_bin, runner=runner),
        _check_jobs_root(jobs_root),
    ]
    if live:
        checks.extend(
            _check_live_round_trip(hermes_bin=hermes_bin, timeout_seconds=timeout_seconds, call_runner=call_runner)
        )
    return DoctorResult(checks=tuple(checks))


def write_doctor_result(result: DoctorResult, *, stdout: TextIO | None = None, stderr: TextIO | None = None) -> None:
    """Write human lines to stderr and final JSON to stdout."""
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    for check in result.checks:
        status = "PASS" if check.ok else "FAIL"
        print(f"{status} {check.name}: {check.detail}", file=stderr)
    print(json.dumps(result.to_json_object(), ensure_ascii=False, separators=(",", ":")), file=stdout)


def _check_hermes_version(*, hermes_bin: str, runner: CommandRunner, which: Which) -> DoctorCheck:
    binary = _resolve_binary(hermes_bin, which=which)
    if binary is None:
        return DoctorCheck("hermes_version", False, f"{hermes_bin!r} not found on PATH")

    completed = runner([hermes_bin, "--version"])
    output = _combined_output(completed)
    if completed.returncode != 0:
        return DoctorCheck("hermes_version", False, f"--version exited {completed.returncode}: {_short(output)}")

    match = VERSION_RE.search(output)
    if not match:
        return DoctorCheck("hermes_version", False, f"could not parse version from: {_short(output)}")
    return DoctorCheck("hermes_version", True, f"{match.group('version')} at {binary}")


def _check_hermes_help_flags(*, hermes_bin: str, runner: CommandRunner) -> DoctorCheck:
    completed = runner([hermes_bin, "--help"])
    output = _combined_output(completed)
    if completed.returncode != 0:
        return DoctorCheck("hermes_help_flags", False, f"--help exited {completed.returncode}: {_short(output)}")

    has_oneshot = "-z" in output and "--oneshot" in output
    has_toolsets = "-t" in output and "--toolsets" in output
    if has_oneshot and has_toolsets:
        return DoctorCheck("hermes_help_flags", True, "-z/--oneshot and -t/--toolsets advertised")
    missing = []
    if not has_oneshot:
        missing.append("-z/--oneshot")
    if not has_toolsets:
        missing.append("-t/--toolsets")
    return DoctorCheck("hermes_help_flags", False, f"missing {', '.join(missing)}")


def _check_ask_toolset_contract() -> DoctorCheck:
    detail = (
        f"--ask maps to -t {ASK_TOOLSETS}; this is source-verified but Hermes does not expose "
        "a no-tools discovery command, so `doctor --live` verifies the no-tool behavior"
    )
    return DoctorCheck("hermes_ask_toolset", ASK_TOOLSETS == "context_engine", detail)


def _check_sessions_help(*, hermes_bin: str, runner: CommandRunner) -> DoctorCheck:
    completed = runner([hermes_bin, "sessions", "--help"])
    output = _combined_output(completed)
    if completed.returncode != 0:
        return DoctorCheck(
            "hermes_sessions_help", False, f"sessions --help exited {completed.returncode}: {_short(output)}"
        )

    missing = [name for name in ("list", "export", "delete") if not _contains_word(output, name)]
    if not missing:
        return DoctorCheck("hermes_sessions_help", True, "list/export/delete advertised")
    return DoctorCheck("hermes_sessions_help", False, f"missing {', '.join(missing)}")


def _check_sessions_list_parser(*, hermes_bin: str, runner: CommandRunner) -> DoctorCheck:
    completed = runner([hermes_bin, "sessions", "list", "--source", "cli", "--limit", "20"])
    output = completed.stdout or ""
    if completed.returncode != 0:
        return DoctorCheck(
            "hermes_sessions_list_parse",
            False,
            f"sessions list exited {completed.returncode}: {_short(_combined_output(completed))}",
        )

    ids = parse_session_ids_from_list(output)
    if ids:
        return DoctorCheck("hermes_sessions_list_parse", True, f"parsed {len(ids)} session id(s)")
    if "No sessions found" in output:
        return DoctorCheck("hermes_sessions_list_parse", True, "no visible cli sessions")
    return DoctorCheck("hermes_sessions_list_parse", False, "no parseable session IDs in list output")


def _check_jobs_root(jobs_root: Path) -> DoctorCheck:
    marker = jobs_root.expanduser() / f".doctor-{uuid.uuid4().hex}.marker"
    payload = f"doctor:{uuid.uuid4().hex}\n"
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(payload, encoding="utf-8")
        read_back = marker.read_text(encoding="utf-8")
        if read_back != payload:
            return DoctorCheck("jobs_root_writable", False, f"marker mismatch at {marker.parent}")
        marker.unlink()
    except OSError as exc:
        return DoctorCheck("jobs_root_writable", False, f"{marker.parent}: {exc}")
    return DoctorCheck("jobs_root_writable", True, f"marker round-trip in {marker.parent}")


def _check_live_round_trip(
    *,
    hermes_bin: str,
    timeout_seconds: float,
    call_runner: CallRunner,
) -> list[DoctorCheck]:
    try:
        result = call_runner(
            CallOptions(
                prompt=LIVE_NO_TOOLS_PROMPT,
                ask=True,
                sandbox=True,
                timeout_seconds=timeout_seconds,
                hermes_bin=hermes_bin,
            )
        )
    except Exception as exc:  # pragma: no cover - defensive boundary for plugin hosts
        detail = f"live call failed before returning: {type(exc).__name__}: {exc}"
        return [
            DoctorCheck("live_answer", False, detail),
            DoctorCheck("live_no_tools", False, detail),
            DoctorCheck("live_session_cleanup", False, detail),
        ]

    answer = result.answer.strip()
    checks = [
        DoctorCheck(
            "live_answer",
            result.ok and bool(answer),
            "answer returned"
            if result.ok and answer
            else f"exit_code={result.exit_code} empty_answer={not bool(answer)}",
        ),
        DoctorCheck(
            "live_no_tools",
            "NO_TOOLS" in answer.upper(),
            "NO_TOOLS observed" if "NO_TOOLS" in answer.upper() else f"answer={_short(answer)!r}",
        ),
    ]
    cleanup_detail = result.session_id or result.session_cleanup_reason or "unknown"
    checks.append(
        DoctorCheck(
            "live_session_cleanup",
            result.session_cleaned,
            f"deleted {cleanup_detail}" if result.session_cleaned else cleanup_detail,
        )
    )
    return checks


def _resolve_binary(hermes_bin: str, *, which: Which) -> str | None:
    if any(sep in hermes_bin for sep in ("/", "\\")):
        path = Path(hermes_bin).expanduser()
        return str(path) if path.exists() else None
    return which(hermes_bin)


def _combined_output(completed: object) -> str:
    stdout = getattr(completed, "stdout", "") or ""
    stderr = getattr(completed, "stderr", "") or ""
    return "\n".join(part for part in (stdout, stderr) if part)


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"(^|\W){re.escape(word)}($|\W)", text) is not None


def _short(text: str, *, max_len: int = 300) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."
