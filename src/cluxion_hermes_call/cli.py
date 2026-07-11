"""Command-line entry point for hermes-call."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import TextIO

from cluxion_hermes_call import __version__
from cluxion_hermes_call.config import default_model_help_line
from cluxion_hermes_call.core import MAX_TIMEOUT_SECONDS, CallOptions, CallResult, run_call
from cluxion_hermes_call.doctor.framework import (
    DoctorResult,
    render_json,
    render_text,
)
from cluxion_hermes_call.doctor.framework import (
    run_doctor as framework_run_doctor,
)
from cluxion_hermes_call.doctor.live import live_checks
from cluxion_hermes_call.doctor.probes import PROBES
from cluxion_hermes_call.jobs import gc_jobs
from cluxion_hermes_call.sessions import gc_sessions


class UsageError(Exception):
    def __init__(self, message: str, *, error: str = "usage_error", hint: str) -> None:
        super().__init__(message)
        self.error = error
        self.message = message
        self.hint = hint


class JsonArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: object, json_errors: bool = False, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.json_errors = json_errors

    def error(self, message: str) -> None:
        if self.json_errors:
            raise UsageError(message, hint=f"Run `{self.prog} --help` for valid arguments.")
        super().error(message)


def add_call_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach hermes-call invocation flags to an argparse parser."""
    parser.add_argument("prompt", nargs="?", metavar="PROMPT", help="Prompt text, or '-' to read from stdin")
    parser.add_argument("--prompt", dest="prompt_alias", help="Prompt text (alternative to the positional PROMPT)")
    parser.add_argument("-m", "--model", help="Per-run Hermes model override passed to hermes -m")
    parser.add_argument(
        "--ask",
        action="store_true",
        help="Answer-only question mode (no file/terminal/write tools; read-only context retrieval)",
    )
    parser.add_argument("-C", "--cd", dest="cwd", help="Run hermes with this subprocess working directory")
    parser.add_argument("--sandbox", action="store_true", help="Run in a fresh ~/.cluxion_hermes job work directory")
    parser.add_argument("--json", action="store_true", help="Print one JSON result object to stdout")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout in seconds (default: 600 for calls; doctor --live default: 120)",
    )
    parser.add_argument("--until-done", action="store_true", help="Resume the owned Hermes session until TASK_COMPLETE")
    parser.add_argument("--max-iterations", type=int, default=8, help="Maximum --until-done turns (default: 8)")
    parser.add_argument("--keep-session", action="store_true", help="Skip session self-cleanup")
    parser.add_argument("--keep", action="store_true", help="Keep the sandbox job directory")
    parser.add_argument("--toolsets", help="Advanced passthrough to hermes -t/--toolsets")
    parser.add_argument(
        "-r",
        "--resume",
        dest="resume_session",
        metavar="SESSION_ID",
        help="Resume an existing Hermes session (hermes chat -Q --resume); the session is never garbage-collected",
    )
    parser.add_argument("-V", "--version", action="store_true", help="Show version and exit")


def build_call_parser(prog: str = "hermes-call", *, json_errors: bool = False) -> argparse.ArgumentParser:
    """Build the main hermes-call parser."""
    parser = JsonArgumentParser(prog=prog, epilog=default_model_help_line(), json_errors=json_errors)
    add_call_arguments(parser)
    return parser


def build_gc_parser(prog: str = "hermes-call gc") -> argparse.ArgumentParser:
    """Build the gc subcommand parser."""
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--sessions", action="store_true", help="Garbage-collect orphaned untitled CLI Hermes sessions")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete sessions (with --sessions); default is dry-run",
    )
    parser.add_argument(
        "--idle-minutes",
        type=int,
        default=10,
        help="Keep untitled sessions updated within this many minutes (default: 10)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip `hermes sessions optimize` after a real session deletion pass",
    )
    parser.add_argument("-V", "--version", action="store_true", help="Show version and exit")
    return parser


def build_doctor_parser(prog: str = "hermes-call doctor", *, json_errors: bool = False) -> argparse.ArgumentParser:
    """Build the doctor subcommand parser."""
    parser = JsonArgumentParser(prog=prog, json_errors=json_errors)
    parser.add_argument("--json", action="store_true", help="Print framework JSON to stdout")
    parser.add_argument("--live", action="store_true", help="Run one tiny live --ask model round-trip")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Live check timeout in seconds (default: 120)",
    )
    parser.add_argument("-V", "--version", action="store_true", help="Show version and exit")
    return parser


def main(argv: list[str] | None = None, stdin: TextIO | None = None) -> int:
    """Run hermes-call and return a process exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    stdin = sys.stdin if stdin is None else stdin

    if args and args[0] == "gc":
        return _main_gc(args[1:])
    if args and args[0] == "doctor":
        return _main_doctor(args[1:])
    return _main_call(args, stdin=stdin)


def _main_gc(argv: list[str]) -> int:
    parser = build_gc_parser()
    try:
        ns = parser.parse_args(argv)
        if ns.idle_minutes <= 0:
            parser.error("--idle-minutes must be greater than 0")
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    if ns.version:
        print(f"hermes-call {__version__}")
        return 0
    if ns.sessions:
        report = gc_sessions(
            dry_run=not ns.apply,
            idle_minutes=ns.idle_minutes,
            optimize=not ns.no_optimize,
        )
        print(
            "sessions "
            f"dry_run={report.dry_run} "
            f"deleted={report.deleted} "
            f"kept_named={report.kept_named} "
            f"skipped_recent={report.skipped_recent} "
            f"skipped_unknown={report.skipped_unknown} "
            f"failed={report.failed} "
            f"optimized={report.optimized}"
        )
        if report.error:
            print(report.error, file=sys.stderr)
            return 2
        return 0 if report.failed == 0 else 1

    removed, kept = gc_jobs()
    print(f"removed={removed} kept={kept}")
    return 0


def _main_doctor(argv: list[str]) -> int:
    parser = build_doctor_parser(json_errors="--json" in argv)
    try:
        ns = parser.parse_args(argv)
        timeout = _resolve_timeout(ns.timeout, default=120.0, parser=parser)
    except UsageError as exc:
        _write_error_json(error=exc.error, message=exc.message, hint=exc.hint, exit_code=2)
        return 2
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    if ns.version:
        print(f"hermes-call {__version__}")
        return 0

    # Framework-based doctor (always), append live checks when --live
    from importlib.resources import files
    from pathlib import Path

    catalog_path = files("cluxion_hermes_call.doctor") / "catalog.json"
    result = framework_run_doctor(
        cwd=Path.cwd(),
        catalog_path=Path(str(catalog_path)),
        probes=PROBES,
        plugin="hermes-call",
        version=__version__,
    )

    if ns.live:
        live_results = live_checks(timeout)
        result = DoctorResult(
            plugin=result.plugin,
            version=result.version,
            checks=result.checks + tuple(live_results),
        )

    if ns.json:
        print(render_json(result))
        return 0 if result.ok else 1

    # text to stderr
    from .doctor.framework import load_catalog

    cat = load_catalog(Path(str(catalog_path)))
    text = render_text(result, cat, verbose=False)
    print(text, file=sys.stderr)
    return 0 if result.ok else 1


def _main_call(argv: list[str], *, stdin: TextIO) -> int:
    parser = build_call_parser(json_errors="--json" in argv)
    try:
        ns = parser.parse_args(argv)
        if ns.version:
            print(f"hermes-call {__version__}")
            return 0
        options = options_from_namespace(ns, stdin=stdin, parser=parser)
    except UsageError as exc:
        _write_error_json(error=exc.error, message=exc.message, hint=exc.hint, exit_code=2)
        return 2
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    result = run_call(options)
    _write_result(result, json_mode=options.json_mode)
    return result.exit_code


def options_from_namespace(
    ns: argparse.Namespace,
    *,
    stdin: TextIO,
    parser: argparse.ArgumentParser,
) -> CallOptions:
    """Convert argparse state into validated core options."""
    prompt = _resolve_prompt(ns.prompt, ns.prompt_alias, stdin=stdin, parser=parser)
    timeout = _resolve_timeout(getattr(ns, "timeout", None), default=600.0, parser=parser)
    if ns.max_iterations <= 0:
        parser.error("--max-iterations must be greater than 0")
    if ns.keep and not ns.sandbox:
        parser.error("--keep requires --sandbox")
    if ns.cwd and ns.sandbox:
        parser.error("--cd and --sandbox cannot be used together")
    if ns.ask and ns.toolsets:
        parser.error("--ask and --toolsets cannot be combined")
    if ns.resume_session and ns.until_done:
        parser.error("--resume and --until-done cannot be combined (until-done owns its session)")

    cwd = Path(ns.cwd).expanduser() if ns.cwd else None
    return CallOptions(
        prompt=prompt,
        ask=bool(ns.ask),
        cwd=cwd,
        sandbox=bool(ns.sandbox),
        json_mode=bool(ns.json),
        timeout_seconds=timeout,
        keep_session=bool(ns.keep_session),
        keep_job=bool(ns.keep),
        toolsets=ns.toolsets,
        model=ns.model,
        until_done=bool(ns.until_done),
        max_iterations=int(ns.max_iterations),
        resume_session=ns.resume_session,
    )


def _resolve_timeout(
    raw: float | None,
    *,
    default: float,
    parser: argparse.ArgumentParser,
) -> float:
    """Apply branch-specific default, then reject non-finite / out-of-range values."""
    timeout = default if raw is None else float(raw)
    if not math.isfinite(timeout):
        parser.error("--timeout must be a finite number within the supported range")
    if timeout <= 0:
        parser.error("--timeout must be greater than 0")
    if timeout > MAX_TIMEOUT_SECONDS:
        parser.error(f"--timeout must be at most {int(MAX_TIMEOUT_SECONDS)}")
    return timeout


def _resolve_prompt(
    positional: str | None,
    alias: str | None,
    *,
    stdin: TextIO,
    parser: argparse.ArgumentParser,
) -> str:
    if positional and alias:
        parser.error("provide PROMPT or --prompt, not both")
    prompt = alias if alias is not None else positional
    if prompt is None:
        parser.error("PROMPT is required")
    if prompt == "-":
        try:
            prompt = stdin.read()
        except UnicodeDecodeError:
            parser.error("stdin is not valid UTF-8; pipe UTF-8 encoded text")
    if not prompt.strip():
        parser.error("PROMPT is empty")
    return prompt


def _write_result(result: CallResult, *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(result.to_json_object(), ensure_ascii=False, separators=(",", ":")))
        return
    if result.answer:
        sys.stdout.write(result.answer)
        if not result.answer.endswith("\n"):
            sys.stdout.write("\n")


def _write_error_json(*, error: str, message: str, hint: str, exit_code: int) -> None:
    print(
        json.dumps(
            {"ok": False, "error": error, "message": message, "hint": hint, "exit_code": exit_code},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
