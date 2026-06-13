"""Command-line entry point for hermes-call."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

from cluxion_hermes_call import __version__
from cluxion_hermes_call.core import CallOptions, CallResult, run_call
from cluxion_hermes_call.doctor import run_doctor, write_doctor_result
from cluxion_hermes_call.jobs import gc_jobs


def add_call_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach hermes-call invocation flags to an argparse parser."""
    parser.add_argument("prompt", nargs="?", metavar="PROMPT", help="Prompt text, or '-' to read from stdin")
    parser.add_argument("-p", "--prompt", dest="prompt_alias", help="Prompt text alias")
    parser.add_argument("--ask", action="store_true", help="Answer-only mode using the verified no-tool toolset")
    parser.add_argument("-C", "--cd", dest="cwd", help="Run hermes with this subprocess working directory")
    parser.add_argument("--sandbox", action="store_true", help="Run in a fresh ~/.cluxion_hermes job work directory")
    parser.add_argument("--json", action="store_true", help="Print one JSON result object to stdout")
    parser.add_argument("--timeout", type=float, default=600.0, help="Timeout in seconds (default: 600)")
    parser.add_argument("--keep-session", action="store_true", help="Skip session self-cleanup")
    parser.add_argument("--keep", action="store_true", help="Keep the sandbox job directory")
    parser.add_argument("--toolsets", help="Advanced passthrough to hermes -t/--toolsets")
    parser.add_argument("-V", "--version", action="store_true", help="Show version and exit")


def build_call_parser(prog: str = "hermes-call") -> argparse.ArgumentParser:
    """Build the main hermes-call parser."""
    parser = argparse.ArgumentParser(prog=prog)
    add_call_arguments(parser)
    return parser


def build_gc_parser(prog: str = "hermes-call gc") -> argparse.ArgumentParser:
    """Build the gc subcommand parser."""
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("-V", "--version", action="store_true", help="Show version and exit")
    return parser


def build_doctor_parser(prog: str = "hermes-call doctor") -> argparse.ArgumentParser:
    """Build the doctor subcommand parser."""
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--live", action="store_true", help="Run one tiny live --ask model round-trip")
    parser.add_argument("--timeout", type=float, default=120.0, help="Live check timeout in seconds (default: 120)")
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
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    if ns.version:
        print(f"hermes-call {__version__}")
        return 0
    removed, kept = gc_jobs()
    print(f"removed={removed} kept={kept}")
    return 0


def _main_doctor(argv: list[str]) -> int:
    parser = build_doctor_parser()
    try:
        ns = parser.parse_args(argv)
        if ns.timeout <= 0:
            parser.error("--timeout must be greater than 0")
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    if ns.version:
        print(f"hermes-call {__version__}")
        return 0

    result = run_doctor(live=bool(ns.live), timeout_seconds=float(ns.timeout))
    write_doctor_result(result)
    return 0 if result.ok else 1


def _main_call(argv: list[str], *, stdin: TextIO) -> int:
    parser = build_call_parser()
    try:
        ns = parser.parse_args(argv)
        if ns.version:
            print(f"hermes-call {__version__}")
            return 0
        options = options_from_namespace(ns, stdin=stdin, parser=parser)
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
    if ns.timeout <= 0:
        parser.error("--timeout must be greater than 0")
    if ns.keep and not ns.sandbox:
        parser.error("--keep requires --sandbox")
    if ns.cwd and ns.sandbox:
        parser.error("--cd and --sandbox cannot be used together")
    if ns.ask and ns.toolsets:
        parser.error("--ask and --toolsets cannot be combined")

    cwd = Path(ns.cwd).expanduser() if ns.cwd else None
    return CallOptions(
        prompt=prompt,
        ask=bool(ns.ask),
        cwd=cwd,
        sandbox=bool(ns.sandbox),
        json_mode=bool(ns.json),
        timeout_seconds=float(ns.timeout),
        keep_session=bool(ns.keep_session),
        keep_job=bool(ns.keep),
        toolsets=ns.toolsets,
    )


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
        return stdin.read()
    return prompt


def _write_result(result: CallResult, *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(result.to_json_object(), ensure_ascii=False, separators=(",", ":")))
        return
    if result.answer:
        sys.stdout.write(result.answer)
        if not result.answer.endswith("\n"):
            sys.stdout.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
