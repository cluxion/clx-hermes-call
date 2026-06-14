"""Hermes plugin entry for the `hermes call` command."""

from __future__ import annotations

import argparse
import sys

from cluxion_hermes_call import __version__
from cluxion_hermes_call.cli import add_call_arguments, options_from_namespace
from cluxion_hermes_call.config import default_model_help_line
from cluxion_hermes_call.core import run_call
from cluxion_hermes_call.doctor import run_doctor, write_doctor_result
from cluxion_hermes_call.jobs import gc_jobs


def register(ctx: object) -> None:
    """Register `hermes call ...` when hosted by Hermes."""
    register_cli_command = getattr(ctx, "register_cli_command", None)
    if not callable(register_cli_command):
        return
    register_cli_command(
        "call",
        "Run one prompt through the configured Hermes Agent",
        _setup_call_parser,
        _handle_call_command,
        description="Use Hermes Agent like an AI API / codex-exec subprocess wrapper.",
    )


def _setup_call_parser(parser: argparse.ArgumentParser) -> None:
    parser.epilog = default_model_help_line()
    add_call_arguments(parser)
    parser.add_argument("--live", action="store_true", help="With `doctor`, run one tiny live --ask round-trip")


def _handle_call_command(args: argparse.Namespace) -> int:
    if getattr(args, "version", False):
        print(f"hermes-call {__version__}")
        return 0
    if getattr(args, "prompt", None) == "gc" and not getattr(args, "prompt_alias", None):
        removed, kept = gc_jobs()
        print(f"removed={removed} kept={kept}")
        return 0
    if getattr(args, "prompt", None) == "doctor" and not getattr(args, "prompt_alias", None):
        result = run_doctor(
            live=bool(getattr(args, "live", False)), timeout_seconds=float(getattr(args, "timeout", 120.0))
        )
        write_doctor_result(result)
        if not result.ok:
            raise SystemExit(1)
        return 0
    if getattr(args, "live", False):
        print("--live is only valid with `hermes call doctor`", file=sys.stderr)
        raise SystemExit(2)

    parser = argparse.ArgumentParser(prog="hermes call")
    options = options_from_namespace(args, stdin=sys.stdin, parser=parser)

    result = run_call(options)
    if options.json_mode:
        import json

        print(json.dumps(result.to_json_object(), ensure_ascii=False, separators=(",", ":")))
    elif result.answer:
        sys.stdout.write(result.answer)
        if not result.answer.endswith("\n"):
            sys.stdout.write("\n")
    if result.exit_code:
        raise SystemExit(result.exit_code)
    return result.exit_code
