"""Hermes plugin entry for the `hermes call` command."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from cluxion_hermes_call import __version__
from cluxion_hermes_call.cli import add_call_arguments, options_from_namespace
from cluxion_hermes_call.config import default_model_help_line
from cluxion_hermes_call.core import MAX_TIMEOUT_SECONDS, run_call
from cluxion_hermes_call.doctor.framework import DoctorResult, render_json
from cluxion_hermes_call.doctor.framework import run_doctor as framework_run_doctor
from cluxion_hermes_call.doctor.live import live_checks
from cluxion_hermes_call.doctor.probes import PROBES
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

    register_command = getattr(ctx, "register_command", None)
    if callable(register_command):

        def _slash_hermes_call(raw_args: str) -> str:
            prompt = raw_args.strip()
            # `-` is a slash-command placeholder, not stdin mode.
            if not prompt or prompt == "-":
                return "Usage: /hermes-call <prompt>"
            from cluxion_hermes_call.cli import options_from_namespace

            options = options_from_namespace(
                argparse.Namespace(
                    prompt=prompt,
                    prompt_alias=None,
                    model=None,
                    ask=False,
                    cwd=str(Path.cwd()),
                    sandbox=False,
                    json=False,
                    timeout=None,
                    until_done=False,
                    max_iterations=8,
                    keep_session=False,
                    keep=False,
                    toolsets=None,
                    resume_session=None,
                ),
                stdin=sys.stdin,
                parser=argparse.ArgumentParser(prog="hermes call"),
            )
            result = run_call(options)
            return result.answer or f"(exit {result.exit_code})"

        def _slash_hermes_call_doctor(raw_args: str) -> str:
            del raw_args
            from importlib.resources import files

            catalog_path = files("cluxion_hermes_call.doctor") / "catalog.json"
            result = framework_run_doctor(
                cwd=Path.cwd(),
                catalog_path=Path(str(catalog_path)),
                probes=PROBES,
                plugin="hermes-call",
                version=__version__,
            )
            return render_json(result)

        register_command(
            "hermes-call",
            _slash_hermes_call,
            description="One-shot Hermes prompt (codex-exec style)",
            args_hint="<prompt>",
        )
        register_command(
            "hermes-call-doctor",
            _slash_hermes_call_doctor,
            description="Run hermes-call plugin doctor checks",
        )


def _setup_call_parser(parser: argparse.ArgumentParser) -> None:
    parser.epilog = default_model_help_line()
    add_call_arguments(parser)
    parser.add_argument("--live", action="store_true", help="With `doctor`, run one tiny live --ask round-trip")


def _handle_call_command(args: argparse.Namespace) -> int:
    if getattr(args, "version", False):
        print(f"hermes-call {__version__}")
        return 0
    prompt = getattr(args, "prompt", None)
    prompt_alias = getattr(args, "prompt_alias", None)
    ask = getattr(args, "ask", False)
    toolsets = getattr(args, "toolsets", None)
    json_mode = getattr(args, "json", False)
    until_done = getattr(args, "until_done", False)
    sandbox = getattr(args, "sandbox", False)
    shaping = bool(prompt_alias or ask or toolsets or json_mode or until_done or sandbox)
    if prompt == "gc" and not shaping:
        removed, kept = gc_jobs()
        print(f"removed={removed} kept={kept}")
        return 0
    if prompt == "doctor" and not shaping:
        from importlib.resources import files
        from pathlib import Path

        timeout = _plugin_resolve_timeout(getattr(args, "timeout", None), default=120.0)
        catalog_path = files("cluxion_hermes_call.doctor") / "catalog.json"
        result = framework_run_doctor(
            cwd=Path.cwd(),
            catalog_path=Path(str(catalog_path)),
            probes=PROBES,
            plugin="hermes-call",
            version=__version__,
        )
        live = getattr(args, "live", False)
        if live:
            live_results = live_checks(timeout)
            result = DoctorResult(
                plugin=result.plugin,
                version=result.version,
                checks=result.checks + tuple(live_results),
            )
        if getattr(args, "json", False):
            print(render_json(result))
        else:
            print(render_json(result), file=sys.stderr)  # text-ish via json for plugin path
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


def _plugin_resolve_timeout(raw: float | None, *, default: float) -> float:
    """Branch-specific timeout default for hosted doctor/live; reject non-finite."""
    timeout = default if raw is None else float(raw)
    if not math.isfinite(timeout) or timeout <= 0 or timeout > MAX_TIMEOUT_SECONDS:
        print(
            f"--timeout must be a finite number between 0 and {int(MAX_TIMEOUT_SECONDS)}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return timeout
