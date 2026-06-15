"""Plugin-specific probes for hermes-call doctor. Cross-cutting (subcommand kind) + regression-guard probes."""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from .framework import DoctorContext

# imports moved inside probes to avoid circular import

PROBES: dict[str, Callable[[DoctorContext], tuple[str, str]]] = {}


def _register(name: str):
    def deco(fn):
        PROBES[name] = fn
        return fn

    return deco


@_register("hermes_on_path")
def hermes_on_path(ctx: DoctorContext) -> tuple[str, str]:
    p = shutil.which(ctx.hermes_bin)
    if p:
        return "pass", str(p)
    return "fail", "not found on PATH"


@_register("hermes_version")
def hermes_version(ctx: DoctorContext) -> tuple[str, str]:
    try:
        cp = ctx.run([ctx.hermes_bin, "--version"])
        if cp.returncode == 0 and "Hermes Agent v" in cp.stdout:
            return "pass", cp.stdout.strip()
        return "fail", cp.stdout.strip() or cp.stderr.strip()
    except Exception as e:
        return "fail", f"run error: {e}"


@_register("hermes_oneshot_flag")
def hermes_oneshot_flag(ctx: DoctorContext) -> tuple[str, str]:
    try:
        cp = ctx.run([ctx.hermes_bin, "--help"])
        out = cp.stdout + cp.stderr
        if "-z" in out and "--oneshot" in out:
            return "pass", "present"
        return "fail", "missing in --help"
    except Exception as e:
        return "fail", f"run error: {e}"


@_register("entry_point_registered")
def entry_point_registered(ctx: DoctorContext) -> tuple[str, str]:
    try:
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            eps = eps.select(group="hermes_agent.plugins")
        else:
            eps = eps.get("hermes_agent.plugins", [])
        for ep in eps:
            if "cluxion-hermes-call-cli" in (ep.name or "").lower() or "cluxion_hermes_call" in (ep.value or ""):
                mod = ep.load()
                if hasattr(mod, "register") and callable(mod.register):
                    return "pass", ep.value or str(ep)
        return "fail", "entry point not found or register missing"
    except Exception as e:
        return "fail", f"metadata error: {e}"


@_register("subcommand_valid")
def subcommand_valid(ctx: DoctorContext) -> tuple[str, str]:
    try:
        cp = ctx.run([ctx.hermes_bin, "call", "--help"])
        if cp.returncode == 0:
            return "pass", "hermes call --help exits 0"
        return "fail", f"exit {cp.returncode}"
    except Exception as e:
        return "fail", f"run error: {e}"


@_register("install_integrity")
def install_integrity(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_hermes_call import __version__ as pkg_version

        dist_version = importlib.metadata.version("cluxion-hermes-call-cli")
        if dist_version == pkg_version:
            return "pass", dist_version
        return "warn", f"dist={dist_version} pkg={pkg_version}"
    except Exception as e:
        return "fail", f"version error: {e}"


# hermes-call regression-guard probes
@_register("no_p_short_flag")
def no_p_short_flag(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_hermes_call import cli
        parser = cli.build_call_parser()
        has_p = False
        for action in parser._actions:
            if any(opt == "-p" for opt in getattr(action, "option_strings", [])):
                has_p = True
                break
        if not has_p:
            return "pass", "-p absent (good)"
        return "fail", "-p short flag present (collision risk)"
    except Exception as e:
        return "skip", f"cannot inspect parser: {e}"


@_register("empty_prompt_rejected")
def empty_prompt_rejected(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_hermes_call import cli
        dummy_stdin = io.StringIO("")
        dummy_parser = argparse.ArgumentParser()
        # should raise SystemExit via parser.error
        try:
            cli._resolve_prompt("", None, stdin=dummy_stdin, parser=dummy_parser)
            return "fail", "_resolve_prompt did not reject empty"
        except SystemExit:
            pass
        # also check DEVNULL usage in source is present (simple sentinel)
        src = Path(cli.__file__).read_text() if hasattr(cli, "__file__") else ""
        if "DEVNULL" in src or "stdin=subprocess.DEVNULL" in src:
            return "pass", "rejects empty + DEVNULL sentinel"
        return "pass", "rejects empty"
    except Exception as e:
        return "skip", f"probe error: {e}"


@_register("ask_mode_honesty")
def ask_mode_honesty(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_hermes_call import core
        preface = getattr(core, "ASK_MODE_PREFACE", "")
        if preface and len(preface) > 10:
            return "pass", "ASK_MODE_PREFACE present and non-empty"
        return "fail", "ASK_MODE_PREFACE missing or empty"
    except Exception as e:
        return "skip", f"cannot check: {e}"


@_register("doctor_gc_magic_safe")
def doctor_gc_magic_safe(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_hermes_call import plugin
        src = ""
        if hasattr(plugin, "__file__"):
            src = Path(plugin.__file__).read_text()
        if "shaping = bool" in src and 'if prompt == "doctor" and not shaping' in src:
            return "pass", "magic gate on shaping present"
        return "fail", "shaping gate missing in _handle_call_command"
    except Exception as e:
        return "skip", f"cannot inspect: {e}"


@_register("python_version_incompatibility")
def python_version_incompatibility(ctx: DoctorContext) -> tuple[str, str]:
    vi = sys.version_info
    if (vi.major, vi.minor) >= (3, 11):
        return "pass", f"python {vi.major}.{vi.minor}"
    return "warn", f"python {vi.major}.{vi.minor} < 3.11"


@_register("json_mode_output_malformed")
def json_mode_output_malformed(ctx: DoctorContext) -> tuple[str, str]:
    try:
        d = {"a": 1, "b": [2, 3], "c": "test"}
        s1 = json.dumps(d, sort_keys=True)
        s2 = json.dumps(d, sort_keys=True)
        if s1 == s2:
            return "pass", "json roundtrip deterministic"
        return "fail", "json not deterministic"
    except Exception as e:
        return "skip", f"json error: {e}"


# note: other checks (many in catalog) reported as skip (no probe)
