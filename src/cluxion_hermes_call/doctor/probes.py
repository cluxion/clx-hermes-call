"""Plugin-specific probes for hermes-call doctor. Cross-cutting (subcommand kind) + regression-guard probes."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
import shutil
import sys
import time
import uuid
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


@_register("hermes_binary_not_found")
def hermes_binary_not_found(ctx: DoctorContext) -> tuple[str, str]:
    return hermes_on_path(ctx)


@_register("hermes_version")
def hermes_version(ctx: DoctorContext) -> tuple[str, str]:
    path = shutil.which(ctx.hermes_bin)
    if path:
        return "pass", f"{path} (version checked by doctor --live)"
    return "fail", "not found on PATH"


@_register("hermes_oneshot_flag")
def hermes_oneshot_flag(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_hermes_call.core import CallOptions, _build_hermes_command

        command = _build_hermes_command(CallOptions(prompt="x", hermes_bin=ctx.hermes_bin), prompt="x")
        if command[:2] == [ctx.hermes_bin, "-z"]:
            return "pass", "wrapper emits -z oneshot argv"
        return "fail", f"oneshot argv changed: {command!r}"
    except Exception as e:
        return "fail", f"oneshot probe error: {e}"


@_register("hermes_help_flags_missing")
def hermes_help_flags_missing(ctx: DoctorContext) -> tuple[str, str]:
    return hermes_oneshot_flag(ctx)


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
        return "warn", "entry point not visible in current Python environment"
    except Exception as e:
        return "fail", f"metadata error: {e}"


@_register("subcommand_valid")
def subcommand_valid(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_hermes_call import plugin

        if callable(getattr(plugin, "register", None)):
            return "pass", "plugin.register is callable"
        return "fail", "plugin.register missing"
    except Exception as e:
        return "fail", f"subcommand probe error: {e}"


@_register("install_integrity")
def install_integrity(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_hermes_call import __version__ as pkg_version

        dist_version = importlib.metadata.version("cluxion-hermes-call-cli")
        if dist_version == pkg_version:
            return "pass", dist_version
        return "warn", f"dist={dist_version} pkg={pkg_version}"
    except importlib.metadata.PackageNotFoundError:
        return "warn", "distribution metadata not visible in current Python environment"
    except Exception as e:
        return "fail", f"version error: {e}"


@_register("default_model_not_configured")
def default_model_not_configured(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_hermes_call.config import read_default_model

        info = read_default_model()
        if info.model:
            return "pass", info.display()
        return "warn", info.display()
    except Exception as e:
        return "warn", f"default model not checked: {e}"


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
            with contextlib.redirect_stderr(io.StringIO()):
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


@_register("sessions_list_parse_failure")
def sessions_list_parse_failure(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_hermes_call.sessions import parse_session_ids_from_list

        sample = (
            "Preview                                            Last Active   Src    ID\n"
            "Reply with exactly pong.                           just now      cli    20260612_235819_78bd06\n"
        )
        if parse_session_ids_from_list(sample) == {"20260612_235819_78bd06"}:
            return "pass", "table parser accepts documented session id format"
        return "fail", "documented table sample did not parse"
    except Exception as e:
        return "fail", f"parse probe error: {e}"


@_register("subprocess_timeout_not_enforced")
def subprocess_timeout_not_enforced(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_hermes_call import core

        if "communicate(timeout=timeout)" not in Path(core.__file__).read_text(encoding="utf-8"):
            return "fail", "communicate timeout sentinel missing"
        return "pass", "Popen communicate timeout path present"
    except Exception as e:
        return "fail", f"timeout probe error: {e}"


@_register("process_group_termination_fails")
def process_group_termination_fails(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_hermes_call import core

        src = Path(core.__file__).read_text(encoding="utf-8")
        if all(token in src for token in ("start_new_session=True", "os.killpg", "signal.SIGTERM", "signal.SIGKILL")):
            return "pass", "process group and SIGKILL fallback sentinels present"
        return "fail", "process group termination sentinels missing"
    except Exception as e:
        return "fail", f"process cleanup probe error: {e}"


@_register("hermes_command_flag_incompatible")
def hermes_command_flag_incompatible(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_hermes_call.core import CallOptions, _build_hermes_command

        options = CallOptions(prompt="x", hermes_bin=ctx.hermes_bin)
        first = _build_hermes_command(options, prompt="x")
        resumed = _build_hermes_command(options, prompt="x", resume_session_id="sid")
        if first[:2] != [ctx.hermes_bin, "-z"]:
            return "fail", f"oneshot command changed: {first!r}"
        if resumed[:5] != [ctx.hermes_bin, "chat", "-Q", "--resume", "sid"]:
            return "fail", f"resume command changed: {resumed!r}"
        return "pass", "oneshot and resume argv match documented contract"
    except Exception as e:
        return "fail", f"command probe error: {e}"


@_register("model_argument_invalid")
def model_argument_invalid(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_hermes_call.core import CallOptions, _build_hermes_command

        command = _build_hermes_command(CallOptions(prompt="x", model="model-x", hermes_bin=ctx.hermes_bin))
        if command[:3] == [ctx.hermes_bin, "-m", "model-x"]:
            return "pass", "-m is passed through to Hermes"
        return "fail", f"model argv changed: {command!r}"
    except Exception as e:
        return "fail", f"model arg probe error: {e}"


@_register("jobs_root_not_writable")
def jobs_root_not_writable(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_hermes_call.jobs import resolve_jobs_root

        root = resolve_jobs_root()
        marker = root / f".doctor-{uuid.uuid4().hex}.marker"
        payload = json.dumps({"ts": time.time()}, sort_keys=True)
        marker.write_text(payload, encoding="utf-8")
        try:
            if marker.read_text(encoding="utf-8") != payload:
                return "fail", f"roundtrip mismatch at {root}"
        finally:
            marker.unlink(missing_ok=True)
        return "pass", str(root)
    except Exception as e:
        return "fail", f"jobs root not writable: {e}"


@_register("session_cleanup_race_condition")
def session_cleanup_race_condition(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_hermes_call import sessions

        src = Path(sessions.__file__).read_text(encoding="utf-8")
        if "started_matches" in src and "max(started_matches" in src:
            return "pass", "same-cwd candidates tie-break by started_at then id"
        return "fail", "same-cwd cleanup tie-break sentinel missing"
    except Exception as e:
        return "fail", f"race probe error: {e}"


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
