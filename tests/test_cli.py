"""Tests for the hermes-call wrapper."""

from __future__ import annotations

import argparse
import io
import json
import os
import sqlite3
import subprocess
import time

import pytest

from cluxion_hermes_call import PostHermes, api, cli, core
from cluxion_hermes_call import plugin as hermes_plugin
from cluxion_hermes_call.core import CallOptions, CallResult, run_call
from cluxion_hermes_call.doctor.framework import CheckResult, DoctorResult
from cluxion_hermes_call.jobs import MARKER_FILE, create_job, delete_job_dir, gc_jobs
from cluxion_hermes_call.sessions import (
    SessionCleanupReport,
    SessionGcReport,
    SessionSnapshot,
    cleanup_created_session,
    default_runner,
    gc_sessions,
    parse_relative_last_active,
    parse_session_ids_from_list,
    parse_session_list_rows,
)


def test_version_flag(capsys):
    assert cli.main(["--version"]) == 0
    assert "hermes-call" in capsys.readouterr().out


def test_usage_error_returns_2(capsys):
    assert cli.main([]) == 2
    assert "PROMPT is required" in capsys.readouterr().err


def test_stdin_prompt_and_json_shape(monkeypatch, capsys):
    seen: dict[str, str] = {}

    def fake_run_call(options: CallOptions) -> CallResult:
        seen["prompt"] = options.prompt
        return CallResult(
            ok=True,
            answer="pong",
            model="grok-4.3",
            duration_ms=12,
            session_cleaned=True,
            exit_code=0,
        )

    monkeypatch.setattr(cli, "run_call", fake_run_call)
    assert cli.main(["-", "--json"], stdin=io.StringIO("hello from stdin")) == 0

    assert seen["prompt"] == "hello from stdin"
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": True,
        "answer": "pong",
        "model": "grok-4.3",
        "duration_ms": 12,
        "session_cleaned": True,
        "exit_code": 0,
    }


@pytest.mark.parametrize(
    ("argv", "error"),
    [
        (["--json", "--prompt", ""], "usage_error"),
        (["--json", "--timeout", "-1", "hi"], "usage_error"),
        (["--json", "--max-iterations", "-1", "hi"], "usage_error"),
    ],
)
def test_json_usage_errors_are_json(argv, error, monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_call", lambda options: pytest.fail("run_call should not start"))

    assert cli.main(argv) == 2

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert payload["error"] == error
    assert payload["message"]
    assert payload["hint"]
    assert captured.err == ""


def test_json_timeout_upper_bound_error_does_not_start(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_call", lambda options: pytest.fail("run_call should not start"))

    assert cli.main(["--json", "--timeout", "999999999", "hi"]) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"] == "usage_error"
    assert "86400" in payload["message"]


def test_json_rejects_null_byte_prompt_before_spawn(capsys):
    assert cli.main(["--json", "--prompt", "bad\0prompt", "--keep-session"]) == 2

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert payload["error"] == "invalid_prompt"
    assert "null byte" in payload["message"]
    assert captured.err == ""


def test_oversize_prompt_is_rejected_before_spawn(monkeypatch):
    monkeypatch.setattr(core.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Popen should not start"))

    result = run_call(CallOptions(prompt="x" * (256 * 1024), keep_session=True))

    assert result.ok is False
    assert result.exit_code == 2
    assert result.status == "prompt_too_large"
    payload = result.to_json_object()
    assert payload["error"] == "prompt_too_large"
    assert "262144" in payload["hint"]


def test_prompt_alias(monkeypatch, capsys):
    seen: dict[str, str] = {}

    def fake_run_call(options: CallOptions) -> CallResult:
        seen["prompt"] = options.prompt
        return CallResult(True, "answer", None, 1, False, 0)

    monkeypatch.setattr(cli, "run_call", fake_run_call)
    assert cli.main(["--prompt", "hello"]) == 0
    assert seen["prompt"] == "hello"
    assert capsys.readouterr().out == "answer\n"


def test_model_and_cwd_pass_through_to_subprocess(monkeypatch, tmp_path):
    calls: list[dict[str, object]] = []

    class FakeProcess:
        pid = 999999
        returncode = 0

        def communicate(self, timeout=None):
            return "ok", ""

    def fake_popen(command, **kwargs):
        calls.append({"command": command, "cwd": kwargs["cwd"]})
        return FakeProcess()

    monkeypatch.setattr(core.subprocess, "Popen", fake_popen)

    result = run_call(
        CallOptions(
            prompt="hello",
            model="grok-4.3",
            cwd=tmp_path,
            keep_session=True,
        )
    )
    assert result.ok is True
    assert calls[-1]["command"] == ["hermes", "-m", "grok-4.3", "-z", "hello"]
    assert calls[-1]["cwd"] == str(tmp_path)

    run_call(CallOptions(prompt="hello", cwd=tmp_path, keep_session=True))
    assert "-m" not in calls[-1]["command"]


def test_help_prints_default_model_line(monkeypatch, capsys):
    monkeypatch.setattr(cli, "default_model_help_line", lambda: "Default model: xai-oauth/grok-4.3")

    assert cli.main(["--help"]) == 0
    assert "Default model: xai-oauth/grok-4.3" in capsys.readouterr().out


def test_doctor_cli_json_shape_and_exit_zero(monkeypatch, capsys):
    result = DoctorResult(plugin="hermes-call", version="0.3.1", checks=())

    def fake_framework_run_doctor(**kw):
        return result

    monkeypatch.setattr(cli, "framework_run_doctor", fake_framework_run_doctor)
    assert cli.main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True


def test_doctor_cli_live_failure_exits_one(monkeypatch, capsys):
    def fake_live(t):
        return [CheckResult(check_id="live_answer", category="live", severity="high", status="fail", detail="fail")]

    monkeypatch.setattr(cli, "live_checks", fake_live)
    # need to make framework also return failing? but for live append fail
    result_ok = DoctorResult(plugin="hermes-call", version="0.3.1", checks=())

    def fake_run(**kw):
        return result_ok

    monkeypatch.setattr(cli, "framework_run_doctor", fake_run)
    assert cli.main(["doctor", "--live", "--json"]) == 1


def test_plugin_doctor_command_wires_to_doctor(monkeypatch, capsys):
    result = DoctorResult(plugin="hermes-call", version="0.3.1", checks=())

    def fake(**kw):
        return result

    monkeypatch.setattr(hermes_plugin, "framework_run_doctor", fake)
    ns = argparse.Namespace(version=False, prompt="doctor", prompt_alias=None, json=False, live=False, timeout=120.0)
    assert hermes_plugin._handle_call_command(ns) == 0
    assert json.loads(capsys.readouterr().err)["ok"] is True


def _completed(command: list[str], returncode: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def _doctor_runner(overrides: dict[tuple[str, ...], subprocess.CompletedProcess[str]] | None = None):
    responses = {
        ("hermes", "--version"): _completed(["hermes", "--version"], stdout="Hermes Agent v0.16.0 (2026.6.5)\n"),
        ("hermes", "--help"): _completed(
            ["hermes", "--help"],
            stdout="-z PROMPT, --oneshot PROMPT\n-t TOOLSETS, --toolsets TOOLSETS\n",
        ),
        ("hermes", "sessions", "--help"): _completed(
            ["hermes", "sessions", "--help"],
            stdout="{list,export,delete}\nlist\nexport\ndelete\n",
        ),
        ("hermes", "sessions", "list", "--source", "cli", "--limit", "20"): _completed(
            ["hermes", "sessions", "list", "--source", "cli", "--limit", "20"],
            stdout="Preview                                            Last Active   Src    ID\n"
            "Reply with exactly pong.                           just now      cli    20260612_235819_78bd06\n",
        ),
    }
    responses.update(overrides or {})

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        return responses.get(tuple(command), _completed(command, 99, stderr=f"unexpected command: {command!r}"))

    return runner


def test_session_list_parser():
    output = """Preview                                            Last Active   Src    ID
───────────────────────────────────────────────────────────────────────────────────────────────
Reply with exactly pong.                           just now      cli    20260612_235819_78bd06
Use a terminal command                             just now      cli    20260612_235819_ad789f
"""
    assert parse_session_ids_from_list(output) == {
        "20260612_235819_78bd06",
        "20260612_235819_ad789f",
    }


def test_parse_session_list_rows_extracts_last_active_by_column():
    output = _gc_list_output(
        [
            {"id": "20260612_235819_78bd06", "last_active": "just now", "preview": "Reply with exactly pong."},
            {"id": "20260612_235819_ad789f", "last_active": "30m ago", "preview": "Use a terminal command"},
        ]
    )
    assert parse_session_list_rows(output) == [
        {"id": "20260612_235819_78bd06", "last_active": "just now", "source": "cli"},
        {"id": "20260612_235819_ad789f", "last_active": "30m ago", "source": "cli"},
    ]


def test_session_cleanup_deletes_exactly_one_new_id():
    calls: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:3] == ["hermes", "sessions", "export"]:
            return subprocess.CompletedProcess(command, 0, '{"id":"b","model":"grok-4.3"}\n', "")
        return subprocess.CompletedProcess(command, 0, "Deleted session 'b'.\n", "")

    report = cleanup_created_session(
        SessionSnapshot(frozenset({"a"})),
        SessionSnapshot(frozenset({"a", "b"})),
        runner=runner,
    )
    assert report.cleaned is True
    assert report.session_id == "b"
    assert report.model == "grok-4.3"
    assert ["hermes", "sessions", "delete", "--yes", "b"] in calls


def test_session_cleanup_refuses_zero_or_many_new_ids():
    deleted: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        deleted.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    zero = cleanup_created_session(SessionSnapshot(frozenset({"a"})), SessionSnapshot(frozenset({"a"})), runner=runner)
    many = cleanup_created_session(
        SessionSnapshot(frozenset({"a"})),
        SessionSnapshot(frozenset({"a", "b", "c"})),
        runner=runner,
    )
    assert zero.cleaned is False
    assert zero.reason == "no_new_session"
    assert many.cleaned is False
    assert many.reason == "multiple_new_sessions:2"
    assert deleted == []


def test_session_cleanup_deletes_unique_exported_cwd_match(tmp_path):
    ours = tmp_path / "ours"
    other = tmp_path / "other"
    calls: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:3] == ["hermes", "sessions", "export"]:
            session_id = command[-1]
            cwd = ours if session_id == "b" else other
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps({"id": session_id, "model": "grok-4.3", "cwd": str(cwd)}) + "\n",
                "",
            )
        if command[:3] == ["hermes", "sessions", "delete"]:
            return subprocess.CompletedProcess(command, 0, "Deleted session 'b'.\n", "")
        return subprocess.CompletedProcess(command, 99, "", "unexpected")

    report = cleanup_created_session(
        SessionSnapshot(frozenset({"a"})),
        SessionSnapshot(frozenset({"a", "b", "c"})),
        runner=runner,
        expected_cwd=ours,
    )

    assert report.cleaned is True
    assert report.session_id == "b"
    assert report.model == "grok-4.3"
    assert ["hermes", "sessions", "delete", "--yes", "b"] in calls


def test_session_cleanup_keeps_ambiguous_exported_cwd_matches(tmp_path):
    calls: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:3] == ["hermes", "sessions", "export"]:
            session_id = command[-1]
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps({"id": session_id, "model": "grok-4.3", "cwd": str(tmp_path)}) + "\n",
                "",
            )
        if command[:3] == ["hermes", "sessions", "delete"]:
            return subprocess.CompletedProcess(command, 0, "Deleted session.\n", "")
        return subprocess.CompletedProcess(command, 99, "", "unexpected")

    report = cleanup_created_session(
        SessionSnapshot(frozenset({"a"})),
        SessionSnapshot(frozenset({"a", "b", "c"})),
        runner=runner,
        expected_cwd=tmp_path,
    )

    assert report.cleaned is False
    assert report.reason == "multiple_new_sessions:2;cwd_match_count:2"
    assert not any(command[:3] == ["hermes", "sessions", "delete"] for command in calls)


def test_session_cleanup_tiebreaks_same_cwd_by_started_at(tmp_path):
    calls: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:3] == ["hermes", "sessions", "export"]:
            session_id = command[-1]
            started_at = 999.0 if session_id == "20260612_120000_bbbb02" else 990.0
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "id": session_id,
                        "model": "grok-4.3",
                        "cwd": str(tmp_path),
                        "started_at": started_at,
                    }
                )
                + "\n",
                "",
            )
        if command[:3] == ["hermes", "sessions", "delete"]:
            return subprocess.CompletedProcess(command, 0, f"Deleted session '{command[-1]}'.\n", "")
        return subprocess.CompletedProcess(command, 99, "", "unexpected")

    report = cleanup_created_session(
        SessionSnapshot(frozenset({"a"})),
        SessionSnapshot(frozenset({"a", "20260612_120000_aaaa01", "20260612_120000_bbbb02"})),
        runner=runner,
        expected_cwd=tmp_path,
    )

    assert report.cleaned is True
    assert report.session_id == "20260612_120000_bbbb02"
    assert ["hermes", "sessions", "delete", "--yes", "20260612_120000_bbbb02"] in calls


def test_session_cleanup_keeps_when_candidate_export_fails(tmp_path):
    calls: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:3] == ["hermes", "sessions", "export"] and command[-1] == "b":
            return subprocess.CompletedProcess(command, 1, "", "export failed")
        if command[:3] == ["hermes", "sessions", "export"]:
            session_id = command[-1]
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps({"id": session_id, "model": "grok-4.3", "cwd": str(tmp_path)}) + "\n",
                "",
            )
        if command[:3] == ["hermes", "sessions", "delete"]:
            return subprocess.CompletedProcess(command, 0, "Deleted session.\n", "")
        return subprocess.CompletedProcess(command, 99, "", "unexpected")

    report = cleanup_created_session(
        SessionSnapshot(frozenset({"a"})),
        SessionSnapshot(frozenset({"a", "b", "c"})),
        runner=runner,
        expected_cwd=tmp_path,
    )

    assert report.cleaned is False
    assert report.reason == "multiple_new_sessions:2;candidate_export_failed:b:export failed"
    assert not any(command[:3] == ["hermes", "sessions", "delete"] for command in calls)


def test_deletion_gate_refuses_path_escape(tmp_path):
    jobs_root = tmp_path / "jobs"
    outside = tmp_path / "outside"
    outside.mkdir()
    decision = delete_job_dir(outside, jobs_root=jobs_root)
    assert decision.allowed is False
    assert decision.reason == "path_escape"
    assert outside.exists()


def test_deletion_gate_refuses_marker_mismatch(tmp_path):
    job = create_job(jobs_root=tmp_path / "jobs")
    (job.root / MARKER_FILE).write_text('{"job_id":"wrong","pid":999999}\n', encoding="utf-8")
    decision = delete_job_dir(job.root, jobs_root=tmp_path / "jobs")
    assert decision.allowed is False
    assert decision.reason == "marker_job_id_mismatch"
    assert job.root.exists()


def test_deletion_gate_refuses_symlinked_job_dir(tmp_path):
    jobs_root = tmp_path / "jobs"
    target = create_job(jobs_root=tmp_path / "real-jobs")
    jobs_root.mkdir()
    link = jobs_root / target.job_id
    link.symlink_to(target.root, target_is_directory=True)

    decision = delete_job_dir(link, jobs_root=jobs_root)
    assert decision.allowed is False
    assert decision.reason == "job_dir_is_symlink"
    assert target.root.exists()


def test_deletion_gate_refuses_live_foreign_pid(tmp_path):
    job = create_job(jobs_root=tmp_path / "jobs")
    (job.root / MARKER_FILE).write_text(json.dumps({"job_id": job.job_id, "pid": 1}) + "\n", encoding="utf-8")
    decision = delete_job_dir(job.root, jobs_root=tmp_path / "jobs")
    assert decision.allowed is False
    assert decision.reason == "pid_alive"
    assert job.root.exists()


def test_timeout_kills_fake_hermes_process(tmp_path):
    fake = tmp_path / "fake-hermes"
    fake.write_text(
        "#!/usr/bin/env python3\nimport time\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)

    result = run_call(
        CallOptions(
            prompt="slow",
            timeout_seconds=0.2,
            keep_session=True,
            hermes_bin=str(fake),
        )
    )
    assert result.ok is False
    assert result.exit_code == 124


def test_timeout_grace_scales_with_requested_timeout(monkeypatch):
    timeouts: list[float | None] = []

    class SlowProcess:
        pid = 12345
        returncode = None

        def communicate(self, timeout=None):
            timeouts.append(timeout)
            if len(timeouts) == 1:
                raise subprocess.TimeoutExpired(["fake"], timeout)
            self.returncode = 124
            return "", ""

    monkeypatch.setattr(core.subprocess, "Popen", lambda *args, **kwargs: SlowProcess())
    monkeypatch.setattr(core.os, "killpg", lambda pid, sig: None)

    result = run_call(CallOptions(prompt="slow", timeout_seconds=0.5, keep_session=True))

    assert result.exit_code == 124
    assert timeouts == [0.5, 0.5]


def test_terminate_process_group_has_sigkill_communicate_timeout(monkeypatch):
    class StubbornProcess:
        pid = 12345
        calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            raise subprocess.TimeoutExpired(["fake"], timeout)

    process = StubbornProcess()
    monkeypatch.setattr(core.os, "killpg", lambda pid, sig: None)

    stdout, stderr = core._terminate_process_group(process)

    assert stdout == ""
    assert process.calls == 2
    assert "did not exit after SIGKILL" in stderr


def test_session_default_runner_times_out_and_returns_completed_process(monkeypatch):
    class SlowProcess:
        pid = 12345
        returncode = None

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(["hermes"], timeout)

    monkeypatch.setenv("CLUXION_HERMES_CALL_SESSION_TIMEOUT", "0.5")
    monkeypatch.setattr("cluxion_hermes_call.sessions.subprocess.run", lambda *a, **k: pytest.fail("use Popen"))
    monkeypatch.setattr("cluxion_hermes_call.sessions.subprocess.Popen", lambda *a, **k: SlowProcess())
    monkeypatch.setattr(core.os, "killpg", lambda pid, sig: None)

    completed = default_runner(["hermes", "sessions", "list"])

    assert completed.returncode == 124
    assert "timed out after 0.5s" in completed.stderr


def test_until_done_loops_until_task_complete(monkeypatch):
    calls: list[dict[str, object]] = []
    outputs = [
        core.HermesProcessResult("first step\nWORK_REMAINS: finish it\n", "", 0, False),
        core.HermesProcessResult("final step\nTASK_COMPLETE\n", "", 0, False),
    ]

    monkeypatch.setattr(core, "capture_session_ids", lambda **kwargs: SessionSnapshot(frozenset({"before"})))
    monkeypatch.setattr(
        core,
        "identify_created_session",
        lambda *args, **kwargs: SessionCleanupReport(False, session_id="owned", model="grok-4.3"),
    )
    monkeypatch.setattr(core, "delete_session", lambda *args, **kwargs: SessionCleanupReport(True, session_id="owned"))

    def fake_run_process(options, *, cwd, prompt, resume_session_id=None, timeout_seconds=None):
        calls.append({"prompt": prompt, "resume_session_id": resume_session_id})
        return outputs.pop(0)

    monkeypatch.setattr(core, "_run_hermes_process_with_prompt", fake_run_process)

    result = run_call(CallOptions(prompt="do it", until_done=True, max_iterations=4))

    assert result.ok is True
    assert result.status == "complete"
    assert result.iterations == 2
    assert result.session_cleaned is True
    assert result.answer == "first step\n\nfinal step"
    assert calls[0]["resume_session_id"] is None
    assert calls[1]["resume_session_id"] == "owned"
    assert "Completion contract for hermes-call --until-done" in calls[0]["prompt"]


def test_strip_completion_marker_removes_marker_lines_before_later_text():
    assert core._strip_completion_marker("answer\nTASK_COMPLETE\nafter\n") == "answer\nafter"
    assert core._strip_completion_marker("answer\nWORK_REMAINS: finish later\nafter\n") == "answer\nafter"


def test_until_done_stops_incomplete_at_max_iterations(monkeypatch):
    monkeypatch.setattr(core, "capture_session_ids", lambda **kwargs: SessionSnapshot(frozenset({"before"})))
    monkeypatch.setattr(
        core,
        "identify_created_session",
        lambda *args, **kwargs: SessionCleanupReport(False, session_id="owned", model="grok-4.3"),
    )
    monkeypatch.setattr(core, "delete_session", lambda *args, **kwargs: SessionCleanupReport(True, session_id="owned"))

    def fake_run_process(options, *, cwd, prompt, resume_session_id=None, timeout_seconds=None):
        return core.HermesProcessResult("partial\nWORK_REMAINS: still open\n", "", 0, False)

    monkeypatch.setattr(core, "_run_hermes_process_with_prompt", fake_run_process)

    result = run_call(CallOptions(prompt="do it", until_done=True, max_iterations=2))

    assert result.ok is False
    assert result.status == "incomplete"
    assert result.iterations == 2
    assert result.last_work_remains == "still open"
    assert result.exit_code == 1
    assert "max iterations reached" in result.answer


def test_until_done_stops_when_work_remains_makes_no_progress(monkeypatch):
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(core, "capture_session_ids", lambda **kwargs: SessionSnapshot(frozenset({"before"})))
    monkeypatch.setattr(
        core,
        "identify_created_session",
        lambda *args, **kwargs: SessionCleanupReport(False, session_id="owned", model="grok-4.3"),
    )
    monkeypatch.setattr(core, "delete_session", lambda *args, **kwargs: SessionCleanupReport(True, session_id="owned"))

    def fake_run_process(options, *, cwd, prompt, resume_session_id=None, timeout_seconds=None):
        calls.append({"prompt": prompt, "resume_session_id": resume_session_id})
        return core.HermesProcessResult("partial\nWORK_REMAINS: same blocker\n", "", 0, False)

    monkeypatch.setattr(core, "_run_hermes_process_with_prompt", fake_run_process)

    result = run_call(CallOptions(prompt="do it", until_done=True, max_iterations=8))

    assert len(calls) == 2
    assert result.ok is False
    assert result.status == "incomplete"
    assert result.iterations == 2
    assert result.last_work_remains == "same blocker"
    assert "no progress observed" in result.answer


def test_until_done_no_session_id_does_not_loop(monkeypatch):
    calls = 0

    monkeypatch.setattr(core, "capture_session_ids", lambda **kwargs: SessionSnapshot(frozenset({"before"})))
    monkeypatch.setattr(
        core,
        "identify_created_session",
        lambda *args, **kwargs: SessionCleanupReport(False, reason="multiple_new_sessions:2"),
    )

    def fake_run_process(options, *, cwd, prompt, resume_session_id=None, timeout_seconds=None):
        nonlocal calls
        calls += 1
        return core.HermesProcessResult("partial\nWORK_REMAINS: need resume\n", "", 0, False)

    monkeypatch.setattr(core, "_run_hermes_process_with_prompt", fake_run_process)

    result = run_call(CallOptions(prompt="do it", until_done=True, max_iterations=3))

    assert calls == 1
    assert result.ok is False
    assert result.status == "incomplete"
    assert result.session_cleanup_reason == "multiple_new_sessions:2"
    assert "could not determine the Hermes session id" in result.answer


def test_posthermes_simple_returns_string_and_structured_returns_object(monkeypatch):
    seen: list[CallOptions] = []

    def fake_run_call(options: CallOptions) -> CallResult:
        seen.append(options)
        return CallResult(
            ok=True,
            answer="api-answer",
            model=options.model,
            duration_ms=1,
            session_cleaned=True,
            exit_code=0,
            status="complete" if options.until_done else None,
            iterations=1 if options.until_done else None,
        )

    monkeypatch.setattr(api, "run_call", fake_run_call)

    assert PostHermes(model="grok-4.3", path="/tmp", prompt="hi") == "api-answer"
    payload = json.loads(PostHermes(model="grok-4.3", path="/tmp", prompt="hi", json=True))
    assert payload["answer"] == "api-answer"
    structured = PostHermes.run(model="grok-4.3", path="/tmp", prompt="hi", until_done=True)

    assert structured.answer == "api-answer"
    assert structured.status == "complete"
    assert seen[0].model == "grok-4.3"
    assert str(seen[0].cwd) == "/tmp"


def test_gc_removes_old_unlocked_and_keeps_fresh_or_locked(tmp_path):
    jobs_root = tmp_path / "jobs"
    old = create_job(jobs_root=jobs_root)
    fresh = create_job(jobs_root=jobs_root)
    locked = create_job(jobs_root=jobs_root)
    (locked.root / MARKER_FILE).write_text(json.dumps({"job_id": locked.job_id, "pid": 1}) + "\n", encoding="utf-8")

    old_time = time.time() - 25 * 60 * 60
    os.utime(old.root / MARKER_FILE, (old_time, old_time))
    os.utime(locked.root / MARKER_FILE, (old_time, old_time))

    removed, kept = gc_jobs(jobs_root=jobs_root)
    assert removed == 1
    assert kept == 2
    assert not old.root.exists()
    assert fresh.root.exists()
    assert locked.root.exists()


live = pytest.mark.skipif(os.getenv("CLUXION_HERMES_CALL_LIVE") != "1", reason="live Hermes smoke disabled")


@live
def test_live_ask_smoke(capsys):
    code = cli.main(["--ask", "Reply with exactly pong.", "--timeout", "120"])
    assert code == 0
    assert "pong" in capsys.readouterr().out.lower()


@live
def test_live_sandbox_smoke(capsys):
    code = cli.main(["--sandbox", "--ask", "Reply with exactly sandbox-ok.", "--timeout", "120", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert "sandbox-ok" in payload["answer"].lower()
    assert set(payload) == {"ok", "answer", "model", "duration_ms", "session_cleaned", "exit_code"}


@live
def test_live_until_done_smoke(capsys):
    code = cli.main(
        [
            "-m",
            "grok-4.3",
            "--until-done",
            "--max-iterations",
            "3",
            "--timeout",
            "180",
            "Reply with exactly LIVE_UNTIL_DONE_OK on one line and TASK_COMPLETE on the final line.",
        ]
    )
    assert code == 0
    assert "LIVE_UNTIL_DONE_OK" in capsys.readouterr().out


def test_doctor_no_usage_on_stderr(monkeypatch, capsys):
    result = DoctorResult(plugin="hermes-call", version="0.3.3", checks=())

    def fake_framework_run_doctor(**kw):
        return result

    monkeypatch.setattr(cli, "framework_run_doctor", fake_framework_run_doctor)
    assert cli.main(["doctor"]) == 0
    err = capsys.readouterr().err
    assert "usage:" not in err.lower()


def _gc_list_output(rows: list[dict[str, str]]) -> str:
    lines = [
        "Preview                                            Last Active   Src    ID",
        "───────────────────────────────────────────────────────────────────────────────────────────────",
    ]
    for row in rows:
        preview = row.get("preview", "untitled preview")
        lines.append(f"{preview:<50} {row['last_active']:<13} cli    {row['id']}")
    return "\n".join(lines) + "\n"


def _gc_export_record(
    session_id: str,
    *,
    title: str | None = None,
    ended_at: float | None = None,
    last_message_at: float | None = None,
    started_at: float | None = None,
) -> str:
    payload = {
        "id": session_id,
        "source": "cli",
        "title": title,
        "ended_at": ended_at,
        "started_at": started_at,
        "messages": [],
    }
    if last_message_at is not None:
        payload["messages"] = [{"timestamp": last_message_at}]
    return json.dumps(payload) + "\n"


def test_gc_sessions_deletes_untitled_stale_when_apply(monkeypatch):
    now = 1_000_000.0
    deleted: list[str] = []
    monkeypatch.setattr("cluxion_hermes_call.sessions._load_rich_cli_sessions", lambda **kwargs: {})

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:4] == ["hermes", "sessions", "list", "--source"]:
            return _completed(
                command,
                stdout=_gc_list_output([{"id": "20260612_120000_aaaa01", "last_active": "30m ago"}]),
            )
        if command[:3] == ["hermes", "sessions", "export"]:
            return _completed(command, stdout=_gc_export_record(command[-1], last_message_at=now - 1800))
        if command[:3] == ["hermes", "sessions", "delete"]:
            deleted.append(command[-1])
            return _completed(command, stdout=f"Deleted session '{command[-1]}'.\n")
        if command[:3] == ["hermes", "sessions", "optimize"]:
            return _completed(command, stdout="optimized\n")
        raise AssertionError(f"unexpected command: {command!r}")

    report = gc_sessions(dry_run=False, idle_minutes=10, runner=runner, now=now)
    assert report.deleted == 1
    assert report.deleted_ids == ["20260612_120000_aaaa01"]
    assert deleted == ["20260612_120000_aaaa01"]
    assert report.optimized is True


def test_gc_sessions_reports_untitled_stale_in_dry_run(monkeypatch):
    now = 1_000_000.0
    deleted: list[str] = []
    monkeypatch.setattr("cluxion_hermes_call.sessions._load_rich_cli_sessions", lambda **kwargs: {})

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:4] == ["hermes", "sessions", "list", "--source"]:
            return _completed(
                command,
                stdout=_gc_list_output([{"id": "20260612_120000_aaaa01", "last_active": "30m ago"}]),
            )
        if command[:3] == ["hermes", "sessions", "export"]:
            return _completed(command, stdout=_gc_export_record(command[-1], last_message_at=now - 1800))
        if command[:3] == ["hermes", "sessions", "delete"]:
            deleted.append(command[-1])
            return _completed(command, stdout=f"Deleted session '{command[-1]}'.\n")
        raise AssertionError(f"unexpected command: {command!r}")

    report = gc_sessions(dry_run=True, idle_minutes=10, runner=runner, now=now)
    assert report.dry_run is True
    assert report.deleted == 1
    assert report.deleted_ids == ["20260612_120000_aaaa01"]
    assert deleted == []


def test_gc_sessions_keeps_named_sessions(monkeypatch):
    now = 1_000_000.0
    monkeypatch.setattr("cluxion_hermes_call.sessions._load_rich_cli_sessions", lambda **kwargs: {})

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:4] == ["hermes", "sessions", "list", "--source"]:
            return _completed(
                command,
                stdout=_gc_list_output([{"id": "20260612_120000_bbbb01", "last_active": "30m ago"}]),
            )
        if command[:3] == ["hermes", "sessions", "export"]:
            return _completed(
                command,
                stdout=_gc_export_record(command[-1], title="Important session", last_message_at=now - 1800),
            )
        raise AssertionError(f"unexpected command: {command!r}")

    report = gc_sessions(dry_run=False, idle_minutes=10, runner=runner, now=now)
    assert report.deleted == 0
    assert report.kept_named == 1
    assert report.kept_named_ids == ["20260612_120000_bbbb01"]


def test_gc_sessions_keeps_recent_untitled_sessions(monkeypatch):
    now = 1_000_000.0
    monkeypatch.setattr("cluxion_hermes_call.sessions._load_rich_cli_sessions", lambda **kwargs: {})

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:4] == ["hermes", "sessions", "list", "--source"]:
            return _completed(
                command,
                stdout=_gc_list_output([{"id": "20260612_120000_cccc01", "last_active": "just now"}]),
            )
        if command[:3] == ["hermes", "sessions", "export"]:
            return _completed(command, stdout=_gc_export_record(command[-1], last_message_at=now - 30))
        raise AssertionError(f"unexpected command: {command!r}")

    report = gc_sessions(dry_run=False, idle_minutes=10, runner=runner, now=now)
    assert report.deleted == 0
    assert report.skipped_recent == 1
    assert report.skipped_recent_ids == ["20260612_120000_cccc01"]


def test_gc_sessions_keeps_unknown_timestamp_fail_closed(monkeypatch):
    now = 1_000_000.0
    monkeypatch.setattr("cluxion_hermes_call.sessions._load_rich_cli_sessions", lambda **kwargs: {})

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:4] == ["hermes", "sessions", "list", "--source"]:
            return _completed(
                command,
                stdout=_gc_list_output([{"id": "20260612_120000_dddd01", "last_active": "?"}]),
            )
        if command[:3] == ["hermes", "sessions", "export"]:
            return _completed(command, stdout=_gc_export_record(command[-1], title=None))
        raise AssertionError(f"unexpected command: {command!r}")

    report = gc_sessions(dry_run=False, idle_minutes=10, runner=runner, now=now)
    assert report.deleted == 0
    assert report.skipped_unknown == 1
    assert report.skipped_unknown_ids == ["20260612_120000_dddd01"]


def test_gc_sessions_uses_list_row_last_active_when_export_lacks_timestamps(monkeypatch):
    # Use wall-clock `now` so list-row relative timestamps align with parse_relative_last_active.
    now = time.time()
    deleted: list[str] = []
    monkeypatch.setattr("cluxion_hermes_call.sessions._load_rich_cli_sessions", lambda **kwargs: {})

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:4] == ["hermes", "sessions", "list", "--source"]:
            return _completed(
                command,
                stdout=_gc_list_output([{"id": "20260612_120000_eeee01", "last_active": "30m ago"}]),
            )
        if command[:3] == ["hermes", "sessions", "export"]:
            return _completed(command, stdout=_gc_export_record(command[-1], title=None))
        if command[:3] == ["hermes", "sessions", "delete"]:
            deleted.append(command[-1])
            return _completed(command, stdout=f"Deleted session '{command[-1]}'.\n")
        if command[:3] == ["hermes", "sessions", "optimize"]:
            return _completed(command, stdout="optimized\n")
        raise AssertionError(f"unexpected command: {command!r}")

    report = gc_sessions(dry_run=False, idle_minutes=10, runner=runner, now=now)
    assert report.deleted == 1
    assert report.deleted_ids == ["20260612_120000_eeee01"]
    assert deleted == ["20260612_120000_eeee01"]


def test_gc_sessions_skips_list_when_rich_present(monkeypatch):
    now = 1_000_000.0
    deleted: list[str] = []
    session_id = "20260612_120000_aaaa01"
    rich = {
        session_id: {
            "id": session_id,
            "source": "cli",
            "title": None,
            "ended_at": None,
            "last_active": now - 1800,
        }
    }
    monkeypatch.setattr("cluxion_hermes_call.sessions._load_rich_cli_sessions", lambda **kwargs: rich)

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:4] == ["hermes", "sessions", "list", "--source"]:
            raise AssertionError("list subprocess must not run when rich DB is present")
        if command[:3] == ["hermes", "sessions", "delete"]:
            deleted.append(command[-1])
            return _completed(command, stdout=f"Deleted session '{command[-1]}'.\n")
        if command[:3] == ["hermes", "sessions", "optimize"]:
            return _completed(command, stdout="optimized\n")
        raise AssertionError(f"unexpected command: {command!r}")

    report = gc_sessions(dry_run=False, idle_minutes=10, runner=runner, now=now)
    assert report.deleted == 1
    assert report.deleted_ids == [session_id]
    assert deleted == [session_id]


def test_gc_sessions_reads_sqlite_db_without_list_or_exports(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    con = sqlite3.connect(db_path)
    con.execute(
        "create table sessions ("
        "id text primary key, source text, title text, ended_at real, started_at real, cwd text)"
    )
    con.execute("create table messages (session_id text, timestamp real)")
    con.execute(
        "insert into sessions (id, source, title, ended_at, started_at, cwd) values (?, ?, ?, ?, ?, ?)",
        ("20260612_120000_aaaa01", "cli", None, None, 1_000_000.0 - 1800, str(tmp_path)),
    )
    con.execute(
        "insert into messages (session_id, timestamp) values (?, ?)",
        ("20260612_120000_aaaa01", 1_000_000.0 - 1800),
    )
    con.commit()
    con.close()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        raise AssertionError(f"unexpected subprocess: {command!r}")

    report = gc_sessions(dry_run=True, idle_minutes=10, runner=runner, now=1_000_000.0)

    assert report.deleted == 1
    assert report.deleted_ids == ["20260612_120000_aaaa01"]
    assert calls == []


def test_gc_sessions_rich_and_list_paths_same_decisions(monkeypatch):
    now = 1_000_000.0
    session_id = "20260612_120000_aaaa01"

    rich = {
        session_id: {
            "id": session_id,
            "source": "cli",
            "title": None,
            "ended_at": None,
            "last_active": now - 1800,
        }
    }
    monkeypatch.setattr("cluxion_hermes_call.sessions._load_rich_cli_sessions", lambda **kwargs: rich)

    def rich_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:4] == ["hermes", "sessions", "list", "--source"]:
            raise AssertionError("list subprocess must not run when rich DB is present")
        raise AssertionError(f"unexpected command: {command!r}")

    rich_report = gc_sessions(dry_run=True, idle_minutes=10, runner=rich_runner, now=now)

    monkeypatch.setattr("cluxion_hermes_call.sessions._load_rich_cli_sessions", lambda **kwargs: {})

    def list_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:4] == ["hermes", "sessions", "list", "--source"]:
            return _completed(
                command,
                stdout=_gc_list_output([{"id": session_id, "last_active": "30m ago"}]),
            )
        if command[:3] == ["hermes", "sessions", "export"]:
            return _completed(command, stdout=_gc_export_record(command[-1], last_message_at=now - 1800))
        raise AssertionError(f"unexpected command: {command!r}")

    list_report = gc_sessions(dry_run=True, idle_minutes=10, runner=list_runner, now=now)
    assert rich_report.deleted == list_report.deleted == 1
    assert rich_report.deleted_ids == list_report.deleted_ids == [session_id]
    assert rich_report.skipped_recent == list_report.skipped_recent == 0
    assert rich_report.kept_named == list_report.kept_named == 0
    assert rich_report.skipped_unknown == list_report.skipped_unknown == 0


def test_gc_sessions_uses_list_fallback_when_rich_absent(monkeypatch):
    now = 1_000_000.0
    list_calls: list[list[str]] = []
    monkeypatch.setattr("cluxion_hermes_call.sessions._load_rich_cli_sessions", lambda **kwargs: {})

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:4] == ["hermes", "sessions", "list", "--source"]:
            list_calls.append(command)
            return _completed(
                command,
                stdout=_gc_list_output([{"id": "20260612_120000_aaaa01", "last_active": "30m ago"}]),
            )
        if command[:3] == ["hermes", "sessions", "export"]:
            return _completed(command, stdout=_gc_export_record(command[-1], last_message_at=now - 1800))
        raise AssertionError(f"unexpected command: {command!r}")

    report = gc_sessions(dry_run=True, idle_minutes=10, runner=runner, now=now)
    assert len(list_calls) == 1
    assert list_calls[0][:4] == ["hermes", "sessions", "list", "--source"]
    assert report.deleted == 1
    assert report.deleted_ids == ["20260612_120000_aaaa01"]


def test_gc_sessions_cli_dry_run_by_default(monkeypatch, capsys):
    now = 1_000_000.0
    deleted: list[str] = []
    monkeypatch.setattr("cluxion_hermes_call.sessions._load_rich_cli_sessions", lambda **kwargs: {})

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:4] == ["hermes", "sessions", "list", "--source"]:
            return _completed(
                command,
                stdout=_gc_list_output([{"id": "20260612_120000_aaaa01", "last_active": "30m ago"}]),
            )
        if command[:3] == ["hermes", "sessions", "export"]:
            return _completed(command, stdout=_gc_export_record(command[-1], last_message_at=now - 1800))
        if command[:3] == ["hermes", "sessions", "delete"]:
            deleted.append(command[-1])
            return _completed(command, stdout=f"Deleted session '{command[-1]}'.\n")
        raise AssertionError(f"unexpected command: {command!r}")

    def fake_gc_sessions(**kwargs):
        return gc_sessions(runner=runner, now=now, **kwargs)

    monkeypatch.setattr(cli, "gc_sessions", fake_gc_sessions)
    assert cli.main(["gc", "--sessions"]) == 0
    out = capsys.readouterr().out
    assert "dry_run=True" in out
    assert "deleted=1" in out
    assert deleted == []


def test_gc_sessions_cli_list_failure_exits_2_for_dry_run_and_apply(monkeypatch, capsys):
    def fake_gc_sessions(**kwargs):
        report = SessionGcReport(dry_run=kwargs["dry_run"])
        report.error = "list_failed:boom"
        return report

    monkeypatch.setattr(cli, "gc_sessions", fake_gc_sessions)

    assert cli.main(["gc", "--sessions"]) == 2
    captured = capsys.readouterr()
    assert (
        captured.out
        == "sessions dry_run=True deleted=0 kept_named=0 skipped_recent=0 skipped_unknown=0 failed=0 optimized=False\n"
    )
    assert "list_failed:boom" in captured.err

    assert cli.main(["gc", "--sessions", "--apply"]) == 2
    captured = capsys.readouterr()
    assert (
        captured.out
        == "sessions dry_run=False deleted=0 kept_named=0 skipped_recent=0 skipped_unknown=0 failed=0 optimized=False\n"
    )
    assert "list_failed:boom" in captured.err


def test_parse_relative_last_active():
    now = 1_700_000_000.0
    assert parse_relative_last_active("just now", now=now) == now
    assert parse_relative_last_active("5m ago", now=now) == now - 300
    assert parse_relative_last_active("?", now=now) is None


def test_capture_uses_bounded_limit():
    """Assert session-capture uses a bounded --limit (newest-N) not full unbounded list."""
    import subprocess

    from cluxion_hermes_call.sessions import capture_session_ids

    captured_cmds: list[list[str]] = []

    def fake_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    # call with default
    snap = capture_session_ids(runner=fake_runner)
    assert snap.ok
    # find the list command
    list_cmds = [c for c in captured_cmds if "sessions" in c and "list" in c]
    assert list_cmds, "should have called sessions list"
    for cmd in list_cmds:
        assert "--limit" in cmd
        idx = cmd.index("--limit")
        lim = int(cmd[idx + 1])
        assert lim <= 50, f"limit should be bounded to <=50, got {lim}"
        assert lim == 50  # our chosen default

    captured_cmds.clear()
    # also explicit small limit works
    capture_session_ids(runner=fake_runner, limit=20)
    list_cmds2 = [c for c in captured_cmds if "sessions" in c and "list" in c]
    assert any("--limit" in c and int(c[c.index("--limit") + 1]) == 20 for c in list_cmds2)
