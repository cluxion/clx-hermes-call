"""Tests for the hermes-call wrapper."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import time

import pytest

from cluxion_hermes_call import PostHermes, api, cli, core
from cluxion_hermes_call import plugin as hermes_plugin
from cluxion_hermes_call.core import CallOptions, CallResult, run_call
from cluxion_hermes_call.doctor import DoctorCheck, DoctorResult, run_doctor, write_doctor_result
from cluxion_hermes_call.jobs import MARKER_FILE, create_job, delete_job_dir, gc_jobs
from cluxion_hermes_call.sessions import (
    SessionCleanupReport,
    SessionSnapshot,
    cleanup_created_session,
    parse_session_ids_from_list,
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
    result = DoctorResult(checks=(DoctorCheck("hermes_version", True, "0.16.0 at /bin/hermes"),))
    seen: dict[str, object] = {}

    def fake_run_doctor(*, live: bool = False, timeout_seconds: float = 120.0) -> DoctorResult:
        seen["live"] = live
        seen["timeout_seconds"] = timeout_seconds
        return result

    monkeypatch.setattr(cli, "run_doctor", fake_run_doctor)

    assert cli.main(["doctor"]) == 0
    captured = capsys.readouterr()
    assert "PASS hermes_version: 0.16.0 at /bin/hermes" in captured.err
    assert json.loads(captured.out) == {
        "ok": True,
        "checks": [{"name": "hermes_version", "ok": True, "detail": "0.16.0 at /bin/hermes"}],
    }
    assert seen == {"live": False, "timeout_seconds": 120.0}


def test_doctor_cli_live_failure_exits_one(monkeypatch, capsys):
    result = DoctorResult(checks=(DoctorCheck("live_no_tools", False, "answer='files'"),))

    def fake_run_doctor(*, live: bool = False, timeout_seconds: float = 120.0) -> DoctorResult:
        assert live is True
        assert timeout_seconds == 3.0
        return result

    monkeypatch.setattr(cli, "run_doctor", fake_run_doctor)

    assert cli.main(["doctor", "--live", "--timeout", "3"]) == 1
    captured = capsys.readouterr()
    assert "FAIL live_no_tools" in captured.err
    assert json.loads(captured.out)["ok"] is False


def test_plugin_doctor_command_wires_to_doctor(monkeypatch, capsys):
    result = DoctorResult(checks=(DoctorCheck("hermes_version", True, "ok"),))
    seen: dict[str, object] = {}

    def fake_run_doctor(*, live: bool = False, timeout_seconds: float = 120.0) -> DoctorResult:
        seen["live"] = live
        seen["timeout_seconds"] = timeout_seconds
        return result

    monkeypatch.setattr(hermes_plugin, "run_doctor", fake_run_doctor)

    ns = argparse.Namespace(version=False, prompt="doctor", prompt_alias=None, live=True, timeout=9.0)
    assert hermes_plugin._handle_call_command(ns) == 0

    captured = capsys.readouterr()
    assert "PASS hermes_version: ok" in captured.err
    assert json.loads(captured.out)["ok"] is True
    assert seen == {"live": True, "timeout_seconds": 9.0}


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


def test_doctor_static_checks_pass_with_mocked_subprocesses(tmp_path):
    result = run_doctor(runner=_doctor_runner(), which=lambda _: "/bin/hermes", jobs_root=tmp_path / "jobs")

    assert result.ok is True
    assert {check.name for check in result.checks} == {
        "hermes_version",
        "hermes_help_flags",
        "hermes_ask_toolset",
        "hermes_sessions_help",
        "hermes_sessions_list_parse",
        "jobs_root_writable",
    }


@pytest.mark.parametrize(
    ("overrides", "which_result", "expected_failed_check"),
    [
        ({}, None, "hermes_version"),
        (
            {("hermes", "--version"): _completed(["hermes", "--version"], stdout="Hermes Agent unknown\n")},
            "/bin/hermes",
            "hermes_version",
        ),
        (
            {("hermes", "--help"): _completed(["hermes", "--help"], stdout="-z PROMPT, --oneshot PROMPT\n")},
            "/bin/hermes",
            "hermes_help_flags",
        ),
        (
            {("hermes", "sessions", "--help"): _completed(["hermes", "sessions", "--help"], stdout="list\nexport\n")},
            "/bin/hermes",
            "hermes_sessions_help",
        ),
        (
            {
                ("hermes", "sessions", "list", "--source", "cli", "--limit", "20"): _completed(
                    ["hermes", "sessions", "list", "--source", "cli", "--limit", "20"],
                    stdout="Title Preview Last Active ID\nnot a parseable table\n",
                )
            },
            "/bin/hermes",
            "hermes_sessions_list_parse",
        ),
    ],
)
def test_doctor_static_failure_modes(tmp_path, overrides, which_result, expected_failed_check):
    result = run_doctor(
        runner=_doctor_runner(overrides),
        which=lambda _: which_result,
        jobs_root=tmp_path / "jobs",
    )
    checks = {check.name: check for check in result.checks}

    assert result.ok is False
    assert checks[expected_failed_check].ok is False


def test_doctor_jobs_root_failure(tmp_path):
    jobs_root = tmp_path / "jobs"
    jobs_root.write_text("not a directory\n", encoding="utf-8")

    result = run_doctor(runner=_doctor_runner(), which=lambda _: "/bin/hermes", jobs_root=jobs_root)
    checks = {check.name: check for check in result.checks}

    assert result.ok is False
    assert checks["jobs_root_writable"].ok is False


def test_doctor_live_checks_one_no_tools_call(tmp_path):
    seen: list[CallOptions] = []

    def fake_call_runner(options: CallOptions) -> CallResult:
        seen.append(options)
        return CallResult(
            ok=True,
            answer="NO_TOOLS\n",
            model="grok-4.3",
            duration_ms=10,
            session_cleaned=True,
            exit_code=0,
            session_id="20260612_235819_78bd06",
            job_deleted=True,
        )

    result = run_doctor(
        live=True,
        timeout_seconds=5.0,
        runner=_doctor_runner(),
        which=lambda _: "/bin/hermes",
        jobs_root=tmp_path / "jobs",
        call_runner=fake_call_runner,
    )
    live_checks = {check.name: check for check in result.checks if check.name.startswith("live_")}

    assert result.ok is True
    assert len(seen) == 1
    assert seen[0].ask is True
    assert seen[0].sandbox is True
    assert seen[0].timeout_seconds == 5.0
    assert live_checks["live_answer"].ok is True
    assert live_checks["live_no_tools"].ok is True
    assert live_checks["live_session_cleanup"].ok is True


def test_write_doctor_result(capsys):
    result = DoctorResult(
        checks=(
            DoctorCheck("ok_check", True, "fine"),
            DoctorCheck("bad_check", False, "broken"),
        )
    )
    write_doctor_result(result)
    captured = capsys.readouterr()

    assert "PASS ok_check: fine" in captured.err
    assert "FAIL bad_check: broken" in captured.err
    assert json.loads(captured.out) == {
        "ok": False,
        "checks": [
            {"name": "ok_check", "ok": True, "detail": "fine"},
            {"name": "bad_check", "ok": False, "detail": "broken"},
        ],
    }


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
