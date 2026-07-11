from __future__ import annotations

from pathlib import Path

from cluxion_hermes_call.core import CallOptions, _build_hermes_command
from cluxion_hermes_call.sessions import SessionMetadata


def test_resume_command_uses_chat_resume_with_model() -> None:
    options = CallOptions(prompt="continue", model="grok-4.3", resume_session=None)
    command = _build_hermes_command(options, resume_session_id="20260702_abc")
    assert command[:5] == ["hermes", "chat", "-Q", "--resume", "20260702_abc"]
    assert "-m" in command and "grok-4.3" in command
    assert command[-1] == "continue"


def test_resume_session_skips_gc(monkeypatch) -> None:
    from cluxion_hermes_call import core

    calls = {"snapshot": 0}

    def _snap(**kwargs: object) -> object:
        calls["snapshot"] += 1
        return core.SessionSnapshot(ids=frozenset(), ok=True)

    monkeypatch.setattr(core, "capture_session_ids", _snap)
    monkeypatch.setattr(
        core,
        "fetch_session_metadata",
        lambda *_args, **_kwargs: SessionMetadata("20260702_abc", cwd=str(Path.cwd())),
    )
    monkeypatch.setattr(
        core,
        "_run_hermes_process",
        lambda options, *, cwd, resume_session_id=None: core.HermesProcessResult(
            stdout="done", stderr="", returncode=0, timed_out=False
        ),
    )
    result = core.run_call(CallOptions(prompt="hi", resume_session="20260702_abc"))
    assert calls["snapshot"] == 0
    assert result.session_cleanup_reason == "resumed_session"


def test_resume_matching_exported_cwd_spawns(monkeypatch, tmp_path) -> None:
    from cluxion_hermes_call import core

    calls = []
    monkeypatch.setattr(
        core,
        "fetch_session_metadata",
        lambda *_args, **_kwargs: SessionMetadata("sid", cwd=str(tmp_path)),
        raising=False,
    )
    monkeypatch.setattr(
        core,
        "_run_hermes_process",
        lambda options, *, cwd, resume_session_id=None: (
            calls.append((cwd, resume_session_id))
            or core.HermesProcessResult("done", "", 0, False)
        ),
    )

    result = core.run_call(CallOptions(prompt="hi", cwd=tmp_path, resume_session="sid"))

    assert result.ok is True
    assert calls == [(tmp_path.resolve(), "sid")]


def test_resume_foreign_or_unknown_cwd_never_spawns(monkeypatch, tmp_path) -> None:
    from cluxion_hermes_call import core

    spawned = []
    monkeypatch.setattr(
        core,
        "_run_hermes_process",
        lambda *_args, **_kwargs: spawned.append(True),
    )
    foreign = tmp_path / "foreign"
    foreign.mkdir()

    for metadata, error in (
        (SessionMetadata("sid", cwd=str(foreign)), "resume_owner_mismatch"),
        (SessionMetadata("sid", error="export failed"), "resume_owner_unknown"),
        (SessionMetadata("sid", cwd=None), "resume_owner_unknown"),
        (SessionMetadata("sid", cwd=str(tmp_path / "deleted")), "resume_owner_unknown"),
    ):
        monkeypatch.setattr(core, "fetch_session_metadata", lambda *_a, _m=metadata, **_k: _m, raising=False)
        result = core.run_call(CallOptions(prompt="hi", cwd=tmp_path, resume_session="sid"))
        assert result.ok is False
        assert result.error == error
        assert result.exit_code == 2
    assert spawned == []


def test_resume_symlink_equivalent_cwd_is_allowed(monkeypatch, tmp_path) -> None:
    from cluxion_hermes_call import core

    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    calls = []
    monkeypatch.setattr(
        core,
        "fetch_session_metadata",
        lambda *_args, **_kwargs: SessionMetadata("sid", cwd=str(alias)),
        raising=False,
    )
    monkeypatch.setattr(
        core,
        "_run_hermes_process",
        lambda options, *, cwd, resume_session_id=None: (
            calls.append(cwd) or core.HermesProcessResult("done", "", 0, False)
        ),
    )

    assert core.run_call(CallOptions(prompt="hi", cwd=real, resume_session="sid")).ok is True
    assert calls == [real.resolve()]


def test_resume_with_sandbox_is_rejected_before_job_or_spawn(monkeypatch) -> None:
    from cluxion_hermes_call import core

    monkeypatch.setattr(core, "create_job", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("job created")))
    monkeypatch.setattr(
        core,
        "_run_hermes_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("spawned")),
    )

    result = core.run_call(CallOptions(prompt="hi", sandbox=True, resume_session="sid"))

    assert result.error == "resume_sandbox_conflict"
    assert result.exit_code == 2


def test_resume_with_until_done_is_rejected_by_core_before_side_effects(monkeypatch) -> None:
    from cluxion_hermes_call import core

    monkeypatch.setattr(
        core,
        "_run_until_done_call",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("until-done started")),
    )
    monkeypatch.setattr(
        core,
        "fetch_session_metadata",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("resume metadata read")),
    )

    result = core.run_call(CallOptions(prompt="hi", until_done=True, resume_session="sid"))

    assert result.error == "resume_until_done_conflict"
    assert result.exit_code == 2


def test_resume_session_rejects_internal_whitespace() -> None:
    from cluxion_hermes_call import core

    result = core.validate_call_options(CallOptions(prompt="hi", resume_session="sid other"))

    assert result is not None
    assert result.error == "invalid_resume_session"
    assert result.exit_code == 2
