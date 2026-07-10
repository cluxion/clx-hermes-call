from __future__ import annotations

import pytest

from cluxion_hermes_call import core
from cluxion_hermes_call.core import CallOptions, HermesProcessResult, sanitize_diagnostic
from cluxion_hermes_call.sessions import SessionCleanupReport

# labelless known-prefix keys must be scrubbed from diagnostics (they slipped through
# the labeled api_key=/bearer patterns before). Assembled from split literals so this
# source file does not itself trip the backup secret-scan; runtime value is a real key shape.
LABELLESS_KEYS = [
    "sk-ant-" + "A" * 24,
    "ghp_" + "B" * 32,
    "xai-" + "C" * 24,
    "hf_" + "D" * 24,
    "AKIA" + "J" * 16,
    "xoxb-" + "1" * 10 + "-" + "E" * 12,
]


def test_labelless_prefix_keys_redacted() -> None:
    for key in LABELLESS_KEYS:
        out = sanitize_diagnostic(f"error: leaked {key} here", prompt="")
        assert key not in out, f"leaked: {key} -> {out}"


def test_labeled_and_bearer_still_redacted() -> None:
    assert "hunter2" not in sanitize_diagnostic("api_key=hunter2", prompt="")
    assert "abc.def" not in sanitize_diagnostic("Authorization: Bearer abc.def", prompt="")


def test_benign_short_prefixes_not_over_redacted() -> None:
    # short strings sharing a prefix (below the min-length) must survive intact
    text = "var ghp_id and token pipeline hf_model here"
    assert sanitize_diagnostic(text, prompt="") == text


def test_prompt_substring_inside_labelless_sk_ant_still_redacted() -> None:
    # prompt-first redaction must not leave a labelless key body suffix unredacted
    body = "SECRETBODYVALUE" + "1" * 8
    key = "sk-ant-" + body
    prompt = "BODY"  # substring of the secret body
    out = sanitize_diagnostic(f"error: leaked {key} here", prompt=prompt)
    assert "[redacted]" in out, f"expected redaction marker: {out}"
    suffix = body[body.index(prompt) + len(prompt) :]
    assert suffix not in out, f"secret body suffix leaked: {suffix!r} in {out!r}"


def test_prompt_with_full_sk_ant_and_tail_fully_omitted() -> None:
    # secret redaction mutates the diagnostic before exact prompt replace; the full prompt
    # (secret + non-secret tail) must still be fully omitted — neither body nor tail may leak
    body = "SECRETBODYVALUE" + "1" * 8
    key = "sk-ant-" + body
    tail = " and then some non-secret user prompt tail"
    prompt = key + tail
    out = sanitize_diagnostic(f"error while handling: {prompt}", prompt=prompt)
    assert body not in out, f"secret body leaked: {out!r}"
    assert "non-secret user prompt tail" not in out, f"prompt tail leaked after secret redaction: {out!r}"


def _partial_overlap_cases() -> list[tuple[str, str, str, str]]:
    # labeled: prompt is secret value + customer tail (starts after api_key= label)
    labeled = (
        "error: api_key=abc123 private customer tail",
        "abc123 private customer tail",
        "abc123",
        "private customer tail",
    )
    # labelless sk-ant: prompt begins inside key body and includes a non-secret tail
    body = "SECRETBODYVALUE" + "1" * 8
    key = "sk-ant-" + body
    tail = " private customer tail"
    prompt = body[6:] + tail  # starts at "BODY..." inside the secret body
    labelless = (
        f"error: leaked {key}{tail}",
        prompt,
        body,
        "private customer tail",
    )
    return [labeled, labelless]


@pytest.mark.parametrize(
    "text,prompt,secret_body,customer_tail",
    _partial_overlap_cases(),
    ids=["labeled_api_key_partial", "labelless_sk_ant_body_partial"],
)
def test_partial_prompt_secret_overlap_omits_body_and_tail(
    text: str,
    prompt: str,
    secret_body: str,
    customer_tail: str,
) -> None:
    # exact prompt replace after secret scrub must not leave body/tail when they only
    # partially overlap the secret match (prompt starts inside the secret, continues past it)
    out = sanitize_diagnostic(text, prompt=prompt)
    assert secret_body not in out, f"secret body leaked: {out!r}"
    assert customer_tail not in out, f"customer tail leaked: {out!r}"
    assert "[prompt omitted]" in out, f"expected [prompt omitted]: {out!r}"


def test_emit_diagnostics_sanitizes_cleanup_reason_sentinel(capsys) -> None:
    """SessionCleanupReport.reason must be scrubbed (not only child stderr)."""
    sentinel = "SUPER_SECRET_PROMPT_TOKEN_XYZ_C98"
    options = CallOptions(prompt=sentinel)
    process_result = HermesProcessResult(stdout="", stderr="", returncode=1, timed_out=False)
    cleanup_report = SessionCleanupReport(
        cleaned=False,
        reason=f"delete_failed: leaked {sentinel} during cleanup",
    )

    core._emit_diagnostics(
        options=options,
        process_result=process_result,
        cleanup_report=cleanup_report,
        exit_code=1,
    )

    err = capsys.readouterr().err
    assert sentinel not in err
    assert "session cleanup skipped" in err
    assert "[prompt omitted]" in err


def test_emit_diagnostics_keeps_static_exit_code_when_prompt_is_digit(capsys) -> None:
    options = CallOptions(prompt="1")
    core._emit_diagnostics(
        options=options,
        process_result=HermesProcessResult(stdout="", stderr="", returncode=1, timed_out=False),
        cleanup_report=SessionCleanupReport(cleaned=True),
        exit_code=1,
    )

    err = capsys.readouterr().err
    assert "hermes exited with code 1" in err
    assert "[prompt omitted]" not in err
