========= Written in Korean first, then English ==========

======== 한국어 ========

# cluxion-hermes-call-cli

Hermes Agent를 AI API처럼 사용하세요. `hermes-call`은 [Hermes Agent](https://hermes-agent.nousresearch.com)에
이미 설정해 둔 모델에 프롬프트 하나를 보내 답을 받아오고 — codex exec 방식 — 자신이 만든 세션을
정리합니다. API 키나 모델 설정이 필요 없습니다. Hermes가 이미 돌리고 있는 기본 모델(xAI,
OpenAI 호환 엔드포인트, 또는 로컬 vLLM/MLX 모델)을 그대로 재사용합니다.

## 설치

```bash
pip install cluxion-hermes-call-cli
# 또는 독립 명령으로:
uv tool install cluxion-hermes-call-cli
```

`hermes`가 PATH에 있고 기본 모델이 설정된 정상 동작 Hermes Agent 설치가 필요합니다.

### Codex / Claude 플러그인 설치

이 repo root는 Codex/Claude 공통 marketplace plugin artifact입니다. 먼저 위 CLI가 host PATH에서
실행 가능해야 합니다.

```bash
codex plugin marketplace add cluxion-local /path/to/cluxion-Hermes-call-cli
codex plugin add cluxion-hermes-call-cli@cluxion-local
```

Claude Code에서는 같은 repo root의 `.claude-plugin/plugin.json`을 플러그인으로 설치하세요. 두 host 모두
`commands/hermes-call.md`와 `skills/hermes-call/SKILL.md`를 발견하고, host가 직접 모델을 호출하지 않고
`hermes-call --json ...` CLI 계약으로 위임합니다.

## 사용

```bash
# 현재 폴더에서 작업 실행 (도구 사용 가능, codex exec 방식)
hermes-call "이 폴더의 실패하는 테스트를 고쳐줘"
hermes-call -C ~/project "pagination 버그 고쳐줘"     # 다른 폴더에서 실행
hermes-call -m "grok-4.3" -C ~/project --prompt "테스트를 고쳐줘"

# 파일을 건드리지 않고 질문만
hermes-call --ask "이 스택 트레이스는 왜 발생해?"

# 한 세션을 이어가며 완료 표식을 볼 때까지 실행
hermes-call --until-done --max-iterations 8 "이 작업을 끝까지 처리해줘"

# 작업을 파이프로 전달 (스크립트나 다른 에이전트의 위임용)
cat task.md | hermes-call -
hermes-call --json --timeout 300 "..."     # {ok, answer, model, ...}; exit 0/1/2/124

# 신뢰 안 되는/실험적 프롬프트는 일회용 폴더에서
hermes-call --sandbox "리팩터를 시도해봐"

# Hermes 업그레이드 후 정상 동작 점검
hermes-call doctor
```

폴더에서 직접 입력하는 프롬프트와 정확히 동일하게 실행됩니다 — 같은 모델, 같은 도구, 같은 `AGENTS.md`
규칙. 매 실행이 만든 세션은 실행 후 삭제를 시도합니다(`--keep-session`으로 보존 가능). 세션 식별이
모호하거나, Hermes list/export/delete가 실패하거나, timeout/error 경로에 걸리면 안전하게 삭제를 건너뛰고
이유를 보고할 수 있습니다. 남은 untitled CLI 세션은 `hermes-call gc --sessions`로 dry-run 확인 후
`hermes-call gc --sessions --apply`로 정리하세요. 기본 모드는 파일을 수정할 수 있으므로(Hermes의
oneshot 모드처럼 도구를 자동 승인), 완전히 신뢰하지 않는 프롬프트에는 `--ask`나 `--sandbox`를 사용하세요.

`--until-done`은 프롬프트 끝에 완료 계약을 덧붙입니다. 마지막 non-empty 줄이 공백과 대소문자를 무시해
`TASK_COMPLETE`이면 완료로 보고, `WORK_REMAINS: ...`이면 같은 세션을 resume해서 이어갑니다. marker는
마지막 줄에만 의미가 있으며 본문 중간의 같은 문자열은 무시됩니다. 완료 감지는 모델의 자기 보고에
의존하므로 완벽하지 않습니다. `--max-iterations`, `--timeout`, 세션 id 식별 실패 중 하나에 걸리면
`status=incomplete`로 정직하게 반환합니다.

현재 로컬 Hermes에서는 `hermes -r <id> -z ...`가 실제 resume을 하지 않는 것으로 검증되었습니다.
그래서 첫 실행은 clean stdout을 위해 `hermes -z`를 쓰고, 이어달리기는 검증된
`hermes chat -Q --resume <id> -q ...` 경로를 사용합니다.

`--json` 모드의 사용법/입력 오류는 stdout에 `{ok:false,error,message,hint,exit_code}` JSON을 쓰고
exit 2로 종료합니다. 프롬프트의 null byte는 `invalid_prompt`, 256KB 이상 프롬프트는
`prompt_too_large`로 spawn 전에 거부합니다. `--timeout`은 최대 86400초이며, timeout 후 종료 대기
최악치는 대략 `timeout + min(5, max(0.5, timeout * 0.5))`입니다.

세션 snapshot/GC용 `hermes sessions ...` subprocess도 추적/정리되며 기본 30초 timeout을 가집니다.
필요하면 `CLUXION_HERMES_CALL_SESSION_TIMEOUT`으로 조정할 수 있습니다.

Python 자동화에서는 import 시 모델 호출이 없습니다.

```python
from cluxion_hermes_call import PostHermes

answer = PostHermes(model="grok-4.3", path=".", prompt="질문")
result = PostHermes.run(model="grok-4.3", path=".", prompt="작업", until_done=True)
```

직접 호출은 성공 시 문자열을 반환하고 실패 시 `PostHermesError`를 raise합니다. `PostHermes.run(...)`은
`ok`, `answer`, `model`, `status`, `iterations`, `session_cleaned` 등을 가진 구조화 결과를 반환합니다.

Hermes 플러그인으로 설치하면 동일한 명령을 `hermes call "..."` 로도 쓸 수 있습니다.

## Hermes 슬래시 커맨드 (0.3.10+)

세션 안에서 codex-exec 스타일 단발 실행:

```
/hermes-call 이 폴더의 실패 테스트를 고쳐줘
/hermes-call-doctor
```

`/` 입력 시 🔌로 표시 · 터미널 `hermes-call`과 동일 엔진, Hermes 세션 cwd·맥락 유지.

## 라이선스

Apache-2.0

============ English ==========

# cluxion-hermes-call-cli

Run your Hermes Agent like an AI API. `hermes-call` sends one prompt to the model you already
configured in [Hermes Agent](https://hermes-agent.nousresearch.com), gives you the answer back
— codex-exec style — and cleans up the session it created. No API key or model setup: it
reuses whatever default model your Hermes already runs (xAI, OpenAI-compatible endpoints, or
local vLLM/MLX models).

## Install

```bash
pip install cluxion-hermes-call-cli
# or, as a standalone command:
uv tool install cluxion-hermes-call-cli
```

Requires a working Hermes Agent install with `hermes` on your PATH and a default model set.

### Codex / Claude plugin install

This repo root is the shared Codex/Claude marketplace plugin artifact. The CLI above must be
available on the host PATH first.

```bash
codex plugin marketplace add cluxion-local /path/to/cluxion-Hermes-call-cli
codex plugin add cluxion-hermes-call-cli@cluxion-local
```

In Claude Code, install the same repo root from `.claude-plugin/plugin.json`. Both hosts discover
`commands/hermes-call.md` and `skills/hermes-call/SKILL.md`; they delegate through the
`hermes-call --json ...` CLI contract instead of owning model execution.

## Use

```bash
# Run a task in the current folder (tools enabled, like codex exec)
hermes-call "fix the failing tests in this folder"
hermes-call -C ~/project "fix the pagination bug"     # run in another folder
hermes-call -m "grok-4.3" -C ~/project --prompt "fix the tests"

# Ask a question without touching files
hermes-call --ask "why does this stack trace happen?"

# Keep resuming one owned session until a completion marker is observed
hermes-call --until-done --max-iterations 8 "finish this task end to end"

# Pipe a task in (for scripts or other agents delegating work)
cat task.md | hermes-call -
hermes-call --json --timeout 300 "..."     # {ok, answer, model, ...}; exit 0/1/2/124

# Run untrusted or experimental prompts in a throwaway folder
hermes-call --sandbox "try a refactor"

# Check it still works after upgrading Hermes
hermes-call doctor
```

It runs exactly the prompt you'd type yourself in that folder — same model, tools, and
`AGENTS.md` rules. The session it creates is cleaned up best-effort and fail-closed after each
run (use `--keep-session` to keep it). Ambiguous session selection, Hermes list/export/delete
failures, timeout paths, and error paths can leave sessions behind. Inspect stale untitled CLI
sessions with `hermes-call gc --sessions`, then delete them with `hermes-call gc --sessions --apply`.
The default mode can modify files (it auto-approves tools, like Hermes' own oneshot mode), so use
`--ask` (answer-only, no tools) or `--sandbox` for prompts you don't fully trust.

`--until-done` appends a completion contract to the prompt. If the last non-empty answer line is
`TASK_COMPLETE` after trimming whitespace and ignoring case, the run is complete. If that final
line starts with `WORK_REMAINS: ...` after the same normalization, `hermes-call` resumes the same
session and continues. Markers only count on the final line; the same text in the body is ignored.
This detection depends on model self-reporting, so it is not proof of real completion.
`--max-iterations`, `--timeout`, and session-id ambiguity all return `status=incomplete` honestly.

On the local verified Hermes build, `hermes -r <id> -z ...` did not actually resume. The wrapper
therefore uses `hermes -z` for the first clean oneshot, then the verified
`hermes chat -Q --resume <id> -q ...` path for continuation.

In `--json` mode, usage/input errors write `{ok:false,error,message,hint,exit_code}` JSON to stdout
and exit 2. Null bytes are rejected as `invalid_prompt`, and prompts of 256KB or more are rejected
as `prompt_too_large` before spawning Hermes. `--timeout` is capped at 86400 seconds; timeout
worst-case is about `timeout + min(5, max(0.5, timeout * 0.5))`.

Session snapshot/GC `hermes sessions ...` subprocesses are tracked and have a default 30 second
timeout. Override it with `CLUXION_HERMES_CALL_SESSION_TIMEOUT` when needed.

Python automation performs no model call at import time:

```python
from cluxion_hermes_call import PostHermes

answer = PostHermes(model="grok-4.3", path=".", prompt="question")
result = PostHermes.run(model="grok-4.3", path=".", prompt="task", until_done=True)
```

The direct call returns a string on success and raises `PostHermesError` on failure.
`PostHermes.run(...)` returns the structured result object with `ok`, `answer`, `model`,
`status`, `iterations`, and `session_cleaned`.

Installed as a Hermes plugin, the same command is available as `hermes call "..."`.

## Hermes slash commands (0.3.10+)

In-session codex-exec style one-shot:

```
/hermes-call fix the failing tests in this folder
/hermes-call-doctor
```

Shows in `/` autocomplete with 🔌 · same engine as terminal `hermes-call`, keeps Hermes session cwd/context.

## License

Apache-2.0
