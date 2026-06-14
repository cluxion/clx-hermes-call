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

## 사용

```bash
# 현재 폴더에서 작업 실행 (도구 사용 가능, codex exec 방식)
hermes-call "이 폴더의 실패하는 테스트를 고쳐줘"
hermes-call -C ~/project "pagination 버그 고쳐줘"     # 다른 폴더에서 실행
hermes-call -m "grok-4.3" -C ~/project -p "테스트를 고쳐줘"

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
규칙. 매 실행이 만든 세션은 실행 후 삭제됩니다(`--keep-session`으로 보존 가능). 기본 모드는 파일을
수정할 수 있으므로(Hermes의 oneshot 모드처럼 도구를 자동 승인), 완전히 신뢰하지 않는 프롬프트에는
`--ask`나 `--sandbox`를 사용하세요.

`--until-done`은 프롬프트 끝에 완료 계약을 덧붙입니다. Hermes가 답변 마지막 줄을 `TASK_COMPLETE`로
끝내면 완료로 보고, `WORK_REMAINS: ...`로 끝내면 같은 세션을 resume해서 이어갑니다. 완료 감지는
모델의 자기 보고에 의존하므로 완벽하지 않습니다. `--max-iterations`, `--timeout`, 세션 id 식별 실패
중 하나에 걸리면 `status=incomplete`로 정직하게 반환합니다.

현재 로컬 Hermes에서는 `hermes -r <id> -z ...`가 실제 resume을 하지 않는 것으로 검증되었습니다.
그래서 첫 실행은 clean stdout을 위해 `hermes -z`를 쓰고, 이어달리기는 검증된
`hermes chat -Q --resume <id> -q ...` 경로를 사용합니다.

Python 자동화에서는 import 시 모델 호출이 없습니다.

```python
from cluxion_hermes_call import PostHermes

answer = PostHermes(model="grok-4.3", path=".", prompt="질문")
result = PostHermes.run(model="grok-4.3", path=".", prompt="작업", until_done=True)
```

직접 호출은 성공 시 문자열을 반환하고 실패 시 `PostHermesError`를 raise합니다. `PostHermes.run(...)`은
`ok`, `answer`, `model`, `status`, `iterations`, `session_cleaned` 등을 가진 구조화 결과를 반환합니다.

Hermes 플러그인으로 설치하면 동일한 명령을 `hermes call "..."` 로도 쓸 수 있습니다.

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

## Use

```bash
# Run a task in the current folder (tools enabled, like codex exec)
hermes-call "fix the failing tests in this folder"
hermes-call -C ~/project "fix the pagination bug"     # run in another folder
hermes-call -m "grok-4.3" -C ~/project -p "fix the tests"

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
`AGENTS.md` rules. The session it creates is deleted after each run (use `--keep-session` to
keep it). The default mode can modify files (it auto-approves tools, like Hermes' own oneshot
mode), so use `--ask` or `--sandbox` for prompts you don't fully trust.

`--until-done` appends a completion contract to the prompt. If Hermes ends its answer with
`TASK_COMPLETE`, the run is complete. If it ends with `WORK_REMAINS: ...`, `hermes-call`
resumes the same session and continues. This detection depends on model self-reporting, so it
is not proof of real completion. `--max-iterations`, `--timeout`, and session-id ambiguity all
return `status=incomplete` honestly.

On the local verified Hermes build, `hermes -r <id> -z ...` did not actually resume. The wrapper
therefore uses `hermes -z` for the first clean oneshot, then the verified
`hermes chat -Q --resume <id> -q ...` path for continuation.

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

## License

Apache-2.0
