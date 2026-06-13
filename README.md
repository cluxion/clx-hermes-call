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

# 파일을 건드리지 않고 질문만
hermes-call --ask "이 스택 트레이스는 왜 발생해?"

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

# Ask a question without touching files
hermes-call --ask "why does this stack trace happen?"

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

Installed as a Hermes plugin, the same command is available as `hermes call "..."`.

## License

Apache-2.0
