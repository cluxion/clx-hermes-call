# Hermes CLI Contract Verified For hermes-call

Verified on 2026-06-14 against the local `hermes` binary on PATH and source at
`/Users/kimtaekyu/.hermes/hermes-agent`.

| Surface | Verified fact | Where verified |
| --- | --- | --- |
| `hermes -z PROMPT` / `--oneshot` | Runs a single prompt and prints only final response text. It creates a session in `~/.hermes/state.db`. | `/tmp/cluxion_design/HERMES_OFFICIAL_FACTS.md`; `hermes --help`; `hermes_cli/oneshot.py`; live `hermes -m grok-4.3 -z ...`. |
| `-m MODEL` / `--model MODEL` | Per-run model override for `-z`, `--tui`, and `chat -q`. Source resolves explicit model without writing config. | `hermes --help`; `hermes chat --help`; `hermes_cli/oneshot.py`; live `hermes -m grok-4.3 -z ...`. |
| `--provider PROVIDER` | Per-run provider override. For oneshot, `--provider` without a model is rejected as ambiguous. | `hermes --help`; `hermes_cli/oneshot.py`. |
| Default model config | Persistent default is read from `~/.hermes/config.yaml` under `model.default` and `model.provider`. Current local config reads as `xai-oauth/grok-4.3`. | `hermes_cli/config.py`; `~/.hermes/config.yaml` model block. |
| `-r ID` / `--resume ID`, `-c [NAME]` / `--continue [NAME]` | Advertised as top-level resume flags. `chat -q` also has explicit `--resume` and `--continue`. | `hermes --help`; `hermes chat --help`; `hermes_cli/main.py`. |
| `hermes -r ID -z PROMPT` | Not usable on this installed Hermes build: a live probe created a new session and did not see the prior prompt. The source oneshot path ignores `args.resume`. | Live probe with session `20260614_222425_5b9c81`; `hermes_cli/main.py` calls `run_oneshot(...)` without resume. Probe sessions were deleted. |
| `hermes chat -Q --resume ID -q PROMPT` | Verified noninteractive resume path. It restores prior context and prints a quiet preamble plus final text. `hermes-call` strips `↻ Resumed session ...` and `session_id: ...` lines. | Live probe resumed `20260614_222558_a19a5b`; cross-surface probe resumed a `hermes -z` session `20260614_222628_fed9ad`; probe sessions were deleted. |
| `-t TOOLSETS` / `--toolsets TOOLSETS` | Comma-separated toolsets for `-z` and `chat -q`. `context_engine` exists with an empty static tool list and is the current `--ask` no-tools mapping. | `hermes --help`; `hermes chat --help`; `toolsets.py`. |
| `hermes sessions list` | Supports `--source SOURCE` and `--limit LIMIT`; table output ends each row with a session id matching `YYYYMMDD_HHMMSS_hex`. | `hermes sessions list --help`; live `hermes sessions list --source cli --limit 50`; `hermes_cli/main.py`. |
| `hermes sessions export` | Supports `output`, `--source`, and `--session-id`; `output=-` writes JSONL to stdout. Export includes `id`, `model`, `cwd`, and `messages`. | `hermes sessions export --help`; live exports; `hermes_cli/main.py`; `hermes_state.py`. |
| `hermes sessions delete` | Syntax is `hermes sessions delete [--yes] session_id`. A successful delete prints `Deleted session '<id>'.` | `hermes sessions delete --help`; live deletion of probe sessions; `hermes_cli/main.py`. |
| Plugin CLI registration | `ctx.register_cli_command(name, help, setup_fn, handler_fn=None, description="")`; `setup_fn` receives an argparse subparser. | `hermes_cli/plugins.py`. |

## Wrapper Design Notes

- Default single-shot mode remains `hermes -z PROMPT`; `-m` is passed only when the caller provides it.
- `-C/--cd` changes the subprocess cwd. The wrapper does not pass a cwd flag to Hermes.
- `--until-done` is opt-in. It appends a completion contract requiring a final line of either `TASK_COMPLETE` or `WORK_REMAINS: ...`.
- The first `--until-done` turn uses `hermes -z` for clean stdout. The created session is selected through the existing fail-closed session diff plus exported `cwd` match.
- Continuation uses the verified local path `hermes chat -Q --resume <owned-id> -q ...`, not `hermes -r <id> -z ...`.
- Cleanup is still fail-closed: only the selected owned session id is deleted, and `--keep-session` preserves it.
- Completion detection is not a proof. A model can emit `TASK_COMPLETE` incorrectly. Caps, timeout, and honest `status=incomplete` reporting are the safety net.
- `--ask` still maps to `-t context_engine`. Hermes does not expose a no-tools discovery command, so `doctor` documents the dependency and `doctor --live` verifies behavior by asking for a tool action and expecting `NO_TOOLS`.
