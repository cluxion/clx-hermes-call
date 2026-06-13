# Hermes CLI Contract Verified For hermes-call

Verified on 2026-06-13 against `Hermes Agent v0.16.0 (2026.6.5) · upstream 4474873d`, binary `hermes` on PATH.

| Surface | Verified fact | Where verified |
| --- | --- | --- |
| `hermes -z PROMPT` / `--oneshot` | Runs a one-shot prompt and prints only the final response text to stdout. Tools, memory, rules, and AGENTS.md load normally. | `hermes --help`; `<hermes install dir>/hermes_cli/oneshot.py` module docstring. |
| `-t TOOLSETS` / `--toolsets TOOLSETS` | Comma-separated toolsets for `-z` and `--tui`; invalid toolsets make oneshot exit with code 2. | `hermes --help`; `hermes -z ... -t none` returned code 2 with “unknown --toolsets entries: none”. |
| Answer-only `--ask` mapping | `-t context_engine` is a valid toolsets value. In the current config it exposes no terminal, file, browser, or web tools; live doctor probe returned `NO_TOOLS`. | `toolsets.py` defines `context_engine` with an empty static tool list; live `hermes-call doctor --live`; `hermes config show` shows default compressor context. |
| Empty toolsets string | `-t ""` is accepted but does not disable tools; Hermes treats it like no explicit toolsets and uses configured CLI tools. | Live `hermes -z ... -t ""`; `oneshot.py` `_normalize_toolsets` returns `None` for an empty string, causing config toolsets to load. |
| `hermes sessions list` | Lists sessions as a human table. It supports `--source SOURCE` and `--limit LIMIT`; there is no JSON flag. IDs are printed in the last column and match `YYYYMMDD_HHMMSS_hex`. The table includes title/preview/activity/id, but not cwd. | `hermes sessions list --help`; live `hermes sessions list --source cli --limit 20`; `hermes_cli/main.py` session parser and table rendering. |
| `hermes sessions export` | Supports `output`, `--source`, and `--session-id`; `output=-` writes JSONL to stdout. A single-session export includes top-level `id`, `source`, `model`, `started_at`, `title`, `cwd`, and `messages`. `cwd` is usable for fail-closed concurrent cleanup candidate matching. | `hermes sessions export --help`; live `hermes sessions export - --source cli --session-id 20260612_233926_f701c7`; `hermes_state.py` `sessions.cwd TEXT` and `export_session()`. |
| `hermes sessions delete` | Syntax is `hermes sessions delete [--yes] session_id`. `--yes` skips confirmation. A successful delete prints `Deleted session '<id>'.` | `hermes sessions delete --help`; `hermes_cli/main.py` delete branch; live nonexistent-ID command verified noninteractive behavior. |
| Plugin CLI registration | `ctx.register_cli_command(name, help, setup_fn, handler_fn=None, description="")`; `setup_fn` receives an argparse subparser and `handler_fn` is installed via `set_defaults(func=...)`. | `<hermes install dir>/hermes_cli/plugins.py` around `register_cli_command`. |

## Notes

- `hermes-call --ask` intentionally uses `-t context_engine`, not `-t none` or `-t ""`.
- `hermes-call doctor` now verifies this contract in seconds: version parse, oneshot/toolsets help flags, sessions subcommands, live list-output parsing, and the jobs-root marker round-trip. `doctor --live` adds one answer-only no-tools probe and asserts session cleanup.
- Concurrent cleanup remains fail-closed. When more than one new session ID appears, `hermes-call` exports only those new IDs and deletes exactly one candidate if its exported `cwd` equals the wrapper run's resolved cwd. Export failure, no cwd, no match, or multiple cwd matches means no delete.
- `--sandbox` is the strongest cleanup mode because each run's cwd is a unique `~/.cluxion_hermes/jobs/<uuid>/work` path.
- The sandbox PID deletion gate is dependency-free. It records the wrapper PID and timestamp, but stdlib cannot reliably compare macOS process create time for an arbitrary live PID. The implementation therefore deletes when the marker PID is the current process or dead, and refuses deletion when another live PID owns the marker.
