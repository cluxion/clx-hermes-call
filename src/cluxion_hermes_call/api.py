"""Python automation API for hermes-call."""

from __future__ import annotations

import json as json_lib
from dataclasses import dataclass
from pathlib import Path

from cluxion_hermes_call.core import CallOptions, CallResult, run_call


class PostHermesError(RuntimeError):
    """Raised by the convenience API when Hermes does not return ok=True."""

    def __init__(self, result: CallResult) -> None:
        self.result = result
        status = result.status or "failed"
        super().__init__(f"PostHermes failed with status={status} exit_code={result.exit_code}")


@dataclass(frozen=True)
class _PostHermes:
    """Callable facade whose direct call returns text and .run returns structure."""

    def __call__(
        self,
        *,
        model: str | None = None,
        path: str | Path | None = None,
        prompt: str,
        until_done: bool = False,
        json: bool = False,
        timeout: float = 600.0,
    ) -> str:
        result = self.run(
            model=model,
            path=path,
            prompt=prompt,
            until_done=until_done,
            timeout=timeout,
        )
        if not result.ok:
            raise PostHermesError(result)
        if json:
            return json_lib.dumps(result.to_json_object(), ensure_ascii=False, separators=(",", ":"))
        return result.answer

    def run(
        self,
        *,
        model: str | None = None,
        path: str | Path | None = None,
        prompt: str,
        until_done: bool = False,
        json: bool = False,
        timeout: float = 600.0,
        max_iterations: int = 8,
        keep_session: bool = False,
        ask: bool = False,
        toolsets: str | None = None,
        hermes_bin: str = "hermes",
    ) -> CallResult:
        """Run Hermes and return the structured CallResult object."""
        del json
        result = run_call(
            CallOptions(
                prompt=prompt,
                ask=ask,
                cwd=Path(path).expanduser() if path is not None else None,
                json_mode=False,
                timeout_seconds=timeout,
                keep_session=keep_session,
                toolsets=toolsets,
                model=model,
                until_done=until_done,
                max_iterations=max_iterations,
                hermes_bin=hermes_bin,
            )
        )
        return result


PostHermes = _PostHermes()
