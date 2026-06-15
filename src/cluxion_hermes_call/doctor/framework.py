"""Plugin-agnostic doctor framework. Identical copy in every plugin."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SEVERITY_RANK: dict[str, int] = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
}


@dataclass(frozen=True)
class CatalogEntry:
    check_id: str
    category: str
    severity: str
    what_it_checks: str
    failure_symptom: str
    likely_causes: tuple[str, ...]
    fix_steps: tuple[str, ...]
    change_robust: str


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    category: str
    severity: str
    status: str
    detail: str


@dataclass(frozen=True)
class DoctorResult:
    plugin: str
    version: str
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return not any(c.status == "fail" for c in self.checks)

    def to_json_object(self) -> dict[str, Any]:
        return {
            "plugin": self.plugin,
            "version": self.version,
            "checks": [
                {
                    "check_id": c.check_id,
                    "category": c.category,
                    "severity": c.severity,
                    "status": c.status,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
            "ok": self.ok,
        }


class DoctorContext:
    def __init__(
        self, cwd: Path, hermes_bin: str, run: Callable[[list[str]], subprocess.CompletedProcess]
    ) -> None:
        self.cwd = cwd
        self.hermes_bin = hermes_bin
        self.run = run


def load_catalog(catalog_path: Path) -> tuple[CatalogEntry, ...]:
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    entries: list[CatalogEntry] = []
    fields = {
        "check_id",
        "category",
        "severity",
        "what_it_checks",
        "failure_symptom",
        "likely_causes",
        "fix_steps",
        "change_robust",
    }
    for item in raw:
        if not isinstance(item, dict):
            continue
        data: dict[str, Any] = {}
        for f in fields:
            val = item.get(f)
            if f in ("likely_causes", "fix_steps") and isinstance(val, list):
                data[f] = tuple(val)
            elif val is not None:
                data[f] = val
        if "check_id" in data:
            entries.append(CatalogEntry(**data))  # type: ignore[arg-type]
    return tuple(entries)


def _make_runner(timeout: float = 8.0) -> Callable[[list[str]], subprocess.CompletedProcess]:
    def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )

    return _run


def run_doctor(
    *,
    cwd: Path,
    hermes_bin: str = "hermes",
    catalog_path: Path,
    probes: dict[str, Callable[[DoctorContext], tuple[str, str]]],
    plugin: str,
    version: str,
) -> DoctorResult:
    catalog = load_catalog(catalog_path)
    ctx = DoctorContext(cwd=cwd, hermes_bin=hermes_bin, run=_make_runner())
    results: list[CheckResult] = []
    for entry in catalog:
        if entry.check_id in probes:
            try:
                status, detail = probes[entry.check_id](ctx)
            except Exception as e:
                status, detail = "fail", f"probe raised {type(e).__name__}: {e}"
        else:
            status, detail = "skip", "no probe registered"
        results.append(
            CheckResult(
                check_id=entry.check_id,
                category=entry.category,
                severity=entry.severity,
                status=status,
                detail=detail,
            )
        )
    results.sort(key=lambda c: (SEVERITY_RANK.get(c.severity, 9), c.check_id))
    return DoctorResult(plugin=plugin, version=version, checks=tuple(results))


def render_json(result: DoctorResult) -> str:
    return json.dumps(
        result.to_json_object(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def render_text(
    result: DoctorResult, catalog: tuple[CatalogEntry, ...], *, verbose: bool = False
) -> str:
    entry_map = {e.check_id: e for e in catalog}
    lines: list[str] = []
    for c in result.checks:
        entry = entry_map.get(c.check_id)
        line = f"{c.status} [{c.severity}] {c.check_id}: {c.detail}"
        lines.append(line)
        if entry and c.status in ("fail", "warn"):
            lines.append(f"  symptom: {entry.failure_symptom}")
            for i, step in enumerate(entry.fix_steps, 1):
                lines.append(f"  {i}. {step}")
        if verbose and entry:
            lines.append(f"  what: {entry.what_it_checks}")
            for cause in entry.likely_causes:
                lines.append(f"  cause: {cause}")
            lines.append(f"  robust: {entry.change_robust}")
    return "\n".join(lines)
