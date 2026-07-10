from __future__ import annotations

import json
import tomllib
from pathlib import Path


def test_root_plugin_artifacts_are_version_synced() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]
    lock = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))

    claude = json.loads(Path(".claude-plugin/plugin.json").read_text(encoding="utf-8"))
    codex = json.loads(Path(".codex-plugin/plugin.json").read_text(encoding="utf-8"))

    assert claude == codex
    assert claude["name"] == "cluxion-hermes-call-cli"
    assert claude["version"] == version
    assert _locked_project_version(lock, "cluxion-hermes-call-cli") == version
    assert claude["commands"] == "./commands"
    assert claude["skills"] == "./skills"
    assert Path("commands/hermes-call.md").is_file()
    assert Path("skills/hermes-call/SKILL.md").is_file()


def test_no_legacy_surface_adapter_forks_exist() -> None:
    assert not Path("adapters").exists()


def _locked_project_version(lock: dict[str, object], package_name: str) -> str | None:
    packages = lock.get("package", [])
    if not isinstance(packages, list):
        return None
    for package in packages:
        if isinstance(package, dict) and package.get("name") == package_name:
            version = package.get("version")
            return str(version) if version is not None else None
    return None


def test_marketplace_manifest_is_version_synced() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]

    marketplace = json.loads(Path(".claude-plugin/marketplace.json").read_text(encoding="utf-8"))
    assert marketplace["plugins"][0]["version"] == version
    assert marketplace["plugins"][0]["source"] == "./"
