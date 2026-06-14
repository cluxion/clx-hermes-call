"""Helpers for reading public Hermes configuration files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DefaultModelInfo:
    """Default Hermes model information read from config.yaml."""

    model: str | None
    provider: str | None
    error: str | None = None

    def display(self) -> str:
        """Return a user-facing default model line value."""
        if self.error:
            return f"unavailable ({self.error})"
        if self.model and self.provider:
            return f"{self.provider}/{self.model}"
        if self.model:
            return self.model
        return "unavailable (model.default is not set)"


def default_config_path() -> Path:
    """Return the Hermes config path used by this lightweight reader."""
    home = os.environ.get("HERMES_HOME")
    if home:
        return Path(home).expanduser() / "config.yaml"
    return Path.home() / ".hermes" / "config.yaml"


def read_default_model(path: Path | None = None) -> DefaultModelInfo:
    """Read model.default and model.provider from ~/.hermes/config.yaml."""
    config_path = default_config_path() if path is None else path
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        return DefaultModelInfo(None, None, f"{config_path}: {exc.strerror or exc}")

    try:
        return _parse_model_block(text)
    except ValueError as exc:
        return DefaultModelInfo(None, None, str(exc))


def default_model_help_line(path: Path | None = None) -> str:
    """Return the help epilog line showing the current Hermes default model."""
    return f"Default model: {read_default_model(path).display()}"


def _parse_model_block(text: str) -> DefaultModelInfo:
    model_value: str | None = None
    provider: str | None = None
    in_model_block = False
    model_indent: int | None = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if not in_model_block:
            if indent == 0 and line.startswith("model:"):
                value = _clean_yaml_scalar(line.split(":", 1)[1])
                if value:
                    return DefaultModelInfo(value, None)
                in_model_block = True
                model_indent = indent
            continue

        if model_indent is not None and indent <= model_indent:
            break
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        value = _clean_yaml_scalar(raw_value)
        if key.strip() in {"default", "model"} and value:
            model_value = value
        elif key.strip() == "provider" and value:
            provider = value

    if not in_model_block:
        return DefaultModelInfo(None, None, "model block not found")
    return DefaultModelInfo(model_value, provider)


def _clean_yaml_scalar(value: str) -> str:
    cleaned = value.split(" #", 1)[0].strip()
    if cleaned in {"", "null", "Null", "NULL", "~"}:
        return ""
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        return cleaned[1:-1]
    return cleaned
