"""cluxion-hermes-call-cli package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from cluxion_hermes_call.api import PostHermes, PostHermesError

try:
    __version__ = version("cluxion-hermes-call-cli")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.3.12"

__all__ = ["PostHermes", "PostHermesError", "__version__"]
