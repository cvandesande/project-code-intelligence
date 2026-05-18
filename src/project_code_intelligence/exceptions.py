"""Project-specific exception contracts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


class ConfigError(ValueError):
    """Raised when environment or CLI-derived configuration is invalid."""


class ProfileLoadError(ValueError):
    """Raised when a code-intelligence profile cannot be loaded."""


class McpProtocolError(ValueError):
    """Raised when an MCP request is well-typed JSON but semantically invalid."""


class McpProtocolTypeError(TypeError):
    """Raised when an MCP request has the wrong JSON shape or value type."""


class McpWritePermissionError(PermissionError):
    """Raised when an MCP write tool is requested while writes are disabled."""


class DatabaseConnectionError(RuntimeError):
    """Raised when the configured PostgreSQL connection is unavailable."""


class SarifFileTooLargeError(ValueError):
    """Raised when a SARIF file exceeds PCI_SARIF_MAX_BYTES.

    Replaces the older `raise ValueError("sarif_file_too_large")` sentinel-string
    pattern; carries structured context so callers don't need to parse a message.
    """

    def __init__(self, *, path: str, size_bytes: int, limit_bytes: int) -> None:
        super().__init__(f"sarif file {path!r} is {size_bytes} bytes (limit {limit_bytes})")
        self.path = path
        self.size_bytes = size_bytes
        self.limit_bytes = limit_bytes


class SarifLoadError(RuntimeError):
    """Raised when SARIF content cannot be loaded; carries structured context.

    Replaces the older `raise RuntimeError(json.dumps(...))` pattern where callers
    had to JSON-decode the message to recover the error context.
    """

    def __init__(self, *, context: Mapping[str, object]) -> None:
        # Preserve the legacy message shape (sorted-keys JSON) so existing
        # failure-record code that reads exc.args[0] keeps working.
        super().__init__(json.dumps(dict(context), sort_keys=True, default=str))
        self.context: Mapping[str, object] = context
