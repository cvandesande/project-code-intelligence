"""Project-specific exception contracts."""

from __future__ import annotations


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
