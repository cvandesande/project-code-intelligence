"""MCP JSON-RPC response and argument-boundary helpers."""

from __future__ import annotations

import json

from project_code_intelligence import config
from project_code_intelligence.exceptions import McpProtocolError, McpProtocolTypeError, McpWritePermissionError
from project_code_intelligence.models import JsonObject

Json = JsonObject
QueryParams = list[object]

DEFAULT_MAX_REQUEST_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_TEXT_CHARS = 8192
DEFAULT_MAX_METADATA_BYTES = 256 * 1024
DEFAULT_MAX_BATCH_ITEMS = 16
DEFAULT_MAX_RECORD_CONTENT_CHARS = 32 * 1024


def result_text(value: object) -> Json:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(value, indent=2, sort_keys=True, default=str),
            }
        ]
    }


def ok(value: object) -> Json:
    return result_text(value)


def mcp_max_request_bytes() -> int:
    return config.env_int(
        "PCI_MCP_MAX_REQUEST_BYTES",
        DEFAULT_MAX_REQUEST_BYTES,
        minimum=1024,
    )


def mcp_max_text_chars() -> int:
    return config.env_int("PCI_MCP_MAX_TEXT_CHARS", DEFAULT_MAX_TEXT_CHARS, minimum=1)


def mcp_max_metadata_bytes() -> int:
    return config.env_int(
        "PCI_MCP_MAX_METADATA_BYTES",
        DEFAULT_MAX_METADATA_BYTES,
        minimum=1024,
    )


def mcp_max_batch_items() -> int:
    return config.env_int("PCI_MCP_MAX_BATCH_ITEMS", DEFAULT_MAX_BATCH_ITEMS, minimum=1)


def mcp_max_record_content_chars() -> int:
    return config.env_int(
        "PCI_MCP_MAX_RECORD_CONTENT_CHARS",
        DEFAULT_MAX_RECORD_CONTENT_CHARS,
        minimum=1024,
    )


def mcp_debug_errors() -> bool:
    return config.env_bool("PCI_MCP_DEBUG_ERRORS", default=False)


def scoped_collection(args: Json) -> str | None:
    raw_requested = args.get("collection")
    requested_empty = isinstance(raw_requested, str) and not raw_requested
    requested = None if requested_empty else optional_text(args, "collection")
    configured = config.configured_collection()
    if requested_empty and (
        not configured or config.configured_collection_defaulted() or config.collection_override_allowed()
    ):
        return None
    if configured and requested and requested != configured and not config.collection_override_allowed():
        if config.configured_collection_defaulted():
            raise McpWritePermissionError(
                "collection does not match the MCP server's inferred cwd scope; "
                "omit collection and pass repo for repo-only lookup, or pass collection='' to ignore "
                "the inferred scope"
            )
        raise McpWritePermissionError(
            "collection does not match PCI_COLLECTION; "
            "omit collection to use the configured scope, or set "
            "PCI_ALLOW_COLLECTION_OVERRIDE=1 for trusted multi-collection access"
        )
    if not requested and optional_text(args, "repo") and config.configured_collection_defaulted():
        return None
    return requested or configured


def require_int(args: Json, name: str, default: int, minimum: int, maximum: int) -> int:
    value = args.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise McpProtocolTypeError(f"{name} must be an integer")
    return max(minimum, min(maximum, value))


def optional_int(args: Json, name: str, minimum: int = 1) -> int | None:
    value = args.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise McpProtocolTypeError(f"{name} must be an integer")
    if value < minimum:
        raise McpProtocolError(f"{name} must be greater than or equal to {minimum}")
    return value


def optional_bool(args: Json, name: str, *, default: bool = False) -> bool:
    value = args.get(name, default)
    if not isinstance(value, bool):
        raise McpProtocolTypeError(f"{name} must be a boolean")
    return value


def optional_text(args: Json, name: str) -> str | None:
    value = args.get(name)
    if value is None:
        return None
    if isinstance(value, str) and not value:
        return None
    if not isinstance(value, str):
        raise McpProtocolTypeError(f"{name} must be a string")
    if len(value) > mcp_max_text_chars():
        raise McpProtocolError(f"{name} exceeds PCI_MCP_MAX_TEXT_CHARS")
    return value
