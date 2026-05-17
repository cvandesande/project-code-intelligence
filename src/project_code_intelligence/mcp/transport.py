"""stdio JSON-RPC transport for the MCP server."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, cast

from project_code_intelligence import config, db, progress
from project_code_intelligence.common import default_collection
from project_code_intelligence.exceptions import McpProtocolError, McpProtocolTypeError, McpWritePermissionError
from project_code_intelligence.mcp.protocol import (
    Json,
    log,
    mcp_debug_errors,
    mcp_max_batch_items,
    mcp_max_request_bytes,
)
from project_code_intelligence.mcp.tool_catalog import validate_tool_arguments
from project_code_intelligence.mcp.tools import TOOLS, advertised_tools

if TYPE_CHECKING:
    from collections.abc import Iterator

    from project_code_intelligence.models import JsonObject, JsonValue

PROTOCOL_VERSION = "2024-11-05"


def set_mcp_environment_defaults() -> None:
    cwd = Path.cwd().resolve()
    if config.DATABASE_SCOPE_PATH_ENV not in os.environ:
        os.environ[config.DATABASE_SCOPE_PATH_ENV] = str(cwd)
    if "PROJECT_CODE_INTELLIGENCE_COLLECTION" not in os.environ:
        os.environ["PROJECT_CODE_INTELLIGENCE_COLLECTION"] = default_collection(cwd)
        os.environ["PROJECT_CODE_INTELLIGENCE_COLLECTION_DEFAULTED"] = "1"


def result_response(request_id: JsonValue, result: Json) -> Json:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def control_response(method: object, request_id: JsonValue) -> Json | None:
    if method == "initialize":
        return result_response(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "project-code-intelligence",
                    "version": "0.1.0",
                },
            },
        )
    if method == "ping":
        return result_response(request_id, {})
    if method == "tools/list":
        return result_response(request_id, {"tools": advertised_tools()})
    if method == "resources/list":
        return result_response(request_id, {"resources": []})
    if method == "prompts/list":
        return result_response(request_id, {"prompts": []})
    return None


def handle_tool_call(request: Json, request_id: JsonValue) -> Json:
    params_value = request.get("params", {})
    if not isinstance(params_value, dict):
        raise McpProtocolTypeError("params must be an object")
    params = params_value
    name = params.get("name")
    arguments_value = params.get("arguments", {})
    if not isinstance(name, str):
        raise McpProtocolTypeError("tool name must be a string")
    if name not in TOOLS:
        raise McpProtocolError(f"unknown tool: {name}")
    if not isinstance(arguments_value, dict):
        raise McpProtocolTypeError("arguments must be an object")
    arguments = arguments_value
    definition, handler = TOOLS[name]
    validate_tool_arguments(definition, arguments)
    if definition.write_tool and not db.allow_writes():
        raise McpWritePermissionError("writes are disabled")
    return result_response(request_id, handler(arguments))


def handle_request(request: Json) -> Json | None:
    method = request.get("method")
    request_id = request.get("id")

    if isinstance(method, str) and method.startswith("notifications/"):
        return None

    response = control_response(method, request_id)
    if response is not None:
        return response

    if method == "tools/call":
        return handle_tool_call(request, request_id)

    if not isinstance(method, str):
        raise McpProtocolTypeError("method must be a string")
    raise McpProtocolError(f"unsupported method: {method}")


def error_response(request_id: JsonValue, code: int, message: str) -> Json:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def write_response(response: Json | list[Json]) -> None:
    _ = sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
    _ = sys.stdout.flush()


def jsonrpc_input_lines() -> Iterator[str | None]:
    max_bytes = mcp_max_request_bytes()
    while True:
        raw = sys.stdin.buffer.readline(max_bytes + 1)
        if not raw:
            return
        if len(raw) > max_bytes:
            while raw and not raw.endswith(b"\n"):
                raw = sys.stdin.buffer.readline(8192)
            yield None
            continue
        yield raw.decode("utf-8", errors="strict")


def handle_batch_request(batch: list[object]) -> list[Json] | None:
    if len(batch) > mcp_max_batch_items():
        raise McpProtocolError("batch exceeds PROJECT_CODE_INTELLIGENCE_MCP_MAX_BATCH_ITEMS")
    responses: list[Json] = []
    for item in batch:
        if not isinstance(item, dict):
            raise McpProtocolTypeError("batch items must be objects")
        response = handle_request(cast("JsonObject", item))
        if response is not None:
            responses.append(response)
    return responses or None


def handle_jsonrpc_value(request_value: object) -> tuple[JsonValue, Json | list[Json] | None]:
    if isinstance(request_value, list):
        return None, handle_batch_request(cast("list[object]", request_value))
    if not isinstance(request_value, dict):
        raise McpProtocolTypeError("request must be an object")
    request = cast("JsonObject", request_value)
    request_id = request.get("id")
    return request_id, handle_request(request)


def request_id_from_jsonrpc_value(request_value: object) -> JsonValue:
    if not isinstance(request_value, dict):
        return None
    request = cast("JsonObject", request_value)
    return request.get("id")


def error_message(exc: BaseException) -> str:
    if isinstance(exc, db.DatabaseConnectionError):
        return str(exc)
    if isinstance(exc, (TypeError, ValueError, PermissionError, json.JSONDecodeError, UnicodeDecodeError)):
        return str(exc)
    if mcp_debug_errors():
        return str(exc)
    return "internal server error"


def main() -> int:
    set_mcp_environment_defaults()
    # MCP is a stdio JSON-RPC service: any progress_event firing inside a tool
    # handler (e.g. embedding endpoint retries during semantic search) should
    # be emitted as JSON on stderr, never as a Rich Live display, regardless
    # of FORCE_COLOR or similar env hints inherited from the launcher.
    _ = progress.set_emitter("json")
    for line_value in jsonrpc_input_lines():
        if line_value is None:
            write_response(
                error_response(None, -32000, "request exceeds PROJECT_CODE_INTELLIGENCE_MCP_MAX_REQUEST_BYTES")
            )
            continue
        line = line_value
        line = line.strip()
        if not line:
            continue
        request_id: JsonValue = None
        try:
            request_value = cast("object", json.loads(line))
            request_id = request_id_from_jsonrpc_value(request_value)
            request_id, response = handle_jsonrpc_value(request_value)
            if response is not None:
                write_response(response)
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive server boundary
            if mcp_debug_errors():
                log(traceback.format_exc())
            else:
                log(f"{type(exc).__name__}: {exc}")
            write_response(error_response(request_id, -32000, error_message(exc)))
    return 0
