"""stdio MCP server entry point."""

from __future__ import annotations

from project_code_intelligence.mcp.db import code_intel_tables_exist, table_regclass_exists
from project_code_intelligence.mcp.tools import (
    TOOLS,
    advertised_tools,
    query_embedding,
    static_status_rows,
    vector_literal_dimensions,
)
from project_code_intelligence.mcp.transport import (
    PROTOCOL_VERSION,
    control_response,
    error_message,
    error_response,
    handle_batch_request,
    handle_jsonrpc_value,
    handle_request,
    handle_tool_call,
    jsonrpc_input_lines,
    main,
    result_response,
    write_response,
)

__all__ = [
    "PROTOCOL_VERSION",
    "TOOLS",
    "advertised_tools",
    "code_intel_tables_exist",
    "control_response",
    "error_message",
    "error_response",
    "handle_batch_request",
    "handle_jsonrpc_value",
    "handle_request",
    "handle_tool_call",
    "jsonrpc_input_lines",
    "main",
    "query_embedding",
    "result_response",
    "static_status_rows",
    "table_regclass_exists",
    "vector_literal_dimensions",
    "write_response",
]


if __name__ == "__main__":
    raise SystemExit(main())
