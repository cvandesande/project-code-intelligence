"""stdio MCP server entry point."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from project_code_intelligence.mcp.db import code_intel_tables_exist, table_regclass_exists
from project_code_intelligence.mcp.semantic import query_embedding, vector_literal_dimensions
from project_code_intelligence.mcp.status import static_status_rows
from project_code_intelligence.mcp.tools import TOOLS, advertised_tools
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
    result_response,
    server_version,
    write_response,
)
from project_code_intelligence.mcp.transport import main as stdio_main

if TYPE_CHECKING:
    from collections.abc import Sequence

_DESCRIPTION = (
    "pci-mcp is a stdio JSON-RPC MCP server. It is launched by an MCP client "
    "(VS Code, Claude Code, Codex, Cline, Zed, etc.); when you run it in a terminal it will "
    "block waiting for JSON-RPC requests on stdin. Configure it through environment variables "
    "(PCI_MCP_DATABASE_URL/USER/PASSWORD, "
    "PCI_COLLECTION, PCI_DATABASE_SCOPE_PATH). "
    "Run `pci-index --mcp-config <client>` to generate a ready-to-paste client config."
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pci-mcp", description=_DESCRIPTION)
    _ = parser.add_argument("--version", action="version", version=f"pci-mcp {server_version()}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if args:
        # argparse handles --help and --version itself (raising SystemExit) and rejects
        # unknown args. Only enter the stdio loop when nothing was supplied.
        _ = _parser().parse_args(args)
    return stdio_main()


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
    "server_version",
    "static_status_rows",
    "stdio_main",
    "table_regclass_exists",
    "vector_literal_dimensions",
    "write_response",
]


if __name__ == "__main__":
    raise SystemExit(main())
