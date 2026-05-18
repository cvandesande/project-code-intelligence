"""Unit tests for `project_code_intelligence.mcp.transport`.

Focuses on the framing, batching, and error-mapping branches not already
exercised by tests/test_mcp_contracts.py — specifically the oversize-line
path in `jsonrpc_input_lines`, the three branches of `handle_batch_request`,
the dispatch table inside `handle_jsonrpc_value`,
`request_id_from_jsonrpc_value`, and the exception-to-message ladder in
`error_message`.
"""

from __future__ import annotations

import io
import json
import unittest
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

from project_code_intelligence import db as pci_db
from project_code_intelligence import process as pci_process
from project_code_intelligence.exceptions import McpProtocolError, McpProtocolTypeError, McpWritePermissionError
from project_code_intelligence.mcp import transport as mcp_transport
from project_code_intelligence.mcp.tool_catalog import ToolDefinition

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject


class FakeStdinBuffer:
    """Minimal `sys.stdin.buffer` stand-in for `jsonrpc_input_lines`.

    Behaviour mirrors `BufferedReader.readline(limit)`: it reads up to the
    given byte cap, stopping early at the next newline. This is enough to
    drive both the happy path and the oversize-line drain path in
    `jsonrpc_input_lines`.
    """

    def __init__(self, data: bytes) -> None:
        self._buf = io.BytesIO(data)

    def readline(self, size: int = -1) -> bytes:
        if size < 0:
            return self._buf.readline()
        # BytesIO.readline(size) honours the cap and the newline boundary.
        return self._buf.readline(size)


class JsonrpcInputLinesTests(unittest.TestCase):
    """Cover the framing iterator in transport.py."""

    def test_yields_decoded_lines_until_eof(self) -> None:
        # Two well-formed lines, then EOF (empty readline) ends the iterator.
        fake_buffer = FakeStdinBuffer(b'{"a":1}\n{"b":2}\n')

        # Pin the limit well above the input so the oversize path stays inert.
        with (
            patch.object(mcp_transport.sys, "stdin", SimpleNamespace(buffer=fake_buffer)),
            patch.object(mcp_transport, "mcp_max_request_bytes", return_value=1024),
        ):
            lines = list(mcp_transport.jsonrpc_input_lines())

        self.assertEqual(lines, ['{"a":1}\n', '{"b":2}\n'])

    def test_oversize_line_yields_none_and_drains_remainder(self) -> None:
        # A single line that exceeds the byte cap. The iterator should yield
        # None for that line (so main() can emit a structured -32000 error)
        # and then return cleanly on EOF — drained, not stuck.
        # Use a payload that is larger than the configured 1024-byte limit.
        oversize_payload = b'{"x":"' + (b"A" * 2048) + b'"}\n'
        fake_buffer = FakeStdinBuffer(oversize_payload)

        with (
            patch.object(mcp_transport.sys, "stdin", SimpleNamespace(buffer=fake_buffer)),
            patch.object(mcp_transport, "mcp_max_request_bytes", return_value=1024),
        ):
            results = list(mcp_transport.jsonrpc_input_lines())

        # Exactly one yield: the None sentinel for the oversize request.
        # The drain loop must consume the rest of the line and return on EOF.
        self.assertEqual(results, [None])

    def test_oversize_line_does_not_prevent_subsequent_well_formed_line(self) -> None:
        # Mixed sequence: oversize line + well-formed follow-up. The iterator
        # should yield None, drain to the newline, then yield the next line.
        oversize_line = b'{"x":"' + (b"B" * 2048) + b'"}\n'
        followup = b'{"jsonrpc":"2.0","id":1}\n'
        fake_buffer = FakeStdinBuffer(oversize_line + followup)

        with (
            patch.object(mcp_transport.sys, "stdin", SimpleNamespace(buffer=fake_buffer)),
            patch.object(mcp_transport, "mcp_max_request_bytes", return_value=1024),
        ):
            results = list(mcp_transport.jsonrpc_input_lines())

        self.assertEqual(results, [None, '{"jsonrpc":"2.0","id":1}\n'])


class HandleBatchRequestTests(unittest.TestCase):
    """Cover the three branches of `handle_batch_request`."""

    def test_oversize_batch_raises_protocol_error(self) -> None:
        # Limit batch items to 1 and submit a 2-item batch.
        with (
            patch.object(mcp_transport, "mcp_max_batch_items", return_value=1),
            self.assertRaises(McpProtocolError) as ctx,
        ):
            _ = mcp_transport.handle_batch_request([
                {"jsonrpc": "2.0", "id": 1, "method": "ping"},
                {"jsonrpc": "2.0", "id": 2, "method": "ping"},
            ])

        self.assertIn("PCI_MCP_MAX_BATCH_ITEMS", str(ctx.exception))

    def test_non_object_batch_item_raises_type_error(self) -> None:
        with (
            patch.object(mcp_transport, "mcp_max_batch_items", return_value=16),
            self.assertRaises(McpProtocolTypeError),
        ):
            # 42 is not a JSON object; the iterator should reject it.
            _ = mcp_transport.handle_batch_request([42])

    def test_batch_of_only_notifications_returns_none(self) -> None:
        # `handle_request` returns None for any method starting with
        # "notifications/", so a batch of only notifications should produce
        # no response payload at all (None, not []).
        responses = mcp_transport.handle_batch_request([
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "method": "notifications/cancelled"},
        ])

        self.assertIsNone(responses)

    def test_batch_collects_responses_for_non_notifications(self) -> None:
        # Two ping requests + one notification. Only the pings produce
        # responses, in submission order.
        responses = mcp_transport.handle_batch_request([
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "ping"},
        ])

        self.assertIsNotNone(responses)
        responses_list = cast("list[dict[str, object]]", responses)
        self.assertEqual(len(responses_list), 2)
        self.assertEqual(responses_list[0]["id"], 1)
        self.assertEqual(responses_list[1]["id"], 2)
        # Both ping responses should carry an empty result body.
        self.assertEqual(responses_list[0]["result"], {})
        self.assertEqual(responses_list[1]["result"], {})


class HandleJsonrpcValueTests(unittest.TestCase):
    """Cover the dispatch table in `handle_jsonrpc_value`."""

    def test_list_request_value_dispatches_to_batch_handler(self) -> None:
        request_id, response = mcp_transport.handle_jsonrpc_value([
            {"jsonrpc": "2.0", "id": 7, "method": "ping"},
        ])

        # Batch responses do not carry a top-level id (the per-item ids live
        # inside the response list).
        self.assertIsNone(request_id)
        self.assertIsInstance(response, list)
        response_list = cast("list[dict[str, object]]", response)
        self.assertEqual(response_list[0]["id"], 7)

    def test_dict_request_value_dispatches_to_single_handler(self) -> None:
        request_id, response = mcp_transport.handle_jsonrpc_value(
            {"jsonrpc": "2.0", "id": 99, "method": "ping"},
        )

        self.assertEqual(request_id, 99)
        # `ping` returns a single response object, not a list.
        self.assertIsInstance(response, dict)
        single = cast("dict[str, object]", response)
        self.assertEqual(single["id"], 99)
        self.assertEqual(single["result"], {})

    def test_non_object_non_array_request_value_raises_type_error(self) -> None:
        with self.assertRaises(McpProtocolTypeError):
            _ = mcp_transport.handle_jsonrpc_value("not a request")


class RequestIdFromJsonrpcValueTests(unittest.TestCase):
    """Cover the small request-id extraction helper."""

    def test_returns_none_for_non_dict_value(self) -> None:
        self.assertIsNone(mcp_transport.request_id_from_jsonrpc_value([1, 2, 3]))
        self.assertIsNone(mcp_transport.request_id_from_jsonrpc_value("string"))
        self.assertIsNone(mcp_transport.request_id_from_jsonrpc_value(None))

    def test_returns_id_field_for_dict_value(self) -> None:
        self.assertEqual(
            mcp_transport.request_id_from_jsonrpc_value({"id": 42, "method": "ping"}),
            42,
        )
        # Missing id key is `.get(...)` → None, not a KeyError.
        self.assertIsNone(mcp_transport.request_id_from_jsonrpc_value({"method": "ping"}))


class ErrorMessageTests(unittest.TestCase):
    """Cover the exception-to-message mapping ladder.

    `error_message` is what turns a raised exception into a user-visible
    JSON-RPC error message — covering each branch protects the diagnostic
    contract for MCP clients.
    """

    def test_database_connection_error_message_passes_through(self) -> None:
        exc = pci_db.DatabaseConnectionError("specific db unreachable text")
        self.assertEqual(mcp_transport.error_message(exc), "specific db unreachable text")

    def test_value_error_message_passes_through(self) -> None:
        # ValueError sits inside the known-exception tuple branch.
        exc = ValueError("bad argument shape")
        self.assertEqual(mcp_transport.error_message(exc), "bad argument shape")

    def test_unknown_exception_returns_internal_server_error_by_default(self) -> None:
        # RuntimeError is not in the known-exception tuple, debug flag off.
        with patch.object(mcp_transport, "mcp_debug_errors", return_value=False):
            self.assertEqual(
                mcp_transport.error_message(RuntimeError("internal detail")),
                "internal server error",
            )

    def test_unknown_exception_passes_through_when_debug_enabled(self) -> None:
        # With debug on, the original message must surface — operators rely
        # on this to triage MCP failures in test environments.
        with patch.object(mcp_transport, "mcp_debug_errors", return_value=True):
            self.assertEqual(
                mcp_transport.error_message(RuntimeError("revealing detail")),
                "revealing detail",
            )

    def test_json_decode_error_message_passes_through(self) -> None:
        # json.JSONDecodeError is in the known tuple too — verify the
        # str(exc) path produces a useful description rather than the
        # opaque fallback.
        try:
            _ = cast("object", json.loads("not json"))
        except json.JSONDecodeError as exc:
            self.assertNotEqual(mcp_transport.error_message(exc), "internal server error")
            return
        self.fail("json.loads should have raised JSONDecodeError")


class GitShortCommitTests(unittest.TestCase):
    """`_git_short_commit` is internal; exercise its branches through
    `server_version`, the only public caller. Pyright strict mode flags
    direct private access (`reportPrivateUsage`), so we drive these
    branches by checking the version string that surfaces them.
    """

    def test_no_commit_suffix_when_process_run_raises_os_error(self) -> None:
        # OSError (e.g. git not installed) → _git_short_commit returns None
        # → server_version emits just the base version, no "+<commit>" tail.
        with (
            patch.object(mcp_transport.process, "run", side_effect=OSError("git missing")),
            patch.object(mcp_transport.importlib_metadata, "version", return_value="0.1.0"),
        ):
            self.assertEqual(mcp_transport.server_version(), "0.1.0")

    def test_no_commit_suffix_when_subprocess_returncode_is_nonzero(self) -> None:
        # Non-checkout cwd → git rev-parse exits non-zero with check=False.
        # _git_short_commit returns None, so server_version omits the commit.
        non_zero = pci_process.CompletedProcess(args=["git"], returncode=128, stdout="", stderr="fatal")
        with (
            patch.object(mcp_transport.process, "run", return_value=non_zero),
            patch.object(mcp_transport.importlib_metadata, "version", return_value="0.1.0"),
        ):
            self.assertEqual(mcp_transport.server_version(), "0.1.0")

    def test_no_commit_suffix_when_subprocess_returns_empty_stdout(self) -> None:
        # `rev-parse` returns 0 but with empty stdout (extremely unusual,
        # but the empty-string check in _git_short_commit guards against it
        # producing "<version>+" with a dangling separator).
        empty = pci_process.CompletedProcess(args=["git"], returncode=0, stdout="   \n", stderr="")
        with (
            patch.object(mcp_transport.process, "run", return_value=empty),
            patch.object(mcp_transport.importlib_metadata, "version", return_value="0.1.0"),
        ):
            self.assertEqual(mcp_transport.server_version(), "0.1.0")


class ServerVersionTests(unittest.TestCase):
    """Pin the PackageNotFoundError fallback so the function never raises."""

    def test_falls_back_to_unknown_when_metadata_lookup_fails(self) -> None:
        # Mirror an environment where the package is not pip-installed.
        with (
            patch.object(
                mcp_transport.importlib_metadata,
                "version",
                side_effect=mcp_transport.importlib_metadata.PackageNotFoundError("project-code-intelligence"),
            ),
            patch.object(mcp_transport, "_git_short_commit", return_value=None),
        ):
            self.assertEqual(mcp_transport.server_version(), "unknown")


class ControlResponseTests(unittest.TestCase):
    """The non-initialize control branches are list endpoints. Pin each one."""

    def test_tools_list_returns_advertised_tools_payload(self) -> None:
        response = mcp_transport.control_response("tools/list", 1)
        self.assertIsNotNone(response)
        body = cast("dict[str, object]", response)
        result = cast("dict[str, object]", body["result"])
        # Advertised tools must be a list. Don't pin the contents — that's
        # the catalog's job to evolve — just the envelope shape.
        self.assertIsInstance(result["tools"], list)

    def test_resources_list_returns_empty_resources_envelope(self) -> None:
        response = mcp_transport.control_response("resources/list", 2)
        self.assertIsNotNone(response)
        body = cast("dict[str, object]", response)
        result = cast("dict[str, object]", body["result"])
        self.assertEqual(result, {"resources": []})

    def test_prompts_list_returns_empty_prompts_envelope(self) -> None:
        response = mcp_transport.control_response("prompts/list", 3)
        self.assertIsNotNone(response)
        body = cast("dict[str, object]", response)
        result = cast("dict[str, object]", body["result"])
        self.assertEqual(result, {"prompts": []})

    def test_unknown_method_returns_none_so_caller_can_dispatch_further(self) -> None:
        # control_response must not absorb unknown methods — main loop
        # depends on a None to route to tools/call or the unsupported-method
        # error.
        self.assertIsNone(mcp_transport.control_response("custom/whatever", 1))


class HandleToolCallBranchTests(unittest.TestCase):
    """Cover the protocol-level rejections in handle_tool_call."""

    def test_missing_name_is_a_protocol_type_error(self) -> None:
        # `params.name` must be a string. Missing → not isinstance(None, str).
        request: JsonObject = {"params": {"arguments": {}}}
        with self.assertRaises(McpProtocolTypeError):
            _ = mcp_transport.handle_tool_call(request, 1)

    def test_unknown_tool_name_is_a_protocol_error(self) -> None:
        request: JsonObject = {"params": {"name": "no_such_tool", "arguments": {}}}
        with self.assertRaises(McpProtocolError) as ctx:
            _ = mcp_transport.handle_tool_call(request, 1)
        self.assertIn("unknown tool", str(ctx.exception))

    def test_write_tool_call_is_rejected_when_writes_are_disabled(self) -> None:
        # Inject a temporary write-tool entry into the registry to exercise
        # the `if definition.write_tool and not db.allow_writes():` branch.
        # `validate_tool_arguments` is keyed off id(definition) into the
        # public catalog, so we also patch it to a passthrough — the
        # boundary check we're after fires after validation but before the
        # handler runs.
        def _unused_handler(_args: JsonObject) -> JsonObject:
            raise AssertionError("handler must not run when writes are disabled")

        def _passthrough_validator(_definition: ToolDefinition, arguments: JsonObject) -> JsonObject:
            return arguments

        write_def = ToolDefinition(
            description="write-only test tool",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            write_tool=True,
        )
        request: JsonObject = {"params": {"name": "_test_write_tool", "arguments": {}}}

        with (
            patch.dict(mcp_transport.TOOLS, {"_test_write_tool": (write_def, _unused_handler)}, clear=False),
            patch.object(mcp_transport, "validate_tool_arguments", side_effect=_passthrough_validator),
            patch.object(mcp_transport.db, "allow_writes", return_value=False),
            self.assertRaises(McpWritePermissionError) as ctx,
        ):
            _ = mcp_transport.handle_tool_call(request, 1)

        self.assertIn("writes are disabled", str(ctx.exception))


class HandleRequestUnsupportedMethodTests(unittest.TestCase):
    """If neither control_response nor tools/call handles a method, the
    request must surface a structured protocol error."""

    def test_unknown_string_method_raises_protocol_error(self) -> None:
        with self.assertRaises(McpProtocolError) as ctx:
            _ = mcp_transport.handle_request({"jsonrpc": "2.0", "id": 1, "method": "made/up"})
        self.assertIn("unsupported method", str(ctx.exception))

    def test_non_string_method_raises_protocol_type_error(self) -> None:
        with self.assertRaises(McpProtocolTypeError):
            _ = mcp_transport.handle_request({"jsonrpc": "2.0", "id": 1, "method": 123})


class ErrorResponseAndWriteResponseTests(unittest.TestCase):
    """Tiny formatting helpers — pin the wire shape and the stdout newline."""

    def test_error_response_envelope_matches_jsonrpc_error_shape(self) -> None:
        response = mcp_transport.error_response(7, -32000, "boom")
        self.assertEqual(
            response,
            {"jsonrpc": "2.0", "id": 7, "error": {"code": -32000, "message": "boom"}},
        )

    def test_write_response_emits_compact_json_followed_by_newline(self) -> None:
        captured = io.StringIO()
        # Replace stdout so we can observe what write_response sent.
        with patch.object(mcp_transport.sys, "stdout", captured):
            mcp_transport.write_response({"jsonrpc": "2.0", "id": 1, "result": {}})

        emitted = captured.getvalue()
        # Compact separators (no spaces) and a trailing newline — clients
        # parse newline-delimited JSON, so the newline is part of the wire
        # contract.
        self.assertTrue(emitted.endswith("\n"))
        # Round-trip the payload to verify it's still valid JSON.
        parsed = cast("dict[str, object]", json.loads(emitted))
        self.assertEqual(parsed["id"], 1)
        # No whitespace between key:value pairs.
        self.assertNotIn(", ", emitted)
        self.assertNotIn(": ", emitted)


if __name__ == "__main__":
    _ = unittest.main()
