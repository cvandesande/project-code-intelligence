"""Tests for shared embedding HTTP helpers."""

from __future__ import annotations

import io
import json
import unittest
from http.server import BaseHTTPRequestHandler
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

from project_code_intelligence.embedding.http_common import (
    json_error,
    normalize_input,
    parse_json_body,
    write_json,
)


class NormalizeInputTests(unittest.TestCase):
    def test_single_string(self) -> None:
        self.assertEqual(normalize_input("hello"), ["hello"])

    def test_list_of_strings(self) -> None:
        self.assertEqual(normalize_input(["a", "b"]), ["a", "b"])

    def test_rejects_empty_list(self) -> None:
        with self.assertRaises(ValueError):
            _ = normalize_input([])

    def test_rejects_non_string_items(self) -> None:
        with self.assertRaises(TypeError):
            _ = normalize_input(["ok", 123])

    def test_rejects_integer(self) -> None:
        with self.assertRaises(TypeError):
            _ = normalize_input(42)

    def test_rejects_none(self) -> None:
        with self.assertRaises(TypeError):
            _ = normalize_input(None)


class JsonErrorTests(unittest.TestCase):
    def test_structure(self) -> None:
        result = json_error("bad request")
        self.assertEqual(result, {"error": {"message": "bad request", "type": "invalid_request_error"}})


class WriteJsonTests(unittest.TestCase):
    def test_writes_json_response(self) -> None:
        wfile = io.BytesIO()
        handler = cast("BaseHTTPRequestHandler", MagicMock(spec=BaseHTTPRequestHandler))
        handler.wfile = wfile
        payload: JsonObject = {"ok": True}
        write_json(handler, 200, payload)
        written = wfile.getvalue()
        self.assertEqual(json.loads(written), {"ok": True})


class ParseJsonBodyTests(unittest.TestCase):
    @staticmethod
    def _make_handler(body: bytes, content_length: str | None = None) -> BaseHTTPRequestHandler:
        handler = cast("BaseHTTPRequestHandler", MagicMock(spec=BaseHTTPRequestHandler))
        handler.rfile = io.BytesIO(body)
        if content_length is not None:
            handler.headers = {"Content-Length": content_length}  # type: ignore[assignment]  # test double
        else:
            handler.headers = {}  # type: ignore[assignment]  # test double
        return handler

    def test_parses_valid_json(self) -> None:
        body = json.dumps({"input": "hello"}).encode()
        handler = self._make_handler(body, str(len(body)))
        result = parse_json_body(handler, max_bytes=4096)
        self.assertEqual(result, {"input": "hello"})

    def test_rejects_missing_content_length(self) -> None:
        handler = self._make_handler(b"{}")
        with self.assertRaises(ValueError):
            _ = parse_json_body(handler, max_bytes=4096)

    def test_rejects_zero_length(self) -> None:
        handler = self._make_handler(b"", "0")
        with self.assertRaises(ValueError):
            _ = parse_json_body(handler, max_bytes=4096)

    def test_rejects_oversized_body(self) -> None:
        handler = self._make_handler(b'{"x": 1}', "9999")
        with self.assertRaises(ValueError):
            _ = parse_json_body(handler, max_bytes=100)

    def test_rejects_non_object(self) -> None:
        body = json.dumps([1, 2, 3]).encode()
        handler = self._make_handler(body, str(len(body)))
        with self.assertRaises(TypeError):
            _ = parse_json_body(handler, max_bytes=4096)


if __name__ == "__main__":
    _ = unittest.main()
