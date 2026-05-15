"""Tests for the Fastembed embedding server's HTTP handler."""

from __future__ import annotations

import io
import json
import unittest
from typing import cast
from unittest.mock import MagicMock

from typing_extensions import override

from project_code_intelligence.embedding.fastembed_server import (
    FastEmbedHandler,
    FastEmbedHTTPServer,
)


class _TestHandler(FastEmbedHandler):
    """Subclass that captures HTTP responses without a real socket."""

    _captured_status: int
    _wfile: io.BytesIO
    _mock_model_name: str

    def __init__(self, path: str, model_name: str = "test-model") -> None:
        self.path = path
        self._wfile = io.BytesIO()
        self.wfile = self._wfile
        self.rfile = io.BytesIO(b"")
        self.headers = {}  # type: ignore[assignment]
        self._mock_model_name = model_name
        self._captured_status = 0

    @override
    def send_response(self, code: int, message: str | None = None) -> None:
        self._captured_status = code

    @override
    def send_header(self, keyword: str, value: str) -> None:
        pass

    @override
    def end_headers(self) -> None:
        pass

    @override
    def log_message(self, format: str, *args: object) -> None:
        pass

    @override
    def fastembed_server(self) -> FastEmbedHTTPServer:
        srv = cast("FastEmbedHTTPServer", MagicMock())
        srv.model_name = self._mock_model_name
        return srv

    @property
    def captured_status(self) -> int:
        return self._captured_status

    @property
    def response_body(self) -> object:
        data = self._wfile.getvalue()
        return cast("object", json.loads(data)) if data else None


class FastEmbedHandlerGetTests(unittest.TestCase):
    def test_healthz_returns_200_with_model_and_framework(self) -> None:
        handler = _TestHandler("/healthz", model_name="jinaai/jina-embeddings-v2-base-code")
        handler.do_GET()
        self.assertEqual(handler.captured_status, 200)
        self.assertEqual(
            handler.response_body,
            {
                "ok": True,
                "model": "jinaai/jina-embeddings-v2-base-code",
                "framework": "Fastembed CPU",
            },
        )

    def test_healthz_advertises_fastembed_cpu_framework(self) -> None:
        handler = _TestHandler("/healthz")
        handler.do_GET()
        body = handler.response_body
        self.assertIsInstance(body, dict)
        self.assertEqual(cast("dict[str, object]", body).get("framework"), "Fastembed CPU")

    def test_unknown_path_returns_404(self) -> None:
        handler = _TestHandler("/unknown")
        handler.do_GET()
        self.assertEqual(handler.captured_status, 404)
