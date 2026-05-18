"""Tests for the Apple Silicon MLX embedding server."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

from typing_extensions import override

import project_code_intelligence.embedding.apple_embed_server as _server
from project_code_intelligence import config
from project_code_intelligence.embedding.apple_embed_server import (
    AppleEmbedHandler,
    AppleEmbedHTTPServer,
    apple_embed_model_name,
    apple_embed_server_is_running,
    embedding_response,
)

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject


# ---------------------------------------------------------------------------
# Typed test double for AppleEmbedHandler (avoids MagicMock Any issues)
# ---------------------------------------------------------------------------


class _TestHandler(AppleEmbedHandler):
    """Subclass that captures HTTP responses without a real socket."""

    _captured_status: int
    _wfile: io.BytesIO
    _mock_model_name: str

    def __init__(
        self,
        path: str,
        body: bytes = b"",
        model_name: str = "test-model",
    ) -> None:
        # Deliberately skip super().__init__ — it requires a real socket.
        self.path = path
        self._wfile = io.BytesIO()
        self.wfile = self._wfile
        self.rfile = io.BytesIO(body)
        # `BaseHTTPRequestHandler.headers` is `email.message.Message` (HTTP
        # headers reuse the stdlib email-RFC parser). Build a real Message
        # so the test stub matches the production type — `parse_json_body`
        # only reads `.get("Content-Length")`, which Message implements.
        headers = Message()
        if body:
            headers["Content-Length"] = str(len(body))
        self.headers = headers
        self._mock_model_name = model_name
        self._captured_status = 0

    @override
    def send_response(self, code: int, message: str | None = None) -> None:
        self._captured_status = code

    @override
    def send_header(self, keyword: str, value: str) -> None:
        if not keyword or not value:
            return

    @override
    def end_headers(self) -> None:
        pass

    @override
    def log_message(self, format: str, *args: object) -> None:
        pass

    @override
    def _embed_server(self) -> AppleEmbedHTTPServer:
        srv = cast("AppleEmbedHTTPServer", MagicMock())
        srv.model_name = self._mock_model_name
        return srv

    @property
    def captured_status(self) -> int:
        return self._captured_status

    @property
    def response_body(self) -> dict[str, object] | None:
        data = self._wfile.getvalue()
        if not data:
            return None
        parsed = cast("object", json.loads(data))
        if not isinstance(parsed, dict):
            raise TypeError("response body is not a JSON object")
        return cast("dict[str, object]", parsed)


# ---------------------------------------------------------------------------
# apple_embed_model_name
# ---------------------------------------------------------------------------


class ModelNameTests(unittest.TestCase):
    def test_returns_default_when_env_not_set(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "PCI_APPLE_EMBED_MODEL"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(apple_embed_model_name(), config.DEFAULT_APPLE_EMBED_MODEL)

    def test_returns_configured_model(self) -> None:
        with patch.dict(os.environ, {"PCI_APPLE_EMBED_MODEL": "custom/model"}):
            self.assertEqual(apple_embed_model_name(), "custom/model")

    def test_falls_back_to_default_when_env_is_empty_string(self) -> None:
        with patch.dict(os.environ, {"PCI_APPLE_EMBED_MODEL": ""}):
            self.assertEqual(apple_embed_model_name(), config.DEFAULT_APPLE_EMBED_MODEL)


# ---------------------------------------------------------------------------
# apple_embed_server_is_running — exercises PID file reading and process check
# ---------------------------------------------------------------------------


class ServerIsRunningTests(unittest.TestCase):
    def test_returns_false_when_no_pid_file(self) -> None:
        with patch.object(_server, "_read_pid", return_value=None):
            self.assertFalse(apple_embed_server_is_running())

    def test_returns_false_when_pid_is_dead(self) -> None:
        with (
            patch.object(_server, "_read_pid", return_value=99999),
            patch.object(_server, "_is_running", return_value=False),
        ):
            self.assertFalse(apple_embed_server_is_running())

    def test_returns_true_when_pid_is_alive(self) -> None:
        with (
            patch.object(_server, "_read_pid", return_value=12345),
            patch.object(_server, "_is_running", return_value=True),
        ):
            self.assertTrue(apple_embed_server_is_running())

    def test_pid_file_with_current_process_returns_true(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pid", delete=False, encoding="utf-8") as f:
            _ = f.write(f"{os.getpid()}\n")
            pid_path = Path(f.name)
        try:
            with patch.object(_server, "APPLE_EMBED_SERVER_PID_FILE", pid_path):
                self.assertTrue(apple_embed_server_is_running())
        finally:
            pid_path.unlink(missing_ok=True)

    def test_pid_file_with_invalid_content_returns_false(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pid", delete=False, encoding="utf-8") as f:
            _ = f.write("not-a-number\n")
            pid_path = Path(f.name)
        try:
            with patch.object(_server, "APPLE_EMBED_SERVER_PID_FILE", pid_path):
                self.assertFalse(apple_embed_server_is_running())
        finally:
            pid_path.unlink(missing_ok=True)

    def test_missing_pid_file_returns_false(self) -> None:
        missing = Path(tempfile.gettempdir()) / "pci-test-nonexistent-pid-file-99xz.pid"
        with patch.object(_server, "APPLE_EMBED_SERVER_PID_FILE", missing):
            self.assertFalse(apple_embed_server_is_running())

    def test_dead_pid_in_file_returns_false(self) -> None:
        fake_pid = os.getpid() + 99999
        try:
            os.kill(fake_pid, 0)
            self.skipTest("fake PID unexpectedly alive")
        except ProcessLookupError:
            pass
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pid", delete=False, encoding="utf-8") as f:
            _ = f.write(f"{fake_pid}\n")
            pid_path = Path(f.name)
        try:
            with patch.object(_server, "APPLE_EMBED_SERVER_PID_FILE", pid_path):
                self.assertFalse(apple_embed_server_is_running())
        finally:
            pid_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# embedding_response
# ---------------------------------------------------------------------------


class EmbeddingResponseTests(unittest.TestCase):
    @staticmethod
    def _call(
        texts: list[str],
        vectors: list[list[float]],
        model_name: str = "my-model",
    ) -> JsonObject:
        with (
            patch.object(_server, "_embed", return_value=vectors),
            patch.object(_server, "_mlx_clear_cache"),
        ):
            return embedding_response(MagicMock(), MagicMock(), model_name, {"input": texts})

    def test_returns_openai_compatible_structure(self) -> None:
        result = self._call(["hello"], [[0.1, 0.2, 0.3]])
        self.assertEqual(result["object"], "list")
        self.assertEqual(result["model"], "my-model")
        data = cast("list[dict[str, object]]", result["data"])
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["object"], "embedding")
        self.assertEqual(data[0]["index"], 0)
        self.assertEqual(data[0]["embedding"], [0.1, 0.2, 0.3])

    def test_indexes_multiple_embeddings(self) -> None:
        result = self._call(["a", "b", "c"], [[1.0], [2.0], [3.0]])
        data = cast("list[dict[str, object]]", result["data"])
        self.assertEqual([d["index"] for d in data], [0, 1, 2])

    def test_usage_contains_prompt_tokens(self) -> None:
        result = self._call(["hello world"], [[0.0]])
        usage = cast("dict[str, int]", result["usage"])
        self.assertIn("prompt_tokens", usage)
        self.assertIn("total_tokens", usage)
        self.assertEqual(usage["prompt_tokens"], usage["total_tokens"])
        self.assertGreater(usage["prompt_tokens"], 0)

    def test_raises_on_vector_count_mismatch(self) -> None:
        with (
            patch.object(_server, "_embed", return_value=[[0.1]]),
            patch.object(_server, "_mlx_clear_cache"),
            self.assertRaises(ValueError),
        ):
            _ = embedding_response(MagicMock(), MagicMock(), "m", {"input": ["a", "b"]})

    def test_clears_cache_even_when_embed_raises(self) -> None:
        mock_clear = MagicMock()
        with (
            patch.object(_server, "_embed", side_effect=RuntimeError("OOM")),
            patch.object(_server, "_mlx_clear_cache", mock_clear),
            self.assertRaises(RuntimeError),
        ):
            _ = embedding_response(MagicMock(), MagicMock(), "m", {"input": "hello"})
        self.assertEqual(mock_clear.call_count, 1)

    def test_rejects_missing_input_field(self) -> None:
        with (
            patch.object(_server, "_embed"),
            patch.object(_server, "_mlx_clear_cache"),
            self.assertRaises(TypeError),
        ):
            _ = embedding_response(MagicMock(), MagicMock(), "m", {})

    def test_cache_is_cleared_on_success(self) -> None:
        mock_clear = MagicMock()
        with (
            patch.object(_server, "_embed", return_value=[[0.1]]),
            patch.object(_server, "_mlx_clear_cache", mock_clear),
        ):
            _ = embedding_response(MagicMock(), MagicMock(), "m", {"input": "hello"})
        self.assertEqual(mock_clear.call_count, 1)


# ---------------------------------------------------------------------------
# AppleEmbedHandler — GET
# ---------------------------------------------------------------------------


class AppleEmbedHandlerGetTests(unittest.TestCase):
    def test_healthz_returns_200_with_model_name(self) -> None:
        handler = _TestHandler("/healthz", model_name="qwen3-0.6b")
        handler.do_GET()
        self.assertEqual(handler.captured_status, 200)
        self.assertEqual(
            handler.response_body,
            {"ok": True, "model": "qwen3-0.6b", "framework": "Apple MLX"},
        )

    def test_healthz_advertises_apple_mlx_framework(self) -> None:
        handler = _TestHandler("/healthz", model_name="qwen3-0.6b")
        handler.do_GET()
        body = handler.response_body
        self.assertIsInstance(body, dict)
        self.assertEqual(cast("dict[str, object]", body).get("framework"), "Apple MLX")

    def test_unknown_path_returns_404(self) -> None:
        handler = _TestHandler("/unknown")
        handler.do_GET()
        self.assertEqual(handler.captured_status, 404)

    def test_root_returns_404(self) -> None:
        handler = _TestHandler("/")
        handler.do_GET()
        self.assertEqual(handler.captured_status, 404)


# ---------------------------------------------------------------------------
# AppleEmbedHandler — POST
# ---------------------------------------------------------------------------


class AppleEmbedHandlerPostTests(unittest.TestCase):
    def test_wrong_path_returns_404(self) -> None:
        body = json.dumps({"input": "hello"}).encode()
        handler = _TestHandler("/wrong", body)
        handler.do_POST()
        self.assertEqual(handler.captured_status, 404)

    def test_missing_content_length_returns_400(self) -> None:
        handler = _TestHandler("/v1/embeddings", b'{"input":"x"}')
        handler.headers = Message()
        handler.do_POST()
        self.assertEqual(handler.captured_status, 400)

    def test_invalid_json_body_returns_400(self) -> None:
        handler = _TestHandler("/v1/embeddings", b"not json")
        handler.do_POST()
        self.assertEqual(handler.captured_status, 400)

    def test_empty_input_array_returns_400(self) -> None:
        body = json.dumps({"input": []}).encode()
        handler = _TestHandler("/v1/embeddings", body)
        with patch.object(_server, "_embed"), patch.object(_server, "_mlx_clear_cache"):
            handler.do_POST()
        self.assertEqual(handler.captured_status, 400)

    def test_embed_value_error_returns_400(self) -> None:
        body = json.dumps({"input": "hello"}).encode()
        handler = _TestHandler("/v1/embeddings", body)
        with (
            patch.object(_server, "_embed", side_effect=ValueError("bad vectors")),
            patch.object(_server, "_mlx_clear_cache"),
        ):
            handler.do_POST()
        self.assertEqual(handler.captured_status, 400)
        body_obj = handler.response_body
        if body_obj is None:
            self.fail("expected response body, got None")
        error = cast("dict[str, str]", body_obj["error"])
        self.assertIn("bad vectors", error["message"])

    def test_embed_runtime_error_returns_400(self) -> None:
        body = json.dumps({"input": "hello"}).encode()
        handler = _TestHandler("/v1/embeddings", body)
        with (
            patch.object(_server, "_embed", side_effect=RuntimeError("MLX OOM")),
            patch.object(_server, "_mlx_clear_cache"),
        ):
            handler.do_POST()
        self.assertEqual(handler.captured_status, 400)
        body_obj = handler.response_body
        if body_obj is None:
            self.fail("expected response body, got None")
        error = cast("dict[str, str]", body_obj["error"])
        self.assertIn("MLX OOM", error["message"])

    def test_success_returns_200_with_embedding_payload(self) -> None:
        body = json.dumps({"input": ["hello", "world"]}).encode()
        handler = _TestHandler("/v1/embeddings", body, model_name="qwen3")
        with (
            patch.object(_server, "_embed", return_value=[[0.1, 0.2], [0.3, 0.4]]),
            patch.object(_server, "_mlx_clear_cache"),
        ):
            handler.do_POST()
        self.assertEqual(handler.captured_status, 200)
        payload = handler.response_body
        if payload is None:
            self.fail("expected response body, got None")
        self.assertEqual(payload["object"], "list")
        self.assertEqual(payload["model"], "qwen3")
        self.assertEqual(len(cast("list[object]", payload["data"])), 2)


if __name__ == "__main__":
    _ = unittest.main()
