"""Tests for the Apple Silicon MLX embedding server."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import project_code_intelligence.embedding.apple_embed_server as _server
from project_code_intelligence import config
from project_code_intelligence.embedding.apple_embed_server import (
    AppleEmbedHandler,
    apple_embed_model_name,
    apple_embed_server_is_running,
    embedding_response,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_get_handler(path: str, model_name: str = "test-model") -> MagicMock:
    handler = MagicMock(spec=AppleEmbedHandler)
    handler.path = path
    handler.wfile = io.BytesIO()
    mock_srv = MagicMock()
    mock_srv.model_name = model_name
    handler._embed_server.return_value = mock_srv
    return handler


def _make_post_handler(path: str, body: bytes, model_name: str = "test-model") -> MagicMock:
    handler = MagicMock(spec=AppleEmbedHandler)
    handler.path = path
    handler.wfile = io.BytesIO()
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    mock_srv = MagicMock()
    mock_srv.model = MagicMock()
    mock_srv.tokenizer = MagicMock()
    mock_srv.model_name = model_name
    handler._embed_server.return_value = mock_srv
    return handler


def _response_body(handler: MagicMock) -> object:
    raw = handler.wfile.getvalue()
    return json.loads(raw) if raw else None


def _status_code(handler: MagicMock) -> int:
    return handler.send_response.call_args[0][0]  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# apple_embed_model_name
# ---------------------------------------------------------------------------


class ModelNameTests(unittest.TestCase):
    def test_returns_default_when_env_not_set(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "PROJECT_CODE_INTELLIGENCE_APPLE_EMBED_MODEL"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(apple_embed_model_name(), config.DEFAULT_APPLE_EMBED_MODEL)

    def test_returns_configured_model(self) -> None:
        with patch.dict(os.environ, {"PROJECT_CODE_INTELLIGENCE_APPLE_EMBED_MODEL": "custom/model"}):
            self.assertEqual(apple_embed_model_name(), "custom/model")

    def test_falls_back_to_default_when_env_is_empty_string(self) -> None:
        with patch.dict(os.environ, {"PROJECT_CODE_INTELLIGENCE_APPLE_EMBED_MODEL": ""}):
            self.assertEqual(apple_embed_model_name(), config.DEFAULT_APPLE_EMBED_MODEL)


# ---------------------------------------------------------------------------
# _read_pid
# ---------------------------------------------------------------------------


class ReadPidTests(unittest.TestCase):
    def test_returns_none_for_missing_file(self) -> None:
        missing = Path(tempfile.gettempdir()) / "pci-test-nonexistent-pid-file-99xz.pid"
        self.assertIsNone(_server._read_pid(missing))

    def test_returns_none_for_non_integer_content(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pid", delete=False, encoding="utf-8") as f:
            f.write("not-a-number\n")
            path = Path(f.name)
        try:
            self.assertIsNone(_server._read_pid(path))
        finally:
            path.unlink(missing_ok=True)

    def test_parses_valid_pid(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pid", delete=False, encoding="utf-8") as f:
            f.write("12345\n")
            path = Path(f.name)
        try:
            self.assertEqual(_server._read_pid(path), 12345)
        finally:
            path.unlink(missing_ok=True)

    def test_strips_whitespace(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pid", delete=False, encoding="utf-8") as f:
            f.write("  99  \n")
            path = Path(f.name)
        try:
            self.assertEqual(_server._read_pid(path), 99)
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# _is_running
# ---------------------------------------------------------------------------


class IsRunningTests(unittest.TestCase):
    def test_returns_false_for_nonexistent_pid(self) -> None:
        fake_pid = os.getpid() + 99999
        try:
            os.kill(fake_pid, 0)
            self.skipTest("fake PID unexpectedly exists")
        except ProcessLookupError:
            pass
        self.assertFalse(_server._is_running(fake_pid))

    def test_returns_true_for_current_process(self) -> None:
        self.assertTrue(_server._is_running(os.getpid()))

    def test_returns_true_when_kill_raises_generic_oserror(self) -> None:
        with patch("os.kill", side_effect=OSError("permission denied")):
            self.assertTrue(_server._is_running(1))


# ---------------------------------------------------------------------------
# apple_embed_server_is_running (tests _read_pid / _is_running integration)
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

    def test_live_pid_file_with_current_process(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pid", delete=False, encoding="utf-8") as f:
            f.write(f"{os.getpid()}\n")
            pid_path = Path(f.name)
        try:
            with patch.object(_server, "APPLE_EMBED_SERVER_PID_FILE", pid_path):
                self.assertTrue(apple_embed_server_is_running())
        finally:
            pid_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# _mlx_clear_cache
# ---------------------------------------------------------------------------


class ClearCacheTests(unittest.TestCase):
    def test_calls_clear_cache_when_present(self) -> None:
        mock_clear = MagicMock()
        mock_mx = MagicMock()
        mock_mx.clear_cache = mock_clear
        with patch.object(_server, "import_module", return_value=mock_mx):
            _server._mlx_clear_cache()
        self.assertEqual(mock_clear.call_count, 1)

    def test_skips_gracefully_when_clear_cache_absent(self) -> None:
        mock_mx = MagicMock(spec=[])
        with patch.object(_server, "import_module", return_value=mock_mx):
            _server._mlx_clear_cache()
        self.assertIsNone(None)  # passes if no exception raised

    def test_absorbs_runtime_error(self) -> None:
        mock_mx = MagicMock()
        mock_mx.clear_cache = MagicMock(side_effect=RuntimeError("metal error"))
        with patch.object(_server, "import_module", return_value=mock_mx):
            _server._mlx_clear_cache()
        self.assertIsNone(None)  # passes if no exception raised


# ---------------------------------------------------------------------------
# embedding_response
# ---------------------------------------------------------------------------


class EmbeddingResponseTests(unittest.TestCase):
    @staticmethod
    def _call(
        texts: list[str],
        vectors: list[list[float]],
        model_name: str = "my-model",
    ) -> object:
        with (
            patch.object(_server, "_embed", return_value=vectors),
            patch.object(_server, "_mlx_clear_cache"),
        ):
            return embedding_response(MagicMock(), MagicMock(), model_name, {"input": texts})

    def test_returns_openai_compatible_structure(self) -> None:
        result = self._call(["hello"], [[0.1, 0.2, 0.3]])
        self.assertEqual(result["object"], "list")  # type: ignore[index]
        self.assertEqual(result["model"], "my-model")  # type: ignore[index]
        data = result["data"]  # type: ignore[index]
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["object"], "embedding")
        self.assertEqual(data[0]["index"], 0)
        self.assertEqual(data[0]["embedding"], [0.1, 0.2, 0.3])

    def test_indexes_multiple_embeddings(self) -> None:
        result = self._call(["a", "b", "c"], [[1.0], [2.0], [3.0]])
        data = result["data"]  # type: ignore[index]
        self.assertEqual([d["index"] for d in data], [0, 1, 2])

    def test_usage_contains_prompt_tokens(self) -> None:
        result = self._call(["hello world"], [[0.0]])
        usage = result["usage"]  # type: ignore[index]
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
            embedding_response(MagicMock(), MagicMock(), "m", {"input": ["a", "b"]})

    def test_clears_cache_even_when_embed_raises(self) -> None:
        mock_clear = MagicMock()
        with (
            patch.object(_server, "_embed", side_effect=RuntimeError("OOM")),
            patch.object(_server, "_mlx_clear_cache", mock_clear),
            self.assertRaises(RuntimeError),
        ):
            embedding_response(MagicMock(), MagicMock(), "m", {"input": "hello"})
        self.assertEqual(mock_clear.call_count, 1)

    def test_rejects_missing_input_field(self) -> None:
        with (
            patch.object(_server, "_embed"),
            patch.object(_server, "_mlx_clear_cache"),
            self.assertRaises(TypeError),
        ):
            embedding_response(MagicMock(), MagicMock(), "m", {})


# ---------------------------------------------------------------------------
# AppleEmbedHandler — GET
# ---------------------------------------------------------------------------


class AppleEmbedHandlerGetTests(unittest.TestCase):
    def test_healthz_returns_200_with_model_name(self) -> None:
        handler = _make_get_handler("/healthz", model_name="qwen3-0.6b")
        AppleEmbedHandler.do_GET(handler)
        self.assertEqual(_status_code(handler), 200)
        self.assertEqual(_response_body(handler), {"ok": True, "model": "qwen3-0.6b"})

    def test_unknown_path_returns_404(self) -> None:
        handler = _make_get_handler("/unknown")
        AppleEmbedHandler.do_GET(handler)
        self.assertEqual(_status_code(handler), 404)

    def test_root_returns_404(self) -> None:
        handler = _make_get_handler("/")
        AppleEmbedHandler.do_GET(handler)
        self.assertEqual(_status_code(handler), 404)


# ---------------------------------------------------------------------------
# AppleEmbedHandler — POST
# ---------------------------------------------------------------------------


class AppleEmbedHandlerPostTests(unittest.TestCase):
    def test_wrong_path_returns_404(self) -> None:
        body = json.dumps({"input": "hello"}).encode()
        handler = _make_post_handler("/wrong", body)
        AppleEmbedHandler.do_POST(handler)
        self.assertEqual(_status_code(handler), 404)

    def test_missing_content_length_returns_400(self) -> None:
        handler = _make_post_handler("/v1/embeddings", b'{"input":"x"}')
        handler.headers = {}
        AppleEmbedHandler.do_POST(handler)
        self.assertEqual(_status_code(handler), 400)

    def test_invalid_json_body_returns_400(self) -> None:
        handler = _make_post_handler("/v1/embeddings", b"not json")
        AppleEmbedHandler.do_POST(handler)
        self.assertEqual(_status_code(handler), 400)

    def test_empty_input_array_returns_400(self) -> None:
        body = json.dumps({"input": []}).encode()
        handler = _make_post_handler("/v1/embeddings", body)
        with patch.object(_server, "_embed"), patch.object(_server, "_mlx_clear_cache"):
            AppleEmbedHandler.do_POST(handler)
        self.assertEqual(_status_code(handler), 400)

    def test_embed_value_error_returns_400(self) -> None:
        body = json.dumps({"input": "hello"}).encode()
        handler = _make_post_handler("/v1/embeddings", body)
        with (
            patch.object(_server, "_embed", side_effect=ValueError("bad vectors")),
            patch.object(_server, "_mlx_clear_cache"),
        ):
            AppleEmbedHandler.do_POST(handler)
        self.assertEqual(_status_code(handler), 400)
        self.assertIn("bad vectors", _response_body(handler)["error"]["message"])  # type: ignore[index]

    def test_embed_runtime_error_returns_400(self) -> None:
        body = json.dumps({"input": "hello"}).encode()
        handler = _make_post_handler("/v1/embeddings", body)
        with (
            patch.object(_server, "_embed", side_effect=RuntimeError("MLX OOM")),
            patch.object(_server, "_mlx_clear_cache"),
        ):
            AppleEmbedHandler.do_POST(handler)
        self.assertEqual(_status_code(handler), 400)
        self.assertIn("MLX OOM", _response_body(handler)["error"]["message"])  # type: ignore[index]

    def test_success_returns_200_with_embedding_payload(self) -> None:
        body = json.dumps({"input": ["hello", "world"]}).encode()
        handler = _make_post_handler("/v1/embeddings", body, model_name="qwen3")
        with (
            patch.object(_server, "_embed", return_value=[[0.1, 0.2], [0.3, 0.4]]),
            patch.object(_server, "_mlx_clear_cache"),
        ):
            AppleEmbedHandler.do_POST(handler)
        self.assertEqual(_status_code(handler), 200)
        payload = _response_body(handler)
        self.assertEqual(payload["object"], "list")  # type: ignore[index]
        self.assertEqual(payload["model"], "qwen3")  # type: ignore[index]
        self.assertEqual(len(payload["data"]), 2)  # type: ignore[index]


# ---------------------------------------------------------------------------
# _load_model — tokenizer extraction
# ---------------------------------------------------------------------------


class LoadModelTokenizerTests(unittest.TestCase):
    def test_extracts_hf_tokenizer_from_wrapper(self) -> None:
        hf_tok = MagicMock(name="HFTokenizer")
        wrapper = MagicMock()
        wrapper._tokenizer = hf_tok

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (MagicMock(), wrapper)

        with patch.object(_server, "import_module", return_value=mock_mlx_lm):
            _, tokenizer = _server._load_model("some/model")

        self.assertIs(tokenizer, hf_tok)

    def test_falls_back_to_auto_tokenizer_when_wrapper_has_no_tokenizer(self) -> None:
        wrapper = MagicMock(spec=[])  # no _tokenizer attribute
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (MagicMock(), wrapper)

        fallback_tok = MagicMock(name="FallbackTokenizer")
        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = fallback_tok

        call_count = 0

        def _import(_name: str) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_mlx_lm
            return mock_transformers

        with patch.object(_server, "import_module", side_effect=_import):
            _, tokenizer = _server._load_model("some/model")

        self.assertIs(tokenizer, fallback_tok)

    def test_raises_when_mlx_lm_not_installed(self) -> None:
        with (
            patch.object(_server, "import_module", side_effect=ImportError("no mlx_lm")),
            self.assertRaises(RuntimeError),
        ):
            _server._load_model("any/model")


if __name__ == "__main__":
    _ = unittest.main()
