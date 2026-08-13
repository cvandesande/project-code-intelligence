"""Daemonized Apple Silicon embedding server backed by MLX.

Loads a sentence-transformers model via mlx_lm (native MLX, 8-bit quantized)
and serves an OpenAI-compatible /v1/embeddings endpoint.  Uses last-token
pooling as required by Qwen3-Embedding and similar decoder-only models.
Writes a PID file so pci-doctor --stop can terminate the server.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from project_code_intelligence.models import JsonObject


from project_code_intelligence import config, process
from project_code_intelligence.embedding.http_common import (
    json_error,
    normalize_input,
    write_json,
)
from project_code_intelligence.embedding.http_common import parse_json_body as _parse_json_body
from project_code_intelligence.process import PopenOptions
from project_code_intelligence.runtime import estimate_embedding_tokens

_mlx_lm_module: object | None
_mlx_lm_import_error: ImportError | None
if not TYPE_CHECKING and sys.platform == "darwin":
    try:
        import mlx_lm as _imported_mlx_lm
    except ImportError as exc:
        _mlx_lm_module = None
        _mlx_lm_import_error = exc
    else:
        _mlx_lm_module = _imported_mlx_lm
        _mlx_lm_import_error = None
else:
    _mlx_lm_module = None
    _mlx_lm_import_error = None

_STATE_DIR = Path.home() / ".local" / "state" / "project-code-intelligence"
APPLE_EMBED_SERVER_PID_FILE = _STATE_DIR / "pci-apple-embed-server.pid"
APPLE_EMBED_SERVER_LOG_FILE = _STATE_DIR / "pci-apple-embed-server.log"


def _err(msg: str) -> None:
    _ = sys.stderr.write(msg + "\n")


def _read_pid(pid_path: Path) -> int | None:
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    else:
        return True


def apple_embed_server_is_running() -> bool:
    """Return True if a pci apple-embed-server daemon is currently running."""
    pid = _read_pid(APPLE_EMBED_SERVER_PID_FILE)
    return pid is not None and _is_running(pid)


def apple_embed_model_name() -> str:
    return (
        config.env_text("PCI_APPLE_EMBED_MODEL", config.DEFAULT_APPLE_EMBED_MODEL) or config.DEFAULT_APPLE_EMBED_MODEL
    )


# ---------------------------------------------------------------------------
# MLX inference helpers
# ---------------------------------------------------------------------------


class _MlxArray(Protocol):
    """Minimal structural type for MLX arrays used in embedding inference."""

    def __getitem__(self, key: object) -> _MlxArray: ...
    def __sub__(self, _other: object) -> _MlxArray: ...
    def __truediv__(self, _other: object) -> _MlxArray: ...
    def __itruediv__(self, _other: object) -> _MlxArray: ...
    def __pow__(self, _other: int | float) -> _MlxArray: ...
    def sum(self, *args: object, **kwargs: object) -> _MlxArray: ...
    def tolist(self) -> list[object]: ...


class _MlxLmModel(Protocol):
    """Structural type for the mlx_lm model object."""

    model: _MlxArray  # transformer backbone (callable as __call__)


class _AutoTokenizerClass(Protocol):
    """Minimal structural type for transformers.AutoTokenizer."""

    def from_pretrained(self, _pretrained_model_name_or_path: str) -> object: ...


def _load_model(model_name: str) -> tuple[_MlxLmModel, object]:
    """Load an MLX model and tokenizer via mlx_lm. Returns (model, HF tokenizer)."""
    if _mlx_lm_module is None:
        raise RuntimeError(
            "mlx_lm is not installed. Run: uv tool install --reinstall project-code-intelligence"
        ) from _mlx_lm_import_error
    load_value = cast("object", getattr(_mlx_lm_module, "load", None))
    if not callable(load_value):
        raise TypeError("mlx_lm.load was not available")
    load_fn = cast("Callable[[str], tuple[_MlxLmModel, object]]", load_value)
    model, wrapper = load_fn(model_name)
    # mlx_lm returns a TokenizerWrapper designed for text generation; we need the
    # underlying HuggingFace tokenizer to call it with padding=True / truncation=True.
    # _tokenizer is the canonical private attribute; fall back to transformers if it
    # ever disappears in a future mlx_lm release.
    hf_tokenizer: object = getattr(wrapper, "_tokenizer", None)
    if hf_tokenizer is None:
        auto_tokenizer = cast("_AutoTokenizerClass", import_module("transformers").AutoTokenizer)
        hf_tokenizer = auto_tokenizer.from_pretrained(model_name)
    _err(f"  Backend: MLX (Apple Silicon) — model: {model_name}")
    return model, hf_tokenizer


def _mlx_clear_cache() -> None:
    try:
        mx = import_module("mlx.core")
        clear_cache = getattr(mx, "clear_cache", None)
        if callable(clear_cache):
            _ = clear_cache()
    except RuntimeError as exc:
        _err(f"mx.clear_cache failed: {exc}")


def _embed(model: _MlxLmModel, tokenizer: object, texts: list[str]) -> list[list[float]]:
    """Run MLX inference and return L2-normalised embeddings (last-token pooling)."""
    mx = import_module("mlx.core")

    tokenize_fn = cast("Callable[..., Mapping[str, object]]", tokenizer)
    encoded = tokenize_fn(texts, padding=True, truncation=True, return_tensors="np")

    to_array = cast("Callable[[object], _MlxArray]", mx.array)
    input_ids = to_array(encoded["input_ids"])
    attention_mask = to_array(encoded["attention_mask"])

    backbone = cast("Callable[[_MlxArray], object]", model.model)
    hidden_out = backbone(input_ids)
    hidden = cast("_MlxArray", hidden_out[0] if isinstance(hidden_out, tuple) else hidden_out)

    # Last-token pooling: index of the final non-padding token per sequence.
    # The backbone uses causal attention and the tokenizer uses right-padding, so
    # padding tokens are always to the right of the last real token and are masked
    # out by the causal mask — the last real token's hidden state is unaffected.
    lengths: _MlxArray = attention_mask.sum(axis=1) - 1
    arange_fn = cast("Callable[[int], _MlxArray]", mx.arange)
    vecs: _MlxArray = hidden[arange_fn(len(texts)), lengths]

    # L2 normalise.
    sqrt_fn = cast("Callable[[_MlxArray], _MlxArray]", mx.sqrt)
    norms = sqrt_fn((vecs**2).sum(axis=-1, keepdims=True))
    vecs /= norms

    eval_fn = cast("Callable[..., None]", mx.eval)
    eval_fn(vecs)

    return [cast("list[float]", row) for row in vecs.tolist()]


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


def _request_max_bytes() -> int:
    return config.env_int("PCI_APPLE_EMBED_MAX_REQUEST_BYTES", 4 * 1024 * 1024, minimum=1024)


class AppleEmbedHTTPServer(HTTPServer):
    model: _MlxLmModel
    tokenizer: object
    model_name: str

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        model: _MlxLmModel,
        tokenizer: object,
        model_name: str,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name


def embedding_response(model: _MlxLmModel, tokenizer: object, model_name: str, request: JsonObject) -> JsonObject:
    texts = normalize_input(request.get("input"))
    try:
        vectors = _embed(model, tokenizer, texts)
    finally:
        _mlx_clear_cache()
    if len(vectors) != len(texts):
        raise ValueError("model returned an unexpected number of vectors")
    prompt_tokens = sum(estimate_embedding_tokens(t) for t in texts)
    return {
        "object": "list",
        "model": model_name,
        "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    }


def _parse_request(handler: BaseHTTPRequestHandler) -> JsonObject:
    return _parse_json_body(handler, max_bytes=_request_max_bytes())


class AppleEmbedHandler(BaseHTTPRequestHandler):
    server_version = "project-code-intelligence-apple-embed/0.1"

    def _embed_server(self) -> AppleEmbedHTTPServer:
        return cast("AppleEmbedHTTPServer", self.server)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            write_json(
                self,
                200,
                {"ok": True, "model": self._embed_server().model_name, "framework": "Apple MLX"},
            )
            return
        write_json(self, 404, json_error("not found"))

    def do_POST(self) -> None:
        if self.path != "/v1/embeddings":
            write_json(self, 404, json_error("not found"))
            return
        server = self._embed_server()
        try:
            request = _parse_request(self)
            payload = embedding_response(server.model, server.tokenizer, server.model_name, request)
        except (UnicodeDecodeError, json.JSONDecodeError, RuntimeError, TypeError, ValueError) as exc:
            write_json(self, 400, json_error(str(exc)))
            return
        write_json(self, 200, payload)


# ---------------------------------------------------------------------------
# Serve entry point (runs in the daemon process)
# ---------------------------------------------------------------------------


def _serve() -> None:
    model_name = apple_embed_model_name()
    host = config.env_text("PCI_EMBEDDING_HOST", "127.0.0.1") or "127.0.0.1"
    port = config.env_int("PCI_EMBEDDING_PORT", 18081, minimum=1)

    _err(f"Loading Apple embed model: {model_name}")
    model, tokenizer = _load_model(model_name)

    server = AppleEmbedHTTPServer(
        (host, port), AppleEmbedHandler, model=model, tokenizer=tokenizer, model_name=model_name
    )
    _err(f"pci apple-embed-server ready on http://{host}:{port}")
    _ = sys.stderr.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# Launcher — daemonizes into _serve
# ---------------------------------------------------------------------------


def main() -> None:
    if "--serve" in sys.argv:
        _serve()
        return

    pid = _read_pid(APPLE_EMBED_SERVER_PID_FILE)
    if pid is not None and _is_running(pid):
        _err(f"pci apple-embed-server already running (PID {pid}).")
        return

    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    with APPLE_EMBED_SERVER_LOG_FILE.open("a") as log_file:
        proc = process.popen(
            [sys.executable, "-m", "project_code_intelligence.embedding.apple_embed_server", "--serve"],
            PopenOptions(
                stdout=log_file,
                stderr=log_file,
                stdin=process.DEVNULL,
                start_new_session=True,
            ),
        )
    _ = APPLE_EMBED_SERVER_PID_FILE.write_text(str(proc.pid) + "\n")
    _err(f"pci apple-embed-server started (PID {proc.pid}). Log: {APPLE_EMBED_SERVER_LOG_FILE}")


if __name__ == "__main__":
    main()
