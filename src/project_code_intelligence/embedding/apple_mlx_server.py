"""Start a daemonized MLX embedding server on Apple Silicon.

Loads a sentence-transformers model (optionally with the MLX backend when
the ``mlx`` package is installed) and serves an OpenAI-compatible
/v1/embeddings endpoint.  Writes a PID file so pci-doctor --stop can
terminate the server.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable

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

_STATE_DIR = Path.home() / ".local" / "state" / "project-code-intelligence"
MLX_SERVER_PID_FILE = _STATE_DIR / "pci-apple-mlx-server.pid"
MLX_SERVER_LOG_FILE = _STATE_DIR / "pci-apple-mlx-server.log"


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


def mlx_server_is_running() -> bool:
    """Return True if a pci-apple-mlx-server daemon is currently running."""
    pid = _read_pid(MLX_SERVER_PID_FILE)
    return pid is not None and _is_running(pid)


def mlx_model_name() -> str:
    return config.env_text("PROJECT_CODE_INTELLIGENCE_MLX_MODEL", config.DEFAULT_MLX_MODEL) or config.DEFAULT_MLX_MODEL


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


class SentenceTransformerModel(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = ...,
        convert_to_numpy: bool = ...,
    ) -> object: ...


def _load_model(model_name: str) -> SentenceTransformerModel:
    try:
        st_module = import_module("sentence_transformers")
    except ImportError as exc:
        raise RuntimeError(
            "sentence_transformers is not installed. "
            "Install the apple-mlx extra: pip install 'project-code-intelligence[apple-mlx]'"
        ) from exc

    cls = cast("Callable[..., SentenceTransformerModel]", st_module.SentenceTransformer)

    mlx_available = False
    try:
        _ = import_module("mlx.core")
        mlx_available = True
    except ImportError:
        pass

    if mlx_available:
        try:
            model = cls(model_name, backend="mlx")
            _err("  Backend: MLX (Apple Silicon native)")
        except TypeError:
            model = cls(model_name)
            _err("  Backend: PyTorch/MPS (MLX backend not supported by this sentence-transformers version)")
    else:
        model = cls(model_name)
        _err("  Backend: PyTorch/MPS (install mlx for native MLX acceleration)")

    return model


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


def _mlx_request_max_bytes() -> int:
    return config.env_int("PROJECT_CODE_INTELLIGENCE_MLX_MAX_REQUEST_BYTES", 4 * 1024 * 1024, minimum=1024)


class MLXHTTPServer(ThreadingHTTPServer):
    model: SentenceTransformerModel
    model_name: str

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        model: SentenceTransformerModel,
        model_name: str,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.model = model
        self.model_name = model_name


def _vector_to_list(value: object) -> list[float]:
    tolist_fn = cast("Callable[[], object]", getattr(value, "tolist", None))
    if callable(tolist_fn):
        value = tolist_fn()
    if not isinstance(value, list):
        raise TypeError(f"unexpected embedding row type: {type(value).__name__}")
    result: list[float] = []
    for item in cast("list[object]", value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError("model returned a vector with non-numeric values")
        result.append(float(item))
    return result


def embedding_response(model: SentenceTransformerModel, model_name: str, request: JsonObject) -> JsonObject:
    texts = normalize_input(request.get("input"))
    raw = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    vectors = [_vector_to_list(row) for row in cast("list[object]", raw)]
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
    return _parse_json_body(handler, max_bytes=_mlx_request_max_bytes())


class MLXHandler(BaseHTTPRequestHandler):
    server_version = "project-code-intelligence-mlx/0.1"

    def _mlx_server(self) -> MLXHTTPServer:
        return cast("MLXHTTPServer", self.server)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            write_json(self, 200, {"ok": True, "model": self._mlx_server().model_name})
            return
        write_json(self, 404, json_error("not found"))

    def do_POST(self) -> None:
        if self.path != "/v1/embeddings":
            write_json(self, 404, json_error("not found"))
            return
        server = self._mlx_server()
        try:
            request = _parse_request(self)
            payload = embedding_response(server.model, server.model_name, request)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            write_json(self, 400, json_error(str(exc)))
            return
        write_json(self, 200, payload)


# ---------------------------------------------------------------------------
# Serve entry point (runs in the daemon process)
# ---------------------------------------------------------------------------


def _serve() -> None:
    model_name = mlx_model_name()
    host = config.env_text("PROJECT_CODE_INTELLIGENCE_EMBEDDING_HOST", "127.0.0.1") or "127.0.0.1"
    port = config.env_int("PROJECT_CODE_INTELLIGENCE_EMBEDDING_PORT", 18081, minimum=1)

    _err(f"Loading MLX embedding model: {model_name}")
    model = _load_model(model_name)

    server = MLXHTTPServer((host, port), MLXHandler, model=model, model_name=model_name)
    _err(f"pci-apple-mlx-server ready on http://{host}:{port}")
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

    pid = _read_pid(MLX_SERVER_PID_FILE)
    if pid is not None and _is_running(pid):
        _err(f"pci-apple-mlx-server already running (PID {pid}).")
        return

    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    with MLX_SERVER_LOG_FILE.open("a") as log_file:
        proc = process.popen(
            [sys.executable, "-m", "project_code_intelligence.embedding.apple_mlx_server", "--serve"],
            PopenOptions(
                stdout=log_file,
                stderr=log_file,
                stdin=process.DEVNULL,
                start_new_session=True,
            ),
        )
    _ = MLX_SERVER_PID_FILE.write_text(str(proc.pid) + "\n")
    _err(f"pci-apple-mlx-server started (PID {proc.pid}). Log: {MLX_SERVER_LOG_FILE}")


if __name__ == "__main__":
    main()
