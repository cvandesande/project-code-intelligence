"""Small OpenAI-compatible FastEmbed embedding server."""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from typing_extensions import override

    from project_code_intelligence.models import JsonObject
else:
    _T = TypeVar("_T")

    def override(method: _T) -> _T:
        return method


from project_code_intelligence import config
from project_code_intelligence.runtime import estimate_embedding_tokens


class FastEmbedModel(Protocol):
    def embed(self, documents: list[str]) -> Iterable[object]: ...


class FastEmbedFactory(Protocol):
    def __call__(self, **kwargs: object) -> FastEmbedModel: ...


class FastEmbedHTTPServer(ThreadingHTTPServer):
    model: FastEmbedModel
    model_name: str

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        model: FastEmbedModel,
        model_name: str,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.model = model
        self.model_name = model_name


def fastembed_model_name() -> str:
    return (
        config.env_text("PROJECT_CODE_INTELLIGENCE_FASTEMBED_MODEL", config.DEFAULT_FASTEMBED_MODEL)
        or config.DEFAULT_FASTEMBED_MODEL
    )


def fastembed_cache_dir() -> str | None:
    return config.env_text("PROJECT_CODE_INTELLIGENCE_FASTEMBED_CACHE_DIR")


def fastembed_request_max_bytes() -> int:
    return config.env_int("PROJECT_CODE_INTELLIGENCE_FASTEMBED_MAX_REQUEST_BYTES", 4 * 1024 * 1024, minimum=1024)


def load_fastembed_model(model_name: str) -> FastEmbedModel:
    try:
        module = import_module("fastembed")
    except ImportError as exc:
        raise RuntimeError(
            "FastEmbed is not installed. Install the local embedding extra with "
            "python -m pip install -e '.[local-embeddings]', or use the Docker Compose cpu profile."
        ) from exc
    text_embedding_value = cast("object", getattr(module, "TextEmbedding", None))
    if not callable(text_embedding_value):
        raise TypeError("fastembed.TextEmbedding was not available")
    kwargs: dict[str, object] = {"model_name": model_name}
    cache_dir = fastembed_cache_dir()
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return cast("FastEmbedFactory", text_embedding_value)(**kwargs)


def normalize_input(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        texts: list[str] = []
        for item in cast("list[object]", value):
            if not isinstance(item, str):
                raise TypeError("embedding input array items must be strings")
            texts.append(item)
        if not texts:
            raise ValueError("embedding input array must not be empty")
        return texts
    raise TypeError("embedding input must be a string or an array of strings")


def vector_values(value: object) -> list[float]:
    tolist_value = cast("object", getattr(value, "tolist", None))
    if callable(tolist_value):
        value = cast("Callable[[], object]", tolist_value)()
    if isinstance(value, list):
        items = cast("list[object]", value)
    elif isinstance(value, tuple):
        items = list(cast("tuple[object, ...]", value))
    else:
        raise TypeError("FastEmbed returned a non-sequence vector")

    vector: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError("FastEmbed returned a vector with non-numeric values")
        vector.append(float(item))
    if not vector:
        raise ValueError("FastEmbed returned an empty vector")
    return vector


def embedding_response(model: FastEmbedModel, model_name: str, request: JsonObject) -> JsonObject:
    texts = normalize_input(request.get("input"))
    vectors = [vector_values(value) for value in model.embed(texts)]
    if len(vectors) != len(texts):
        raise ValueError("FastEmbed returned an unexpected number of vectors")
    prompt_tokens = sum(estimate_embedding_tokens(text) for text in texts)
    return {
        "object": "list",
        "model": model_name,
        "data": [
            {
                "object": "embedding",
                "index": index,
                "embedding": vector,
            }
            for index, vector in enumerate(vectors)
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    }


def json_error(message: str) -> JsonObject:
    return {"error": {"message": message, "type": "invalid_request_error"}}


def write_json(handler: BaseHTTPRequestHandler, status: int, payload: JsonObject) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    _ = handler.wfile.write(body)


def parse_json_body(handler: BaseHTTPRequestHandler) -> JsonObject:
    length_header = handler.headers.get("Content-Length")
    if not length_header:
        raise ValueError("Content-Length is required")
    try:
        length = int(length_header)
    except ValueError as exc:
        raise ValueError("Content-Length must be an integer") from exc
    if length <= 0:
        raise ValueError("request body is empty")
    if length > fastembed_request_max_bytes():
        raise ValueError("request body exceeds PROJECT_CODE_INTELLIGENCE_FASTEMBED_MAX_REQUEST_BYTES")
    raw = handler.rfile.read(length)
    value = cast("object", json.loads(raw.decode("utf-8")))
    if not isinstance(value, dict):
        raise TypeError("request body must be a JSON object")
    return cast("JsonObject", value)


class FastEmbedHandler(BaseHTTPRequestHandler):
    server_version = "project-code-intelligence-fastembed/0.1"

    def fastembed_server(self) -> FastEmbedHTTPServer:
        return cast("FastEmbedHTTPServer", self.server)

    @override
    def log_message(self, format: str, *args: object) -> None:
        if config.env_bool("PROJECT_CODE_INTELLIGENCE_FASTEMBED_ACCESS_LOG", default=False):
            super().log_message(format, *args)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            write_json(self, 200, {"ok": True, "model": self.fastembed_server().model_name})
            return
        write_json(self, 404, json_error("not found"))

    def do_POST(self) -> None:
        if self.path != "/v1/embeddings":
            write_json(self, 404, json_error("not found"))
            return
        server = self.fastembed_server()
        try:
            request = parse_json_body(self)
            payload = embedding_response(server.model, server.model_name, request)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            write_json(self, 400, json_error(str(exc)))
            return
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive HTTP boundary
            write_json(self, 500, json_error(str(exc)))
            return
        write_json(self, 200, payload)


def serve() -> int:
    model_name = fastembed_model_name()
    model = load_fastembed_model(model_name)
    host = (
        config.env_text("PROJECT_CODE_INTELLIGENCE_FASTEMBED_HOST", config.DEFAULT_FASTEMBED_HOST)
        or config.DEFAULT_FASTEMBED_HOST
    )
    port = config.env_int("PROJECT_CODE_INTELLIGENCE_FASTEMBED_PORT", config.DEFAULT_FASTEMBED_PORT, minimum=1)
    server = FastEmbedHTTPServer((host, port), FastEmbedHandler, model=model, model_name=model_name)
    _ = sys.stderr.write(f"FastEmbed server listening on {host}:{port} with model {model_name}\n")
    _ = sys.stderr.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main() -> int:
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
