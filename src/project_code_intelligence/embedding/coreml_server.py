"""OpenAI-compatible embedding server using Apple Core ML (ANE + GPU + CPU)."""

from __future__ import annotations

import json
import os
import signal
import sys
import time

# Suppress "PyTorch was not found" advisory from transformers; we only use the
# tokenizer and do not need PyTorch.  Must be set before any transitive import.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")  # pyright: ignore[reportUnusedCallResult]

# Suppress HuggingFace Hub progress bars during cached model verification.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # pyright: ignore[reportUnusedCallResult]

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import override

    from project_code_intelligence.models import JsonObject
else:
    _T = TypeVar("_T")

    def override(method: _T) -> _T:
        return method


from project_code_intelligence import config
from project_code_intelligence.embedding.coreml_compute_plan import (
    format_compute_plan,
    print_compute_plan,
)
from project_code_intelligence.embedding.coreml_lifecycle import (
    is_pid_alive,
    read_pid_file,
    remove_pid_file,
    write_pid_file,
)
from project_code_intelligence.embedding.http_common import (
    json_error,
    normalize_input,
    write_json,
)
from project_code_intelligence.embedding.http_common import parse_json_body as _parse_json_body
from project_code_intelligence.runtime import estimate_embedding_tokens

DEFAULT_COREML_HOST = "127.0.0.1"
DEFAULT_COREML_PORT = 18081


# ---------------------------------------------------------------------------
# Numpy helper protocols - typed wrappers around dynamically imported numpy.
# ---------------------------------------------------------------------------


class NdArray(Protocol):
    """Minimal protocol for numpy ndarray operations used in pooling."""

    def flatten(self) -> NdArray: ...

    def tolist(self) -> list[float]: ...

    def astype(self, dtype: object) -> NdArray: ...

    def __mul__(self, other: object) -> NdArray: ...

    def __truediv__(self, other: object) -> NdArray: ...


class NumpyModule(Protocol):
    """Subset of numpy used by the mean-pooling path."""

    int32: object

    def expand_dims(self, a: object, axis: int) -> NdArray: ...

    def sum(self, a: object, *, axis: int) -> NdArray: ...

    def clip(self, a: object, a_min: float | None, a_max: float | None) -> NdArray: ...


class NumpyLinalgModule(Protocol):
    def norm(self, x: object, *, axis: int, keepdims: bool) -> NdArray: ...


class CoreMLTokenizer(Protocol):
    def __call__(self, text: list[str], **kwargs: object) -> object: ...


class CoreMLModel(Protocol):
    def predict(self, input_data: dict[str, object]) -> dict[str, object]: ...


class CoreMLEmbedder:
    """Wraps a Core ML model and tokenizer into an embedding pipeline."""

    def __init__(
        self, model: CoreMLModel, tokenizer: CoreMLTokenizer, model_name: str, *, max_length: int | None = None
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length

    def embed(self, texts: list[str]) -> list[list[float]]:
        np = _import_numpy()
        vectors: list[list[float]] = []
        pad_kwargs: dict[str, object] = {"padding": True, "truncation": True, "return_tensors": "np"}
        if self.max_length is not None:
            pad_kwargs["padding"] = "max_length"
            pad_kwargs["max_length"] = self.max_length
        for text in texts:
            encoded = self.tokenizer([text], **pad_kwargs)
            input_ids = _to_numpy_int32(encoded, "input_ids", np=np)
            attention_mask = _to_numpy_int32(encoded, "attention_mask", np=np)
            prediction = self.model.predict({"input_ids": input_ids, "attention_mask": attention_mask})
            embedding = _extract_embedding(prediction, attention_mask, np=np)
            vectors.append(embedding)
        return vectors


def _import_numpy() -> NumpyModule:
    try:
        return cast("NumpyModule", import_module("numpy"))
    except ImportError as exc:
        raise RuntimeError(
            "numpy is not installed. Install the apple embedding extra with "
            "python -m pip install -e '.[apple-embeddings]'."
        ) from exc


def _import_numpy_linalg() -> NumpyLinalgModule:
    return cast("NumpyLinalgModule", import_module("numpy.linalg"))


def _to_numpy_int32(encoded: object, key: str, *, np: NumpyModule) -> NdArray:
    value = cast("NdArray | None", getattr(encoded, key, None))
    if value is None:
        mapping = cast("dict[str, NdArray]", encoded)
        value = mapping.get(key)
    if value is None:
        raise ValueError(f"tokenizer output missing '{key}'")
    return value.astype(np.int32)


def _flatten_to_list(arr: NdArray) -> list[float]:
    return arr.flatten().tolist()


def _first_output(prediction: dict[str, object], *keys: str) -> object | None:
    """Return the first non-None prediction value for the given keys."""
    for key in keys:
        value = prediction.get(key)
        if value is not None:
            return value
    return None


def _extract_embedding(prediction: dict[str, object], attention_mask: NdArray, *, np: NumpyModule) -> list[float]:
    token_embeddings = _first_output(prediction, "last_hidden_state", "hidden_states", "token_embeddings")
    if token_embeddings is None:
        sentence_embedding = _first_output(prediction, "sentence_embedding", "embeddings")
        if sentence_embedding is not None:
            return _flatten_to_list(cast("NdArray", sentence_embedding))
        raise ValueError(f"Core ML model returned unexpected output keys: {sorted(prediction.keys())}")

    return _mean_pool_and_normalize(cast("NdArray", token_embeddings), attention_mask, np=np)


def _mean_pool_and_normalize(token_embeddings: NdArray, attention_mask: NdArray, *, np: NumpyModule) -> list[float]:
    mask_expanded = np.expand_dims(attention_mask, -1)
    summed = np.sum(token_embeddings * mask_expanded, axis=1)
    count = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
    pooled = summed / count
    linalg = _import_numpy_linalg()
    norm = linalg.norm(pooled, axis=-1, keepdims=True)
    normalized = pooled / np.clip(norm, a_min=1e-9, a_max=None)
    return _flatten_to_list(normalized)


def coreml_model_name(env: config.Env | None = None) -> str:
    return (
        config.env_text("PROJECT_CODE_INTELLIGENCE_COREML_MODEL", config.DEFAULT_COREML_MODEL, env=env)
        or config.DEFAULT_COREML_MODEL
    )


def coreml_request_max_bytes() -> int:
    return config.env_int("PROJECT_CODE_INTELLIGENCE_COREML_MAX_REQUEST_BYTES", 4 * 1024 * 1024, minimum=1024)


class _ProgressIndicator:
    """Context manager that prints dots from a forked child process.

    A background *thread* cannot reliably print during heavy C-extension work
    (e.g. Core ML model compilation) because the GIL blocks thread scheduling.
    A forked child process is independent and prints dots on time regardless of
    GIL contention in the parent.

    Stderr is temporarily redirected to ``/dev/null`` in the parent so library
    warnings do not break the progress line.  The child writes to a saved copy
    of the original stderr fd.
    """

    def __init__(self, label: str) -> None:
        self._label = label
        self._saved_stderr: int | None = None
        self._child_pid: int | None = None

    def __enter__(self) -> _ProgressIndicator:
        # Duplicate the real stderr before redirecting fd 2 to /dev/null.
        saved_fd = os.dup(2)
        self._saved_stderr = saved_fd
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        _ = os.dup2(devnull_fd, 2)
        os.close(devnull_fd)

        pid = os.fork()
        if pid == 0:
            # Child: print dots until killed.
            _ = os.write(saved_fd, f"  {self._label} .".encode())
            try:
                while True:
                    time.sleep(0.5)
                    _ = os.write(saved_fd, b".")
            except (KeyboardInterrupt, SystemExit):
                pass
            os._exit(0)  # child must not run parent cleanup
        else:
            self._child_pid = pid
        return self

    def __exit__(self, *_args: object) -> None:
        # Stop the child process.
        if self._child_pid is not None:
            os.kill(self._child_pid, signal.SIGTERM)
            _ = os.waitpid(self._child_pid, 0)
            self._child_pid = None
        # Write the "done" message and restore stderr.
        if self._saved_stderr is not None:
            _ = os.write(self._saved_stderr, b". done\n")
            _ = os.dup2(self._saved_stderr, 2)
            os.close(self._saved_stderr)
            self._saved_stderr = None


def load_coreml_embedder(model_name: str) -> CoreMLEmbedder:
    try:
        ct = import_module("coremltools")
    except ImportError as exc:
        raise RuntimeError(
            "coremltools is not installed. Install the apple embedding extra with "
            "python -m pip install -e '.[apple-embeddings]'."
        ) from exc

    try:
        transformers = import_module("transformers")
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed. Install the apple embedding extra with "
            "python -m pip install -e '.[apple-embeddings]'."
        ) from exc

    try:
        huggingface_hub = import_module("huggingface_hub")
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install the apple embedding extra with "
            "python -m pip install -e '.[apple-embeddings]'."
        ) from exc

    compute_unit_cls = cast("object", getattr(ct, "ComputeUnit", None))
    if compute_unit_cls is None:
        raise RuntimeError("coremltools.ComputeUnit is not available")
    compute_all = cast("object", getattr(compute_unit_cls, "ALL", None))
    if compute_all is None:
        raise RuntimeError("coremltools.ComputeUnit.ALL is not available")

    cache_dir = config.env_text("PROJECT_CODE_INTELLIGENCE_COREML_CACHE_DIR")
    with _ProgressIndicator("Resolving model"):
        model_path = _resolve_model_path(model_name, huggingface_hub=huggingface_hub, cache_dir=cache_dir)

    ct_models = cast("object", getattr(ct, "models", None))
    ml_model_cls = cast("Callable[..., CoreMLModel] | None", getattr(ct_models, "MLModel", None) if ct_models else None)
    if ml_model_cls is None:
        raise RuntimeError("coremltools.models.MLModel is not available")
    with _ProgressIndicator("Compiling model"):
        model = ml_model_cls(model_path, compute_units=compute_all)
    max_length = _detect_max_length(model)

    tokenizer_name = config.env_text("PROJECT_CODE_INTELLIGENCE_COREML_TOKENIZER", model_name)
    auto_tokenizer = cast("object", getattr(transformers, "AutoTokenizer", None))
    if auto_tokenizer is None:
        raise RuntimeError("transformers.AutoTokenizer is not available")
    from_pretrained = cast("Callable[..., CoreMLTokenizer] | None", getattr(auto_tokenizer, "from_pretrained", None))
    if from_pretrained is None:
        raise RuntimeError("transformers.AutoTokenizer.from_pretrained is not available")
    with _ProgressIndicator("Loading tokenizer"):
        tokenizer = from_pretrained(tokenizer_name)

    return CoreMLEmbedder(model=model, tokenizer=tokenizer, model_name=model_name, max_length=max_length)


def _resolve_model_path(model_name: str, *, huggingface_hub: object, cache_dir: str | None) -> str:
    if Path(model_name).exists():
        return model_name
    for suffix in (".mlpackage", ".mlmodelc"):
        candidate = Path(model_name + suffix)
        if candidate.exists():
            return str(candidate)

    snapshot_download = cast("Callable[..., str] | None", getattr(huggingface_hub, "snapshot_download", None))
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub.snapshot_download is not available")

    # First try downloading only mlpackage/mlmodelc files (nested layout).
    kwargs: dict[str, object] = {"repo_id": model_name, "allow_patterns": ["*.mlpackage/*", "*.mlmodelc/*"]}
    if cache_dir:
        kwargs["local_dir"] = cache_dir
    local_dir = snapshot_download(**kwargs)
    local_path = Path(local_dir)
    for suffix in (".mlpackage", ".mlmodelc"):
        candidates = sorted(local_path.rglob(f"*{suffix}"))
        if candidates:
            return str(candidates[0])

    # Some repos have the mlpackage at the root (Manifest.json + Data/).
    # Download the full repo and check for this layout.
    full_kwargs: dict[str, object] = {"repo_id": model_name}
    if cache_dir:
        full_kwargs["local_dir"] = cache_dir
    local_dir = snapshot_download(**full_kwargs)
    local_path = Path(local_dir)
    if _is_mlpackage_dir(local_path):
        return _materialize_mlpackage(local_path, cache_dir=cache_dir)

    raise FileNotFoundError(
        f"No .mlpackage or .mlmodelc found after downloading {model_name}. "
        "Convert a model with coremltools or use a repository that includes a Core ML model."
    )


def _is_mlpackage_dir(path: Path) -> bool:
    """Check if a directory has the standard mlpackage layout at its root."""
    return (path / "Manifest.json").is_file() and (path / "Data").is_dir()


def _materialize_mlpackage(source: Path, *, cache_dir: str | None) -> str:
    """Copy an mlpackage from the HF cache to a named .mlpackage directory.

    HuggingFace stores downloads with symlinks into a blob store which
    coremltools' compiler cannot follow. Copying the files into a
    conventionally-named directory resolves this.
    """
    import shutil  # noqa: PLC0415

    dest_parent = Path(cache_dir) if cache_dir else source.parent
    dest = dest_parent / (source.name + ".mlpackage")
    if dest.exists() and _is_mlpackage_dir(dest):
        return str(dest)
    dest_tmp = dest.with_suffix(".mlpackage.tmp")
    if dest_tmp.exists():
        shutil.rmtree(dest_tmp)
    _ = shutil.copytree(source, dest_tmp, symlinks=False)
    _ = dest_tmp.rename(dest)
    return str(dest)


def _detect_max_length(model: CoreMLModel) -> int | None:
    """Read the fixed sequence length from a Core ML model spec, if any."""
    get_spec = getattr(model, "get_spec", None)
    if get_spec is None:
        return None
    try:
        spec = cast("object", get_spec())
    except Exception:  # noqa: BLE001 - spec access can fail for various reasons
        return None
    description = cast("object | None", getattr(spec, "description", None))
    inputs = cast("object | None", getattr(description, "input", None)) if description is not None else None
    if inputs is None:
        return None
    for inp in cast("list[object]", inputs):
        name = cast("str | None", getattr(inp, "name", None))
        if name != "input_ids":
            continue
        type_field = cast("object | None", getattr(inp, "type", None))
        multi_array = cast("object | None", getattr(type_field, "multiArrayType", None)) if type_field else None
        shape = cast("list[int] | None", getattr(multi_array, "shape", None)) if multi_array else None
        min_shape_rank = 2  # [batch, seq_len]
        if shape is not None and len(shape) >= min_shape_rank:
            return int(shape[-1])
    return None


def embedding_response(embedder: CoreMLEmbedder, request: JsonObject) -> JsonObject:
    texts = normalize_input(request.get("input"))
    vectors = embedder.embed(texts)
    if len(vectors) != len(texts):
        raise ValueError("Core ML model returned an unexpected number of vectors")
    prompt_tokens = sum(estimate_embedding_tokens(text) for text in texts)
    return {
        "object": "list",
        "model": embedder.model_name,
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


def parse_json_body(handler: BaseHTTPRequestHandler) -> JsonObject:
    return _parse_json_body(handler, max_bytes=coreml_request_max_bytes())


class CoreMLHTTPServer(ThreadingHTTPServer):
    embedder: CoreMLEmbedder

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        embedder: CoreMLEmbedder,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.embedder = embedder


class CoreMLHandler(BaseHTTPRequestHandler):
    server_version = "project-code-intelligence-coreml/0.1"

    def coreml_server(self) -> CoreMLHTTPServer:
        return cast("CoreMLHTTPServer", self.server)

    @override
    def log_message(self, format: str, *args: object) -> None:
        if config.env_bool("PROJECT_CODE_INTELLIGENCE_COREML_ACCESS_LOG", default=False):
            super().log_message(format, *args)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            write_json(self, 200, {"ok": True, "model": self.coreml_server().embedder.model_name})
            return
        write_json(self, 404, json_error("not found"))

    def do_POST(self) -> None:
        if self.path != "/v1/embeddings":
            write_json(self, 404, json_error("not found"))
            return
        server = self.coreml_server()
        try:
            request = parse_json_body(self)
            payload = embedding_response(server.embedder, request)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            write_json(self, 400, json_error(str(exc)))
            return
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive HTTP boundary
            write_json(self, 500, json_error(str(exc)))
            return
        write_json(self, 200, payload)


def diagnose() -> int:
    """Load the Core ML model, inspect its compute plan, and benchmark inference."""
    model_name = coreml_model_name()
    _ = sys.stderr.write(f"Core ML diagnostics for {model_name}\n")
    _ = sys.stderr.write("=" * 60 + "\n\n")

    _ = sys.stderr.write("Loading model...\n")
    _ = sys.stderr.flush()
    embedder = load_coreml_embedder(model_name)

    _ = sys.stderr.write("\nCompute plan analysis:\n")
    huggingface_hub = import_module("huggingface_hub")
    cache_dir = config.env_text("PROJECT_CODE_INTELLIGENCE_COREML_CACHE_DIR")
    model_path = _resolve_model_path(model_name, huggingface_hub=huggingface_hub, cache_dir=cache_dir)
    print_compute_plan(model_path)

    _ = sys.stderr.write("\nInference benchmark:\n")
    _ = sys.stderr.flush()
    sample = "The quick brown fox jumps over the lazy dog."
    _ = embedder.embed([sample])  # warmup
    n_runs = 10
    start = time.monotonic()
    for _i in range(n_runs):
        _ = embedder.embed([sample])
    elapsed = time.monotonic() - start
    dim = len(embedder.embed([sample])[0])
    _ = sys.stderr.write(f"  Dimension: {dim}\n")
    _ = sys.stderr.write(f"  Latency: {elapsed / n_runs * 1000:.1f} ms/embedding ({n_runs} runs)\n")

    _ = sys.stderr.write(
        "\nRuntime verification:\n"
        "  While running inference, confirm ANE power draw with:\n"
        "    sudo powermetrics --samplers ane_energy -i 1000 -n 5\n"
        "  Non-zero ANE power confirms Neural Engine usage.\n"
    )
    return 0


def _daemonize() -> int | None:
    """Fork into a background daemon.  Returns a write-fd the child must
    write to once ready, or *None* in the parent (which should ``return 0``).
    The parent blocks until the child signals readiness so startup progress
    is printed before the shell prompt returns.
    """
    ready_read_fd, ready_write_fd = os.pipe()
    pid = os.fork()
    if pid > 0:
        # Parent: wait for the child to signal readiness then exit.
        os.close(ready_write_fd)
        _ = os.read(ready_read_fd, 1)  # blocks until child writes or pipe breaks
        os.close(ready_read_fd)
        return None
    # Child: detach from the controlling terminal but keep stderr for
    # loading progress.  stdin/stdout go to /dev/null immediately.
    os.close(ready_read_fd)
    os.setsid()
    devnull = os.open(os.devnull, os.O_RDWR)
    _ = os.dup2(devnull, sys.stdin.fileno())
    _ = os.dup2(devnull, sys.stdout.fileno())
    os.close(devnull)
    return ready_write_fd


def _silence_stderr() -> None:
    """Redirect stderr to /dev/null."""
    devnull = os.open(os.devnull, os.O_RDWR)
    _ = os.dup2(devnull, sys.stderr.fileno())
    os.close(devnull)


def serve(*, foreground: bool = False) -> int:
    # Check for an already-running instance.
    existing_pid = read_pid_file()
    if existing_pid is not None and is_pid_alive(existing_pid):
        _ = sys.stderr.write(f"Core ML server is already running (PID {existing_pid}).\n")
        _ = sys.stderr.write("Stop it first with: pci-doctor --stop\n")
        return 1

    ready_write_fd: int | None = None
    if not foreground:
        ready_write_fd = _daemonize()
        if ready_write_fd is None:
            return 0  # parent

    # Load the model in the process that will actually serve requests.
    # Core ML's Objective-C/Metal/ANE runtime handles do not survive fork().
    model_name = coreml_model_name()
    _ = sys.stderr.write(f"Loading Core ML model {model_name} with compute_units=ALL (ANE + GPU + CPU)...\n")
    _ = sys.stderr.flush()
    embedder = load_coreml_embedder(model_name)

    # Log the compute plan so users can see ANE/GPU/CPU scheduling at startup.
    with _ProgressIndicator("Loading compute plan"):
        huggingface_hub = import_module("huggingface_hub")
        cache_dir = config.env_text("PROJECT_CODE_INTELLIGENCE_COREML_CACHE_DIR")
        model_path = _resolve_model_path(model_name, huggingface_hub=huggingface_hub, cache_dir=cache_dir)
        compute_plan_text = format_compute_plan(model_path)
    if compute_plan_text:
        _ = sys.stderr.write(compute_plan_text)
        _ = sys.stderr.flush()

    host = config.env_text("PROJECT_CODE_INTELLIGENCE_COREML_HOST", DEFAULT_COREML_HOST) or DEFAULT_COREML_HOST
    port = config.env_int("PROJECT_CODE_INTELLIGENCE_COREML_PORT", DEFAULT_COREML_PORT, minimum=1)
    server = CoreMLHTTPServer((host, port), CoreMLHandler, embedder=embedder)
    max_info = f", max_length={embedder.max_length}" if embedder.max_length else ""
    _ = sys.stderr.write(f"Core ML embedding server listening on {host}:{port} with model {model_name}{max_info}\n")

    if not foreground:
        _ = sys.stderr.write(f"Daemonized as PID {os.getpid()}. Stop with: pci-doctor --stop\n")
        _ = sys.stderr.flush()
        # Signal the parent that loading is done so it can return the shell.
        _ = os.write(ready_write_fd, b"\n")  # type: ignore[arg-type]
        os.close(ready_write_fd)  # type: ignore[arg-type]
        # Now redirect stderr to /dev/null for the long-running server.
        _silence_stderr()
    else:
        _ = sys.stderr.flush()

    write_pid_file()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        remove_pid_file()
    return 0


def main() -> int:
    if "--diagnose" in sys.argv:
        return diagnose()
    foreground = "--foreground" in sys.argv
    return serve(foreground=foreground)


if __name__ == "__main__":
    raise SystemExit(main())
