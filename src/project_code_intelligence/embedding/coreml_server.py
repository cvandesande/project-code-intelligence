"""OpenAI-compatible embedding server using Apple Core ML (ANE + GPU + CPU)."""

from __future__ import annotations

import json
import os
import sys

# Suppress "PyTorch was not found" advisory from transformers; we only use the
# tokenizer and do not need PyTorch.  Must be set before any transitive import.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")  # pyright: ignore[reportUnusedCallResult]

# Suppress HuggingFace Hub progress bars during cached model verification.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # pyright: ignore[reportUnusedCallResult]

import signal
import time
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
from project_code_intelligence.runtime import estimate_embedding_tokens

DEFAULT_COREML_MODEL = "ewchampion/Qwen3-Embedding-0.6B-coreml-4bit.mlpackage"
DEFAULT_COREML_HOST = "127.0.0.1"
DEFAULT_COREML_PORT = 18081
DEFAULT_PID_DIR = Path.home() / ".cache" / "project-code-intelligence"
PID_FILE_NAME = "pci-coreml-server.pid"


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
        config.env_text("PROJECT_CODE_INTELLIGENCE_COREML_MODEL", DEFAULT_COREML_MODEL, env=env) or DEFAULT_COREML_MODEL
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
    if length > coreml_request_max_bytes():
        raise ValueError("request body exceeds PROJECT_CODE_INTELLIGENCE_COREML_MAX_REQUEST_BYTES")
    raw = handler.rfile.read(length)
    value = cast("object", json.loads(raw.decode("utf-8")))
    if not isinstance(value, dict):
        raise TypeError("request body must be a JSON object")
    return cast("JsonObject", value)


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


def _compile_for_plan(model_path: str) -> str | None:
    """Compile a .mlpackage to .mlmodelc for compute plan inspection."""
    if model_path.endswith(".mlmodelc"):
        return model_path
    try:
        ct = import_module("coremltools")
    except ImportError:
        return None
    utils_mod = cast("object", getattr(ct, "utils", None))
    compile_fn = getattr(utils_mod, "compile_model", None) if utils_mod else None
    if compile_fn is None:
        return None
    try:
        return cast("str", compile_fn(model_path))
    except Exception:  # noqa: BLE001 - coremltools compile can raise arbitrary errors
        return None


def _classify_device(
    device: object,
    *,
    ane_cls: type[object] | None,
    gpu_cls: type[object] | None,
    cpu_cls: type[object] | None,
) -> str:
    """Return a device label for a preferred_compute_device instance."""
    if ane_cls is not None and isinstance(device, ane_cls):
        return "ANE"
    if gpu_cls is not None and isinstance(device, gpu_cls):
        return "GPU"
    if cpu_cls is not None and isinstance(device, cpu_cls):
        return "CPU"
    return "unknown"


def _plan_operations(plan: object) -> list[object]:
    """Extract the flat list of ML Program operations from a compute plan."""
    model_structure = cast("object | None", getattr(plan, "model_structure", None))
    program = cast("object | None", getattr(model_structure, "program", None)) if model_structure is not None else None
    functions_raw = cast("object | None", getattr(program, "functions", None)) if program is not None else None
    if functions_raw is None:
        return []
    ops: list[object] = []
    for function in cast("dict[str, object]", functions_raw).values():
        block = cast("object | None", getattr(function, "block", None))
        operations_raw = cast("object | None", getattr(block, "operations", None)) if block is not None else None
        if operations_raw is not None:
            ops.extend(cast("list[object]", operations_raw))
    return ops


def _walk_plan_operations(
    plan: object,
    *,
    ane_cls: type[object] | None,
    gpu_cls: type[object] | None,
    cpu_cls: type[object] | None,
) -> tuple[dict[str, int], float, int]:
    """Walk ML Program operations and count device assignments.

    Returns (counts_by_device, ane_weight, total_ops).
    """
    ops = _plan_operations(plan)
    if not ops:
        return {}, 0.0, 0

    get_usage = getattr(plan, "get_compute_device_usage_for_mlprogram_operation", None)
    get_cost = getattr(plan, "get_estimated_cost_for_mlprogram_operation", None)
    counts: dict[str, int] = {"ANE": 0, "GPU": 0, "CPU": 0, "unknown": 0}
    ane_weight = 0.0

    for op in ops:
        if get_usage is None:
            counts["unknown"] += 1
            continue
        usage = cast("object | None", get_usage(op))
        if usage is None:
            counts["unknown"] += 1
            continue
        device = cast("object | None", getattr(usage, "preferred_compute_device", None))
        if device is None:
            counts["unknown"] += 1
            continue
        label = _classify_device(device, ane_cls=ane_cls, gpu_cls=gpu_cls, cpu_cls=cpu_cls)
        counts[label] = counts.get(label, 0) + 1
        if label == "ANE" and get_cost is not None:
            cost = cast("object | None", get_cost(op))
            weight = cast("object | None", getattr(cost, "weight", None)) if cost is not None else None
            if isinstance(weight, float):
                ane_weight += weight
    return counts, ane_weight, len(ops)


def _load_compute_plan(model_path: str) -> object | None:
    """Compile model and load its MLComputePlan, or return None on failure."""
    try:
        compute_plan_mod = import_module("coremltools.models.compute_plan")
    except ImportError:
        _ = sys.stderr.write("  MLComputePlan requires coremltools 8.0+; skipping compute plan analysis.\n")
        return None

    ct = import_module("coremltools")
    compute_unit_cls = cast("object", getattr(ct, "ComputeUnit", None))
    compute_all = cast("object", getattr(compute_unit_cls, "ALL", None)) if compute_unit_cls is not None else None
    if compute_all is None:
        _ = sys.stderr.write("  ComputeUnit.ALL not available.\n")
        return None

    plan_cls = cast("object", getattr(compute_plan_mod, "MLComputePlan", None))
    load_fn = getattr(plan_cls, "load_from_path", None) if plan_cls is not None else None
    if load_fn is None:
        _ = sys.stderr.write("  MLComputePlan.load_from_path not available.\n")
        return None

    compiled_path = _compile_for_plan(model_path)
    if compiled_path is None:
        _ = sys.stderr.write("  Could not compile model for compute plan inspection.\n")
        return None

    try:
        return cast("object", load_fn(compiled_path, compute_units=compute_all))
    except Exception as exc:  # noqa: BLE001 - coremltools internals can raise arbitrary errors
        _ = sys.stderr.write(f"  Could not load compute plan: {exc}\n")
        return None


def _format_compute_plan(model_path: str) -> str | None:
    """Load a Core ML compute plan and return a device assignment summary string."""
    plan = _load_compute_plan(model_path)
    if plan is None:
        return None

    try:
        compute_device_mod = import_module("coremltools.models.compute_device")
    except ImportError:
        return None

    ane_cls = cast("type[object] | None", getattr(compute_device_mod, "MLNeuralEngineComputeDevice", None))
    gpu_cls = cast("type[object] | None", getattr(compute_device_mod, "MLGPUComputeDevice", None))
    cpu_cls = cast("type[object] | None", getattr(compute_device_mod, "MLCPUComputeDevice", None))

    counts, ane_weight, total = _walk_plan_operations(plan, ane_cls=ane_cls, gpu_cls=gpu_cls, cpu_cls=cpu_cls)

    if total == 0:
        return None

    lines: list[str] = [f"\n  Compute plan ({total} operations):"]
    for name in ("ANE", "GPU", "CPU"):
        count = counts.get(name, 0)
        if count > 0:
            lines.append(f"    {name}: {count} ops ({100 * count // total}%)")
    unknown = counts.get("unknown", 0)
    if unknown > 0:
        lines.append(f"    Unassigned: {unknown} ops")

    if counts.get("ANE", 0) > 0:
        ane_count = counts["ANE"]
        lines.append(f"\n  Neural Engine will handle {ane_count}/{total} operations.")
        if ane_weight > 0:
            lines.append(f"  Estimated ANE workload share: {ane_weight:.0%}")
    else:
        lines.extend([
            "\n  No operations scheduled for Neural Engine.",
            "  This model may not support ANE acceleration on this hardware.",
        ])
    lines.append("")
    return "\n".join(lines)


def print_compute_plan(model_path: str) -> None:
    """Load a Core ML compute plan and print device assignment summary to stderr."""
    text = _format_compute_plan(model_path)
    if text:
        _ = sys.stderr.write(text)


def diagnose() -> int:
    """Load the Core ML model, inspect its compute plan, and benchmark inference."""
    import time  # noqa: PLC0415

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


def _pid_file_path() -> Path:
    """Return the path to the PID file for the Core ML server."""
    return DEFAULT_PID_DIR / PID_FILE_NAME


def _write_pid_file() -> None:
    """Write the current process PID to the PID file."""
    pid_file = _pid_file_path()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    _ = pid_file.write_text(str(os.getpid()) + "\n")


def _remove_pid_file() -> None:
    """Remove the PID file if it exists."""
    _pid_file_path().unlink(missing_ok=True)


def _read_pid_file() -> int | None:
    """Read the PID from the PID file, or None if absent/invalid."""
    try:
        text = _pid_file_path().read_text().strip()
        return int(text) if text else None
    except (FileNotFoundError, ValueError):
        return None


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists but we can't signal it
    return True


def stop_server() -> bool:
    """Stop a running Core ML server via PID file. Returns True if a signal was sent."""
    pid = _read_pid_file()
    if pid is None:
        return False
    if not _is_pid_alive(pid):
        _remove_pid_file()
        return False
    import signal  # noqa: PLC0415

    os.kill(pid, signal.SIGTERM)
    _remove_pid_file()
    return True


def serve(*, foreground: bool = False) -> int:
    # Check for an already-running instance.
    existing_pid = _read_pid_file()
    if existing_pid is not None and _is_pid_alive(existing_pid):
        _ = sys.stderr.write(f"Core ML server is already running (PID {existing_pid}).\n")
        _ = sys.stderr.write("Stop it first with: pci-doctor --stop\n")
        return 1

    model_name = coreml_model_name()
    _ = sys.stderr.write(f"Loading Core ML model {model_name} with compute_units=ALL (ANE + GPU + CPU)...\n")
    _ = sys.stderr.flush()
    embedder = load_coreml_embedder(model_name)

    # Log the compute plan so users can see ANE/GPU/CPU scheduling at startup.
    with _ProgressIndicator("Loading compute plan"):
        huggingface_hub = import_module("huggingface_hub")
        cache_dir = config.env_text("PROJECT_CODE_INTELLIGENCE_COREML_CACHE_DIR")
        model_path = _resolve_model_path(model_name, huggingface_hub=huggingface_hub, cache_dir=cache_dir)
        compute_plan_text = _format_compute_plan(model_path)
    if compute_plan_text:
        _ = sys.stderr.write(compute_plan_text)
        _ = sys.stderr.flush()

    host = config.env_text("PROJECT_CODE_INTELLIGENCE_COREML_HOST", DEFAULT_COREML_HOST) or DEFAULT_COREML_HOST
    port = config.env_int("PROJECT_CODE_INTELLIGENCE_COREML_PORT", DEFAULT_COREML_PORT, minimum=1)
    server = CoreMLHTTPServer((host, port), CoreMLHandler, embedder=embedder)
    max_info = f", max_length={embedder.max_length}" if embedder.max_length else ""
    _ = sys.stderr.write(f"Core ML embedding server listening on {host}:{port} with model {model_name}{max_info}\n")
    _ = sys.stderr.flush()

    if not foreground:
        pid = os.fork()
        if pid > 0:
            # Parent: print the daemon PID and exit so the user gets their shell back.
            _ = sys.stderr.write(f"Daemonized as PID {pid}. Stop with: pci-doctor --stop\n")
            return 0
        # Child: detach from the controlling terminal.
        os.setsid()

    _write_pid_file()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        _remove_pid_file()
    return 0


def main() -> int:
    if "--diagnose" in sys.argv:
        return diagnose()
    foreground = "--foreground" in sys.argv
    return serve(foreground=foreground)


if __name__ == "__main__":
    raise SystemExit(main())
