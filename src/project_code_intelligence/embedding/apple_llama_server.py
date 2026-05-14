"""Start a daemonized llama.cpp embedding server on Apple Silicon.

Downloads the default GGUF model on first run if not already cached.
Writes a PID file so pci-doctor --stop can terminate the server.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from project_code_intelligence import process
from project_code_intelligence.process import PopenOptions

_DEFAULT_HF_MODEL_REPO = "nomic-ai/nomic-embed-code-GGUF"
_DEFAULT_HF_MODEL_FILE = "nomic-embed-code.Q8_0.gguf"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "project-code-intelligence" / "models"

_STATE_DIR = Path.home() / ".local" / "state" / "project-code-intelligence"
LLAMA_SERVER_PID_FILE = _STATE_DIR / "pci-apple-llama-server.pid"
LLAMA_SERVER_LOG_FILE = _STATE_DIR / "pci-apple-llama-server.log"


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _err(msg: str) -> None:
    _ = sys.stderr.write(msg + "\n")


def _resolve_llama_server() -> str:
    server_name = _env("PROJECT_CODE_INTELLIGENCE_LLAMA_SERVER", "llama-server")
    candidate = Path(server_name)
    if candidate.is_absolute() and candidate.is_file() and os.access(candidate, os.X_OK):
        return server_name
    found = shutil.which(server_name)
    if found:
        return found
    _err(f"llama.cpp ({server_name}) was not found on PATH.")
    _err("Install llama.cpp via Homebrew: brew install llama.cpp")
    sys.exit(1)


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


def llama_server_is_running() -> bool:
    """Return True if a pci-apple-llama-server daemon is currently running."""
    pid = _read_pid(LLAMA_SERVER_PID_FILE)
    return pid is not None and _is_running(pid)


_HF_MODEL_ID_PARTS = 2
_MODEL_FILE_EXTENSIONS = {".gguf", ".bin", ".safetensors"}


def looks_like_hf_model_id(value: str) -> bool:
    p = Path(value)
    return not p.is_absolute() and len(p.parts) == _HF_MODEL_ID_PARTS and p.suffix.lower() not in _MODEL_FILE_EXTENSIONS


def _ensure_model() -> Path:
    model_str = _env("PROJECT_CODE_INTELLIGENCE_LLAMA_MODEL")
    if model_str:
        model_path = Path(model_str)
        if model_path.is_file() and model_path.stat().st_size > 0:
            return model_path
        _err("PROJECT_CODE_INTELLIGENCE_LLAMA_MODEL must be a path to a local .gguf file.")
        _err(f"  Value set: {model_str}")
        if looks_like_hf_model_id(model_str):
            _err("  This looks like a HuggingFace model ID, not a local path.")
            _err("  To download a specific model, set:")
            _err(f"    PROJECT_CODE_INTELLIGENCE_HF_MODEL_REPO={model_str}")
            _err("    PROJECT_CODE_INTELLIGENCE_HF_MODEL_FILE=<filename>.gguf")
            _err("  Then unset PROJECT_CODE_INTELLIGENCE_LLAMA_MODEL and re-run.")
        sys.exit(1)

    repo = _env("PROJECT_CODE_INTELLIGENCE_HF_MODEL_REPO", _DEFAULT_HF_MODEL_REPO)
    file = _env("PROJECT_CODE_INTELLIGENCE_HF_MODEL_FILE", _DEFAULT_HF_MODEL_FILE)
    cache_dir = Path(_env("PROJECT_CODE_INTELLIGENCE_LLAMA_MODEL_CACHE_DIR", str(_DEFAULT_CACHE_DIR)))
    model_path = cache_dir / file

    if model_path.is_file() and model_path.stat().st_size > 0:
        return model_path

    if not repo or not file:
        _err("Model not found and PROJECT_CODE_INTELLIGENCE_HF_MODEL_REPO/FILE are not set.")
        _err("Set PROJECT_CODE_INTELLIGENCE_LLAMA_MODEL=/path/to/model.gguf.")
        sys.exit(1)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_url = f"https://huggingface.co/{repo}/resolve/main/{file}?download=true"
    _err(f"Downloading embedding model (one-time setup): {repo}/{file}")

    curl = shutil.which("curl")
    if not curl:
        _err("curl is required to download the model but was not found on PATH.")
        sys.exit(1)

    hf_token = _env("HF_TOKEN")
    cmd: list[str] = [curl, "-fL", "--retry", "3", "--retry-delay", "2", "-o", str(model_path)]
    if hf_token:
        cmd += ["-H", f"Authorization: Bearer {hf_token}"]
    cmd.append(model_url)

    _ = process.run(cmd, process.RunOptions(check=True))
    return model_path


def _build_server_cmd(server: str, model: Path) -> list[str]:
    cmd: list[str] = [
        server,
        "--embedding",
        "--model",
        str(model),
        "--pooling",
        "last",
        "--host",
        _env("PROJECT_CODE_INTELLIGENCE_EMBEDDING_HOST", "127.0.0.1"),
        "--port",
        _env("PROJECT_CODE_INTELLIGENCE_EMBEDDING_PORT", "18081"),
        "--ctx-size",
        _env("PROJECT_CODE_INTELLIGENCE_LLAMA_SERVER_CTX", "40960"),
        "--batch-size",
        _env("PROJECT_CODE_INTELLIGENCE_LLAMA_SERVER_BATCH", "2048"),
        "--ubatch-size",
        _env("PROJECT_CODE_INTELLIGENCE_LLAMA_SERVER_UBATCH", "1024"),
    ]
    if parallel := _env("PROJECT_CODE_INTELLIGENCE_LLAMA_SERVER_PARALLEL", "8"):
        cmd += ["--parallel", parallel]
    if n_gpu := _env("PROJECT_CODE_INTELLIGENCE_LLAMA_SERVER_N_GPU_LAYERS"):
        cmd += ["--n-gpu-layers", n_gpu]
    if _env("PROJECT_CODE_INTELLIGENCE_LLAMA_SERVER_KV_UNIFIED", "1") == "1":
        cmd.append("--kv-unified")
    if _env("PROJECT_CODE_INTELLIGENCE_LLAMA_SERVER_CACHE_PROMPT", "0") == "0":
        cmd.append("--no-cache-prompt")
    if cache_ram := _env("PROJECT_CODE_INTELLIGENCE_LLAMA_SERVER_CACHE_RAM", "0"):
        cmd += ["--cache-ram", cache_ram]
    if _env("PROJECT_CODE_INTELLIGENCE_LLAMA_SERVER_CACHE_IDLE_SLOTS", "0") == "0":
        cmd.append("--no-cache-idle-slots")
    if _env("PROJECT_CODE_INTELLIGENCE_LLAMA_SERVER_WEBUI", "0") == "0":
        cmd.append("--no-webui")
    if log_verbosity := _env("PROJECT_CODE_INTELLIGENCE_LLAMA_SERVER_LOG_VERBOSITY"):
        cmd += ["--log-verbosity", log_verbosity]
    if extra := _env("PROJECT_CODE_INTELLIGENCE_LLAMA_SERVER_EXTRA_ARGS"):
        cmd += extra.split()
    cmd.append("--no-warmup")
    return cmd


def main() -> None:
    server = _resolve_llama_server()

    pid = _read_pid(LLAMA_SERVER_PID_FILE)
    if pid is not None and _is_running(pid):
        _err(f"pci-apple-llama-server already running (PID {pid}).")
        return

    model = _ensure_model()
    cmd = _build_server_cmd(server, model)

    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LLAMA_SERVER_LOG_FILE.open("a") as log_file:
        proc = process.popen(
            cmd,
            PopenOptions(
                stdout=log_file,
                stderr=log_file,
                stdin=process.DEVNULL,
                start_new_session=True,
            ),
        )
    _ = LLAMA_SERVER_PID_FILE.write_text(str(proc.pid) + "\n")
    _err(f"llama-server started (PID {proc.pid}). Log: {LLAMA_SERVER_LOG_FILE}")
