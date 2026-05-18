#!/usr/bin/env python3
"""Embed stdin text with llama.cpp and print one JSON vector.

This wrapper is intentionally separate from the ingester. llama.cpp command
line flags and output formats change over time, so the code-intelligence ingest
path only depends on this stable contract:

    stdin text -> stdout JSON array of floats
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

from project_code_intelligence import config, process

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonValue


def configured_llama_dir() -> Path | None:
    value = os.environ.get("PCI_LLAMA_CPP_DIR")
    return Path(value) if value else None


def llama_dir() -> Path:
    configured = configured_llama_dir()
    if configured is not None:
        return configured
    binary = llama_embedding_binary()
    if binary.is_absolute():
        return binary.parent
    return Path()


def resolve_executable(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.parent != Path():
        return path
    found = shutil.which(value)
    return Path(found) if found else path


def llama_embedding_binary() -> Path:
    configured_binary = os.environ.get("PCI_LLAMA_EMBEDDING_BIN")
    if configured_binary:
        return resolve_executable(configured_binary)
    configured_dir = configured_llama_dir()
    if configured_dir is not None:
        return configured_dir / "llama-embedding"
    return resolve_executable("llama-embedding")


def llama_timeout_seconds() -> int:
    return config.env_int("PCI_LLAMA_TIMEOUT_SECONDS", 3600, minimum=1)


def build_command(prompt_file: Path) -> list[str]:
    binary = llama_embedding_binary()
    model = os.environ.get("PCI_LLAMA_MODEL")
    use_default_gemma = os.environ.get("PCI_LLAMA_EMBD_GEMMA_DEFAULT") == "1"
    if not model and not use_default_gemma:
        raise RuntimeError(
            "Set PCI_LLAMA_MODEL=/path/to/embedding-model.gguf, "
            "or set PCI_LLAMA_EMBD_GEMMA_DEFAULT=1 to let llama.cpp "
            "use its default EmbeddingGemma model."
        )

    command = [
        str(binary),
        "--embd-output-format",
        "json",
        "-f",
        str(prompt_file),
    ]
    if model:
        command.extend(["--model", model])
    elif use_default_gemma:
        command.append("--embd-gemma-default")

    extra = os.environ.get("PCI_LLAMA_EXTRA_ARGS")
    if extra:
        command.extend(shlex.split(extra))
    return command


def extract_embedding(payload: object) -> list[float]:
    values: object
    if isinstance(payload, list):
        payload_list = cast("list[object]", payload)
        if payload_list and isinstance(payload_list[0], list):
            values = cast("list[object]", payload_list[0])
        else:
            values = payload_list
    elif isinstance(payload, dict):
        payload_obj = cast("dict[str, object]", payload)
        data = payload_obj.get("data")
        if isinstance(data, list) and data:
            data_items = cast("list[object]", data)
            first = data_items[0]
            values = cast("dict[str, object]", first).get("embedding") if isinstance(first, dict) else first
        elif "embedding" in payload:
            values = payload_obj["embedding"]
        else:
            raise ValueError("embedding JSON did not contain data[0].embedding")
    else:
        raise TypeError("embedding output must be JSON object or array")

    if not isinstance(values, list):
        raise TypeError("embedding vector must be a list")
    if not values:
        raise ValueError("embedding vector is empty or not a list")
    out: list[float] = []
    for value in cast("list[object]", values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("embedding vector contains non-numeric values")
        out.append(float(value))
    return out


def parse_llama_stdout(stdout: str) -> JsonValue:
    text = stdout.strip()
    if not text:
        raise ValueError("llama-embedding returned empty stdout")

    decoder = json.JSONDecoder()
    candidates = [idx for idx, char in enumerate(text) if char in "[{"]
    last_error: Exception | None = None
    for idx in candidates:
        try:
            payload, end = cast("tuple[object, int]", decoder.raw_decode(text[idx:]))
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if text[idx + end :].strip():
            continue
        return cast("JsonValue", payload)
    if last_error:
        raise last_error
    raise ValueError("could not find JSON payload in llama-embedding stdout")


def embed_text(text: str) -> list[float]:
    if not text.strip():
        raise ValueError("stdin text is empty")

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt") as prompt:
        _ = prompt.write(text)
        _ = prompt.flush()
        command = build_command(Path(prompt.name))
        env = os.environ.copy()
        binary = Path(command[0])
        lib_dir = configured_llama_dir() or (binary.parent if binary.is_absolute() else None)
        if lib_dir is not None:
            lib_path = str(lib_dir)
            env["LD_LIBRARY_PATH"] = (
                lib_path if not env.get("LD_LIBRARY_PATH") else lib_path + ":" + env["LD_LIBRARY_PATH"]
            )
        proc = process.run(
            command,
            process.RunOptions(
                check=True,
                capture_output=True,
                env=env,
                timeout=llama_timeout_seconds(),
            ),
        )

    payload = parse_llama_stdout(proc.stdout)
    return extract_embedding(payload)


def main() -> int:
    try:
        vector = embed_text(sys.stdin.read())
    except (json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    _ = sys.stdout.write(json.dumps(vector, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
