"""Shared local embedding helpers for code-intelligence tooling."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import cast

from project_code_intelligence import db, process
from project_code_intelligence.embedding import llama


def llama_batch_embeddings(texts: list[str], batch_size: int) -> list[str]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    vectors: list[str] = []
    separator = "\n<#project-code-intelligence-embedding-separator#>\n"
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        for text in batch:
            if separator.strip() in text:
                raise ValueError("chunk content contains llama embedding separator")
        payload = separator.join(batch)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt") as prompt:
            _ = prompt.write(payload)
            _ = prompt.flush()
            command = llama.build_command(Path(prompt.name))
            command.extend(["--embd-separator", separator.strip()])
            env = os.environ.copy()
            lib_path = str(llama.llama_dir())
            env["LD_LIBRARY_PATH"] = (
                lib_path if not env.get("LD_LIBRARY_PATH") else lib_path + ":" + env["LD_LIBRARY_PATH"]
            )
            try:
                proc = process.run(
                    command,
                    process.RunOptions(
                        check=True,
                        capture_output=True,
                        env=env,
                        timeout=llama.llama_timeout_seconds(),
                    ),
                )
            except process.CalledProcessError as exc:
                stderr = cast("object", exc.stderr)
                stderr_tail = (stderr if isinstance(stderr, str) else "").strip()[-1200:]
                raise RuntimeError(f"llama.cpp embedding failed: {stderr_tail}") from exc
        parsed = llama.parse_llama_stdout(proc.stdout)
        data = parsed.get("data") if isinstance(parsed, dict) else None
        if not isinstance(data, list) or len(data) != len(batch):
            raise ValueError(
                f"llama.cpp returned {0 if not isinstance(data, list) else len(data)} "
                f"embeddings for a batch of {len(batch)} chunks"
            )
        vectors.extend(db.vector_literal(llama.extract_embedding(item)) for item in data)
    return vectors
