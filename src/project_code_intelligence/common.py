"""Small shared helpers for code-intelligence ingestion."""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_repos(value: str) -> list[str]:
    return [item.strip().rstrip("/") or "." for item in value.split(",") if item.strip()]


def default_collection(root: Path) -> str:
    name = root.name.strip()
    return name or "default"


def default_database_name(scope_path: Path) -> str:
    resolved = scope_path.expanduser().resolve(strict=False)
    digest = sha256_text(str(resolved))[:8]
    slug = re.sub(r"[^a-z0-9]+", "_", resolved.name.lower()).strip("_") or "default"
    max_slug_chars = 63 - len("pci_") - len("_") - len(digest)
    slug = slug[:max_slug_chars].rstrip("_") or "default"
    return f"pci_{slug}_{digest}"


def database_scope_path_for_root_repos(root: Path, repos: list[str]) -> Path:
    resolved_root = root.expanduser().resolve(strict=False)
    if len(repos) == 1 and repos[0] != ".":
        return (resolved_root / repos[0]).resolve(strict=False)
    return resolved_root


def source_path_for(repo: str, rel_path: str) -> str:
    return rel_path if repo == "." else f"{repo}/{rel_path}"


def repo_for_source_path(source_path: str | None, repos: list[str], default_repo: str | None = None) -> str | None:
    if not source_path:
        return default_repo
    for repo in repos:
        if repo != "." and (source_path == repo or source_path.startswith(f"{repo}/")):
            return repo
    if repos == ["."] and default_repo is None:
        return "."
    return default_repo
