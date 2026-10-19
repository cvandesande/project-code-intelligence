"""Small shared helpers for code-intelligence ingestion."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import traceback
from pathlib import Path
from typing import cast


class _StderrLog:
    """Writes to stderr so an unexpected boundary failure isn't silent."""

    @staticmethod
    def exception(message: str) -> None:
        _ = sys.stderr.write(f"{message}\n{traceback.format_exc()}")
        _ = sys.stderr.flush()


log = _StderrLog()


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


REINDEX_MARKER_NAME = "pci-reindex.json"


def read_reindex_marker(git_dir: Path) -> tuple[Path, list[str], str | None] | None:
    """Validated (cwd, repo_paths, collection) from a reindex marker in ``git_dir``,
    or None on any I/O, parse, or shape problem. Every ``pci index`` run writes this
    marker into each indexed repo's ``.git`` directory (see
    ``hooks.runtime.write_reindex_markers``), so it is the durable record of which
    workspace invocation a checkout belongs to."""
    try:
        data = cast("object", json.loads((git_dir / REINDEX_MARKER_NAME).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    marker = cast("dict[object, object]", data)
    cwd = marker.get("cwd")
    raw = marker.get("repo_paths")
    if not (isinstance(cwd, str) and Path(cwd).is_dir() and isinstance(raw, list)):
        return None
    raw_paths = cast("list[object]", raw)
    paths = [p for p in raw_paths if isinstance(p, str)]
    if not paths or len(paths) != len(raw_paths):
        return None
    # Deleted member repos stay in the list: the single-vs-multi-repo shape of
    # the ORIGINAL invocation decides the database scope, so dropping a deleted
    # path would silently move the surviving repos to a different database.
    # Callers that touch the checkouts filter with `existing_repo_paths`.
    if not any(Path(p).is_dir() for p in paths):
        return None
    collection = marker.get("collection")
    return Path(cwd), paths, collection if isinstance(collection, str) else None


def existing_repo_paths(paths: list[str]) -> list[str]:
    """The marker repo paths that still exist on disk (member repos can be deleted)."""
    return [p for p in paths if Path(p).is_dir()]


def workspace_marker_for_path(path: Path) -> tuple[Path, list[str], str | None] | None:
    """Reindex marker of the repo enclosing ``path``, or None.

    Walks up to the nearest ``.git`` directory only: a checkout nested inside a
    multi-repo workspace carries its own marker recording the workspace invocation,
    which is how a child directory resolves to the parent workspace's database and
    collection instead of inferring a fresh standalone identity from its own name.
    A ``.git`` file (linked worktree) stops the walk without a marker -- worktrees
    are resolved through their main repo by the reindex hook, not here.
    """
    resolved = path.expanduser().resolve(strict=False)
    for candidate in (resolved, *resolved.parents):
        git_entry = candidate / ".git"
        if git_entry.is_dir():
            return read_reindex_marker(git_entry)
        if git_entry.exists():
            return None
    return None


def database_scope_path_for_marker(cwd: Path, repo_paths: list[str]) -> Path:
    """Scope path a marker's invocation would have inferred at index time.

    Mirrors ``database_scope_path_for_root_repos``/CLI inference: a single-repo
    invocation scopes to that repo's own path; a multi-repo workspace scopes to
    the invocation directory.
    """
    if len(repo_paths) == 1:
        return Path(repo_paths[0]).expanduser().resolve(strict=False)
    return cwd.expanduser().resolve(strict=False)


def database_scope_path_for_root_repos(root: Path, repos: list[str]) -> Path:
    resolved_root = root.expanduser().resolve(strict=False)
    if len(repos) == 1 and repos[0] != ".":
        return (resolved_root / repos[0]).resolve(strict=False)
    return resolved_root


def last_symbol_component(symbol: str) -> str:
    """Bare name of a possibly qualified symbol: ``DatabaseSettings.from_env`` -> ``from_env``.

    Call sites record bare names while parsers store qualified ones, so every
    name-based join between the two needs this projection.
    """
    return symbol.rpartition(".")[2]


def source_path_for(repo: str, rel_path: str) -> str:
    return rel_path if repo == "." else f"{repo}/{rel_path}"


def repo_relative_path(source_path: str | None, repo: str | None) -> str | None:
    """Inverse of `source_path_for`: strip the repo prefix from a stored source_path.

    Returns a path relative to the repo root — what a consumer with cwd at the repo
    root would pass to `Read`/`open`. When repo is None, ".", or empty (workspace ==
    repo root indexing), the source_path is already repo-relative and returned unchanged.
    """
    if not source_path:
        return source_path
    if not repo or repo == ".":
        return source_path
    prefix = f"{repo}/"
    if source_path.startswith(prefix):
        return source_path[len(prefix) :]
    return source_path


def repo_for_source_path(source_path: str | None, repos: list[str], default_repo: str | None = None) -> str | None:
    if not source_path:
        return default_repo
    for repo in repos:
        if repo != "." and (source_path == repo or source_path.startswith(f"{repo}/")):
            return repo
    if repos == ["."] and default_repo is None:
        return "."
    return default_repo
