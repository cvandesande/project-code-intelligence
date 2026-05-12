"""SARIF URI and workspace path resolution."""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from project_code_intelligence.common import repo_for_source_path, source_path_for
from project_code_intelligence.models import SarifPathResolution


@dataclass(frozen=True)
class SarifPathContext:
    root: Path
    repos: list[str]
    default_repo: str | None
    uri_base_ids: dict[str, str]
    known_source_paths: set[str] | None = None


def combine_sarif_base_uri(base_uri: str, uri: str) -> str:
    parsed_base = urllib.parse.urlparse(base_uri)
    if parsed_base.scheme == "file":
        base_path = Path(urllib.parse.unquote(parsed_base.path))
        return str(base_path / uri)
    if parsed_base.scheme:
        return urllib.parse.urljoin(base_uri, uri)
    base = Path(normalize_sarif_uri(base_uri))
    if base.is_absolute():
        return str(base / uri)
    base_text = normalize_sarif_uri(base_uri).rstrip("/")
    if not base_text:
        return uri
    return f"{base_text}/{uri}"


def resolve_sarif_uri_base(uri: str, uri_base_id: str | None, uri_base_ids: dict[str, str]) -> str:
    if not uri_base_id:
        return uri
    base_uri = uri_base_ids.get(uri_base_id)
    if not base_uri:
        return uri
    return combine_sarif_base_uri(base_uri, uri)


def source_path_known_or_exists(
    root: Path,
    repo: str,
    rel_path: str,
    known_source_paths: set[str] | None,
) -> bool:
    rel = Path(rel_path)
    if rel.is_absolute() or ".." in rel.parts:
        return False
    source_path = source_path_for(repo, rel_path)
    if known_source_paths and source_path in known_source_paths:
        return True
    repo_root = root if repo == "." else root / repo
    return (repo_root / rel_path).exists()


def source_path_for_existing_repo_path(
    root: Path,
    repos: list[str],
    rel_path: str,
    known_source_paths: set[str] | None,
) -> tuple[str | None, str | None]:
    for repo in repos:
        if source_path_known_or_exists(root, repo, rel_path, known_source_paths):
            return source_path_for(repo, rel_path), repo
    return None, None


def path_mapping_for_workspace_path(root: Path, source_path: str, known_source_paths: set[str] | None) -> str:
    if known_source_paths and source_path in known_source_paths:
        return "indexed_source"
    if (root / source_path).exists():
        return "workspace_relative"
    return "unresolved_relative"


def path_mapping_for_repo_path(
    root: Path,
    repo: str,
    rel_path: str,
    known_source_paths: set[str] | None,
) -> str:
    source_path = source_path_for(repo, rel_path)
    if known_source_paths and source_path in known_source_paths:
        return "indexed_source"
    repo_root = root if repo == "." else root / repo
    if (repo_root / rel_path).exists():
        return "existing_repo_file"
    return "unresolved_relative"


def normalize_sarif_uri(uri: str) -> str:
    value = urllib.parse.unquote(uri).replace("\\", "/")
    if value.startswith("file://"):
        value = urllib.parse.urlparse(value).path
    while value.startswith("./"):
        value = value[2:]
    return value


def relative_to_or_none(path: Path, base: Path) -> str | None:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return None


def resolve_absolute_sarif_source_path(context: SarifPathContext, normalized: str) -> SarifPathResolution:
    candidate = Path(normalized)
    rel_workspace = relative_to_or_none(candidate, context.root)
    if rel_workspace:
        repo = repo_for_source_path(rel_workspace, context.repos, context.default_repo)
        return SarifPathResolution(
            rel_workspace,
            repo,
            path_mapping_for_workspace_path(context.root, rel_workspace, context.known_source_paths),
        )
    for repo in context.repos:
        repo_root = context.root if repo == "." else context.root / repo
        rel_repo = relative_to_or_none(candidate, repo_root)
        if rel_repo:
            return SarifPathResolution(
                source_path_for(repo, rel_repo),
                repo,
                path_mapping_for_repo_path(context.root, repo, rel_repo, context.known_source_paths),
            )
    return SarifPathResolution(normalized, context.default_repo, "external_absolute")


def resolve_relative_sarif_source_path(context: SarifPathContext, normalized: str) -> SarifPathResolution:
    normalized = normalized.lstrip("/")
    repo = repo_for_source_path(normalized, context.repos, None)
    if repo:
        return SarifPathResolution(
            normalized,
            repo,
            path_mapping_for_workspace_path(context.root, normalized, context.known_source_paths),
        )
    source_path, existing_repo = source_path_for_existing_repo_path(
        context.root,
        context.repos,
        normalized,
        context.known_source_paths,
    )
    if source_path and existing_repo:
        return SarifPathResolution(
            source_path,
            existing_repo,
            path_mapping_for_repo_path(context.root, existing_repo, normalized, context.known_source_paths),
        )
    if context.default_repo and source_path_known_or_exists(
        context.root,
        context.default_repo,
        normalized,
        context.known_source_paths,
    ):
        return SarifPathResolution(
            source_path_for(context.default_repo, normalized),
            context.default_repo,
            path_mapping_for_repo_path(context.root, context.default_repo, normalized, context.known_source_paths),
        )
    repo = context.default_repo or repo_for_source_path(normalized, context.repos, context.default_repo)
    return SarifPathResolution(normalized, repo, "unresolved_relative")


def resolve_sarif_source_path(
    context: SarifPathContext,
    uri: str | None,
    uri_base_id: str | None = None,
) -> SarifPathResolution:
    if not uri:
        return SarifPathResolution(None, context.default_repo, "missing_uri")
    normalized = normalize_sarif_uri(resolve_sarif_uri_base(uri, uri_base_id, context.uri_base_ids))
    if Path(normalized).is_absolute():
        return resolve_absolute_sarif_source_path(context, normalized)
    return resolve_relative_sarif_source_path(context, normalized)


def source_path_from_sarif_uri(
    context: SarifPathContext,
    uri: str | None,
    uri_base_id: str | None = None,
) -> tuple[str | None, str | None]:
    resolution = resolve_sarif_source_path(context, uri, uri_base_id=uri_base_id)
    return resolution.source_path, resolution.repo
