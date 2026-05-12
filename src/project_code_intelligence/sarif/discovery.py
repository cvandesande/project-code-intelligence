"""SARIF file discovery helpers."""

from __future__ import annotations

from pathlib import Path

from project_code_intelligence import config, profile_context
from project_code_intelligence.sarif.paths import relative_to_or_none


def glob_pattern_paths(root: Path, pattern: str) -> list[Path]:
    pattern_path = Path(pattern)
    if not any(char in pattern for char in "*?["):
        return [pattern_path if pattern_path.is_absolute() else root / pattern_path]
    if pattern_path.is_absolute():
        anchor = Path(pattern_path.anchor)
        return list(anchor.glob(str(pattern_path.relative_to(anchor))))
    return list(root.glob(pattern))


def explicit_sarif_patterns(values: list[str] | None) -> list[str]:
    patterns: list[str] = []
    for value in values or []:
        patterns.extend(item.strip() for item in value.split(",") if item.strip())
    env_value = config.env_text("PROJECT_CODE_INTELLIGENCE_SARIF")
    if env_value:
        patterns.extend(item.strip() for item in env_value.split(",") if item.strip())
    return patterns


def discover_sarif_files(
    root: Path, repos: list[str], explicit_patterns: list[str], *, include_profile: bool
) -> list[Path]:
    patterns = list(explicit_patterns)
    if include_profile:
        patterns.extend(profile_context.active_profile.sarif_globs(repos))
    found: dict[str, Path] = {}
    for pattern in patterns:
        for path in glob_pattern_paths(root, pattern):
            if path.is_file() and (path.name.endswith(".sarif") or path.name.endswith(".sarif.json")):
                found[str(path.resolve())] = path.resolve()
    return [found[key] for key in sorted(found)]


def repo_for_sarif_file(root: Path, repos: list[str], path: Path) -> str | None:
    for repo in repos:
        repo_root = root if repo == "." else root / repo
        if relative_to_or_none(path, repo_root):
            return repo
    return None
