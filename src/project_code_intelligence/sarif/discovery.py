"""SARIF file discovery helpers."""

from __future__ import annotations

from pathlib import Path

from project_code_intelligence import config, profile_context
from project_code_intelligence.sarif.paths import relative_to_or_none

SARIF_FIXTURE_MARKERS = frozenset({"fixture", "fixtures", "test", "tests"})


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


def is_sarif_file(path: Path) -> bool:
    return path.is_file() and (path.name.endswith(".sarif") or path.name.endswith(".sarif.json"))


def is_probable_sarif_fixture(path: Path) -> bool:
    lower_parts = {part.lower() for part in path.parts}
    if not lower_parts.intersection(SARIF_FIXTURE_MARKERS):
        return False
    name = path.name.lower()
    return name.endswith(("-expected.sarif", "-expected.sarif.json"))


def discover_sarif_files(
    root: Path, repos: list[str], explicit_patterns: list[str], *, include_profile: bool
) -> list[Path]:
    patterns = [(pattern, True) for pattern in explicit_patterns]
    if include_profile:
        patterns.extend((pattern, False) for pattern in profile_context.active_profile.sarif_globs(repos))
    found: dict[str, Path] = {}
    for pattern, explicit in patterns:
        for path in glob_pattern_paths(root, pattern):
            if is_sarif_file(path) and (explicit or not is_probable_sarif_fixture(path)):
                found[str(path.resolve())] = path.resolve()
    return [found[key] for key in sorted(found)]


def repo_for_sarif_file(root: Path, repos: list[str], path: Path) -> str | None:
    for repo in repos:
        repo_root = root if repo == "." else root / repo
        if relative_to_or_none(path, repo_root):
            return repo
    return None
