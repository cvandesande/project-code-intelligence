"""SARIF file discovery helpers."""

from __future__ import annotations

import os
from pathlib import Path

from project_code_intelligence import config, process, profile_context
from project_code_intelligence.git_utils import GIT_TIMEOUT_SECONDS, git_binary
from project_code_intelligence.sarif.paths import relative_to_or_none

SARIF_FIXTURE_MARKERS = frozenset({"fixture", "fixtures", "test", "tests"})
PROFILE_SARIF_PRUNE_DIRS = frozenset({
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build_dir",
    "dl",
    "node_modules",
    "staging_dir",
    "target",
    "tmp",
    "venv",
})


def glob_pattern_paths(root: Path, pattern: str) -> list[Path]:
    pattern_path = Path(pattern)
    if not any(char in pattern for char in "*?["):
        return [pattern_path if pattern_path.is_absolute() else root / pattern_path]
    if pattern_path.is_absolute():
        anchor = Path(pattern_path.anchor)
        return list(anchor.glob(str(pattern_path.relative_to(anchor))))
    return list(root.glob(pattern))


def profile_recursive_sarif_base(root: Path, pattern: str) -> tuple[Path, str] | None:
    suffix = ""
    if pattern.endswith("/**/*.sarif"):
        suffix = ".sarif"
        prefix = pattern.removesuffix("/**/*.sarif")
    elif pattern.endswith("/**/*.sarif.json"):
        suffix = ".sarif.json"
        prefix = pattern.removesuffix("/**/*.sarif.json")
    elif pattern == "**/*.sarif":
        suffix = ".sarif"
        prefix = ""
    elif pattern == "**/*.sarif.json":
        suffix = ".sarif.json"
        prefix = ""
    else:
        return None
    base = Path(prefix)
    return (base if base.is_absolute() else root / base, suffix)


def _git_listed_paths_with_suffix(base: Path, suffix: str) -> list[Path] | None:
    """Enumerate tracked + untracked files under `base` via git, returning those
    that end with `suffix`. Returns None when `base` is not inside a git working
    tree, the git binary is missing, or the git invocation fails — caller should
    then fall back to filesystem walking.

    `--exclude-standard` is intentionally omitted from the untracked pass: SARIF
    reports are commonly written into gitignored output directories (`out/`,
    `reports/`, `build/`), and we want to discover them anyway. Git's tree
    walker is still significantly faster than Python's `os.walk` because it is
    written in C and emits paths as strings without per-entry Path allocation.
    """
    binary = git_binary()
    if binary is None:
        return None
    paths: list[Path] = []
    for extra in (["ls-files"], ["ls-files", "--others"]):
        try:
            proc = process.run(
                [binary, "-C", str(base), *extra],
                process.RunOptions(check=True, capture_output=True, timeout=GIT_TIMEOUT_SECONDS),
            )
        except (OSError, process.CalledProcessError, process.TimeoutExpired):
            return None
        paths.extend(base / line for line in proc.stdout.splitlines() if line.endswith(suffix))
    return paths


def profile_recursive_sarif_paths(root: Path, pattern: str) -> list[Path] | None:
    recursive = profile_recursive_sarif_base(root, pattern)
    if recursive is None:
        return None
    base, suffix = recursive
    if not base.is_dir():
        return []
    git_paths = _git_listed_paths_with_suffix(base, suffix)
    if git_paths is not None:
        return git_paths
    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [name for name in dirnames if name not in PROFILE_SARIF_PRUNE_DIRS]
        current = Path(dirpath)
        paths.extend(current / name for name in filenames if name.endswith(suffix))
    return paths


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
        paths = None if explicit else profile_recursive_sarif_paths(root, pattern)
        for path in paths if paths is not None else glob_pattern_paths(root, pattern):
            if is_sarif_file(path) and (explicit or not is_probable_sarif_fixture(path)):
                found[str(path.resolve())] = path.resolve()
    return [found[key] for key in sorted(found)]


def repo_for_sarif_file(root: Path, repos: list[str], path: Path) -> str | None:
    for repo in repos:
        repo_root = root if repo == "." else root / repo
        if relative_to_or_none(path, repo_root):
            return repo
    return None
