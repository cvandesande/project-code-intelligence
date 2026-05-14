"""Git inventory and file classification for code-intelligence ingestion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING

from project_code_intelligence import process, profile_context
from project_code_intelligence.common import sha256_bytes, sha256_text, source_path_for
from project_code_intelligence.git_utils import GIT_TIMEOUT_SECONDS, git_binary, run_git
from project_code_intelligence.language_profiles import language_has_metadata, language_metadata_for_file
from project_code_intelligence.models import (
    BINARY_SUFFIXES,
    CHUNKER_VERSION,
    PARSER_VERSION,
    SCHEMA_VERSION,
    SOURCE_LANGUAGES,
    TEXT_NAMES,
    TEXT_SUFFIXES,
    IntelFile,
    JsonObject,
    PreviousFileState,
    Snapshot,
)
from project_code_intelligence.profile_context import repo_role_for

if TYPE_CHECKING:
    from collections.abc import Mapping

C_LANGUAGE_SUFFIXES = frozenset({".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx"})
ASM_SUFFIXES = frozenset({".s", ".S"})
KCONFIG_NAMES = frozenset({"Kconfig", "Config.in", "Config-defaults.in"})
CONFIG_NAMES = frozenset({"config", "inittab"})
CONFIG_SUFFIXES = frozenset({
    ".cfg",
    ".cnf",
    ".conf",
    ".config",
    ".default",
    ".defaults",
    ".init",
    ".seed",
    ".service",
})
SHELL_SUFFIXES = frozenset({
    ".common",
    ".failsafe",
    ".hotplug",
    ".initd",
    ".local",
    ".script",
    ".usb",
    ".usbmisc",
    ".user",
})
BOOT_SCRIPT_SUFFIXES = frozenset({".bootscript", ".scr"})
LINKER_SCRIPT_SUFFIXES = frozenset({".ld", ".lds"})
PORCELAIN_STATUS_MIN_LENGTH = 4
PORCELAIN_STATUS_PATH_OFFSET = 3
BUILD_FILE_NAMES = frozenset({
    ".bazelrc",
    "cargo.toml",
    "build.gradle",
    "build.gradle.kts",
    "go.mod",
    "go.sum",
    "jenkinsfile",
    "pyproject.toml",
    "setup.py",
    "settings.gradle",
    "settings.gradle.kts",
    "build.sbt",
    "package.json",
    "tsconfig.json",
})
DOCKERFILE_NAMES = frozenset({"Dockerfile", "Containerfile"})
CMAKE_NAMES = frozenset({"CMakeLists.txt"})
MESON_NAMES = frozenset({"meson.build", "meson_options.txt"})
BAZEL_NAMES = frozenset({"BUILD", "BUILD.bazel", "WORKSPACE", "WORKSPACE.bazel", "MODULE.bazel"})
CONTENT_CLASS_FLAG_ORDER = (
    ("is_doc", "doc"),
    ("is_build", "build"),
    ("is_config", "config"),
    ("is_test", "test"),
    ("is_vendor", "vendor"),
    ("is_source", "source"),
    ("is_generated", "generated"),
)

LANGUAGE_SUFFIXES = (
    {
        ".rs": "rust",
        ".go": "go",
        ".py": "python",
        ".java": "java",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".cs": "csharp",
        ".swift": "swift",
        ".m": "objective_c",
        ".mm": "objective_cpp",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".scss": "scss",
        ".sass": "scss",
        ".vue": "vue",
        ".svelte": "svelte",
        ".graphql": "graphql",
        ".gql": "graphql",
        ".bzl": "starlark",
        ".star": "starlark",
        ".groovy": "groovy",
        ".gradle": "groovy",
        ".ps1": "powershell",
        ".psm1": "powershell",
        ".psd1": "powershell",
        ".scala": "scala",
        ".sbt": "scala",
        ".ex": "elixir",
        ".exs": "elixir",
        ".erl": "erlang",
        ".hrl": "erlang",
        ".zig": "zig",
        ".php": "php",
        ".rb": "ruby",
        ".xml": "xml",
        ".sql": "sql",
        ".proto": "protobuf",
        ".tf": "terraform",
        ".tfvars": "terraform",
        ".hcl": "terraform",
        ".cmake": "cmake",
        ".pl": "perl",
        ".pm": "perl",
        ".awk": "awk",
        ".l": "lex",
        ".y": "yacc",
        ".mk": "make",
        ".in": "kconfig",
        ".dts": "dts",
        ".dtsi": "dts",
        ".dtso": "dts",
        ".patch": "patch",
        ".diff": "patch",
        ".json": "json",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".lua": "lua",
        ".uc": "ucode",
        ".md": "doc",
        ".rst": "doc",
    }
    | dict.fromkeys((".m4", ".am", ".ac"), "autotools")
    | dict.fromkeys(SHELL_SUFFIXES | {".sh"}, "shell")
    | dict.fromkeys(LINKER_SCRIPT_SUFFIXES, "linker_script")
    | dict.fromkeys(BOOT_SCRIPT_SUFFIXES, "boot_script")
    | dict.fromkeys(CONFIG_SUFFIXES, "config")
)


@dataclass(frozen=True)
class DiscoveryReuse:
    previous_files: Mapping[str, PreviousFileState]
    reuse_unchanged_blobs: bool = False
    dirty_paths: frozenset[str] = frozenset()


def run_git_required(root: Path, args: list[str]) -> str:
    value = run_git(root, args)
    if value is None:
        raise ValueError(f"git {' '.join(args)} failed in {root}; pass --root pointing at a Git checkout")
    return value


def git_dirty_fingerprint(root: Path) -> str | None:
    status = run_git(root, ["status", "--porcelain=v1"])
    if not status:
        return None
    staged = run_git(root, ["diff", "--cached", "--binary"]) or ""
    unstaged = run_git(root, ["diff", "--binary"]) or ""
    return sha256_text(status + "\n" + staged + "\n" + unstaged)


def git_dirty_paths(root: Path) -> set[str]:
    status = run_git(root, ["status", "--porcelain=v1"]) or ""
    dirty_paths: set[str] = set()
    for line in status.splitlines():
        if len(line) < PORCELAIN_STATUS_MIN_LENGTH:
            continue
        path_text = line[PORCELAIN_STATUS_PATH_OFFSET:]
        if " -> " in path_text:
            old_path, new_path = path_text.split(" -> ", 1)
            dirty_paths.add(old_path)
            dirty_paths.add(new_path)
        else:
            dirty_paths.add(path_text)
    return dirty_paths


def git_ls_files(repo_root: Path) -> list[tuple[str | None, str]]:
    binary = git_binary()
    if binary is None:
        raise FileNotFoundError("git was not found on PATH")
    proc = process.run(
        [binary, "ls-files", "-s"],
        process.RunOptions(
            cwd=repo_root,
            check=True,
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
        ),
    )
    out: list[tuple[str | None, str]] = []
    for line in proc.stdout.splitlines():
        match = re.match(r"^\d+\s+([0-9a-f]{40,64})\s+\d+\t(.+)$", line)
        if match:
            out.append((match.group(1), match.group(2)))
        elif line:
            out.append((None, line.rsplit(None, 1)[-1]))
    return out


def git_untracked_files(repo_root: Path) -> list[str]:
    binary = git_binary()
    if binary is None:
        return []
    proc = process.run(
        [binary, "ls-files", "--others", "--exclude-standard"],
        process.RunOptions(
            cwd=repo_root,
            check=False,
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
        ),
    )
    return [line for line in proc.stdout.splitlines() if line]


def shebang_language(data: bytes) -> str | None:
    first_line = data.splitlines()[:1]
    if not first_line:
        return None
    line = first_line[0].decode("utf-8", errors="replace").strip().lower()
    if not line.startswith("#!"):
        return None
    language: str | None = None
    if "perl" in line:
        language = "perl"
    elif "pwsh" in line or "powershell" in line:
        language = "powershell"
    elif "ucode" in line:
        language = "ucode"
    elif "bash" in line or "ash" in line or re.search(r"(?:^|/|\s)sh(?:\s|$)", line):
        language = "shell"
    return language


def language_for_read_file(path: str, data: bytes, *, read_ok: bool) -> str:
    language = profile_context.active_profile.language_for_path(path) or language_for(path)
    if read_ok and language == "c" and Path(path).suffix.lower() == ".h":
        header = data[:8192]
        if b"@interface" in header or b"@protocol" in header or b"@implementation" in header or b"#import" in header:
            return "objective_c"
    if read_ok and language == "text":
        return shebang_language(data) or language
    return language


def is_config_name(name: str) -> bool:
    return name in CONFIG_NAMES or name.startswith("Config.") or bool(re.match(r"^config-\d+(?:\.\d+)*$", name))


def is_shell_path(path: str) -> bool:
    return "/init.d/" in path or "/preinit/" in path


def special_name_language(path: str, name: str) -> str | None:
    path_lower = path.lower()
    name_lower = name.lower()
    language: str | None = None
    if name in DOCKERFILE_NAMES or name_lower.endswith(".dockerfile"):
        language = "dockerfile"
    elif name in CMAKE_NAMES:
        language = "cmake"
    elif name in MESON_NAMES:
        language = "meson"
    elif name in BAZEL_NAMES or name_lower == ".bazelrc":
        language = "bazel"
    elif name == "Jenkinsfile":
        language = "groovy"
    elif path_lower.endswith((".pkr.hcl", ".pkrvars.hcl")):
        language = "packer"
    return language


def language_for(path: str) -> str:
    file_path = Path(path)
    name = file_path.name
    suffix = file_path.suffix
    normalized_suffix = suffix.lower()
    language = LANGUAGE_SUFFIXES.get(normalized_suffix, "text")
    if normalized_suffix in C_LANGUAGE_SUFFIXES:
        language = "c"
    elif suffix in ASM_SUFFIXES:
        language = "asm"
    elif name == "Makefile":
        language = "make"
    elif name in KCONFIG_NAMES:
        language = "kconfig"
    elif special_language := special_name_language(path, name):
        language = special_language
    elif is_shell_path(path):
        language = "shell"
    elif is_config_name(name):
        language = "config"
    return language


def content_class_for(language: str, flags: Mapping[str, bool]) -> str:
    if language == "patch":
        return "patch"
    return next((content_class for flag, content_class in CONTENT_CLASS_FLAG_ORDER if flags[flag]), "other")


def file_role_for(path: str, content_class: str) -> str:
    if "/init.d/" in path:
        return "runtime-service"
    if path.startswith("package/"):
        return "package"
    if path.startswith("include/"):
        return "source-include"
    if path.startswith(("scripts/", "tools/")):
        return "tooling"
    return content_class


def classification_flags(path: str, language: str) -> dict[str, bool]:
    parts = path.split("/")
    file_path = Path(path)
    name = file_path.name.lower()
    suffix = file_path.suffix.lower()
    is_test = any(part in {"test", "tests", "testing", "selftests"} for part in parts) or "test" in name
    is_generated = any(part in {"generated", "autogenerated"} for part in parts) or "generated" in name
    is_vendor = any(part in {"vendor", "third_party", "3rdparty"} for part in parts)
    is_doc = language == "doc" or parts[0] in {"docs", "doc", "Documentation"} or suffix in {".md", ".rst"}
    is_build = (
        language
        in {
            "autotools",
            "bazel",
            "boot_script",
            "cmake",
            "dockerfile",
            "linker_script",
            "make",
            "meson",
            "packer",
            "starlark",
        }
        or file_path.name == "Makefile"
        or "/cmake/" in path
        or name in BUILD_FILE_NAMES
    )
    is_config = (
        language
        in {"css", "graphql", "kconfig", "config", "json", "packer", "scss", "sql", "terraform", "toml", "xml", "yaml"}
        or "Config" in file_path.name
    )
    is_source = language in SOURCE_LANGUAGES
    return {
        "is_generated": is_generated,
        "is_vendor": is_vendor,
        "is_test": is_test,
        "is_source": is_source,
        "is_build": is_build,
        "is_config": is_config,
        "is_doc": is_doc,
    }


def classify_file(path: str, language: str) -> JsonObject:
    flags = classification_flags(path, language)
    content_class = content_class_for(language, flags)
    file_role = file_role_for(path, content_class)

    return {
        "file_role": file_role,
        "content_class": content_class,
        **flags,
    }


def inspect_inventory_file(path: Path, max_file_bytes: int) -> tuple[str | None, bytes, int, bool]:
    suffix = path.suffix.lower()
    reason: str | None = None
    if suffix in BINARY_SUFFIXES:
        reason = "binary_suffix"
    try:
        size_bytes = path.stat().st_size
    except OSError:
        return "read_error", b"", 0, False
    if size_bytes > max_file_bytes:
        return reason or "file_too_large", b"", size_bytes, False
    try:
        data = path.read_bytes()
    except OSError:
        return "read_error", b"", size_bytes, False
    if reason is None and b"\0" in data[:4096]:
        reason = "binary_nul"
    return reason, data, size_bytes, True


def file_from_previous_state(
    snapshot: Snapshot,
    rel_path: str,
    abs_path: Path,
    previous: PreviousFileState,
) -> IntelFile:
    return IntelFile(
        collection=snapshot.collection,
        repo=snapshot.repo,
        repo_role=snapshot.repo_role,
        branch=snapshot.branch,
        commit_sha=snapshot.commit_sha,
        tree_sha=snapshot.tree_sha,
        source_path=previous.source_path,
        repo_rel_path=rel_path,
        abs_path=abs_path,
        git_blob_sha=previous.git_blob_sha,
        file_sha256=previous.file_sha256,
        size_bytes=previous.size_bytes,
        language=previous.language,
        file_role=previous.file_role,
        content_class=previous.content_class,
        is_generated=previous.is_generated,
        is_vendor=previous.is_vendor,
        is_test=previous.is_test,
        is_source=previous.is_source,
        is_build=previous.is_build,
        is_config=previous.is_config,
        is_doc=previous.is_doc,
        skipped_reason=previous.skipped_reason,
        metadata=dict(previous.metadata),
    )


def reusable_previous_file(
    source_path: str,
    repo_rel_path: str,
    git_blob_sha: str | None,
    *,
    reuse: DiscoveryReuse,
) -> PreviousFileState | None:
    if not reuse.reuse_unchanged_blobs or git_blob_sha is None or repo_rel_path in reuse.dirty_paths:
        return None
    previous = reuse.previous_files.get(source_path)
    if previous is None or previous.git_blob_sha != git_blob_sha:
        return None
    return previous


def should_parse_text(repo_rel_path: str, language: str, skipped_reason: str | None) -> bool:
    if skipped_reason:
        return False
    name = Path(repo_rel_path).name
    suffix = Path(repo_rel_path).suffix
    if suffix in TEXT_SUFFIXES or name in TEXT_NAMES:
        return True
    if "/files/" in repo_rel_path or "/base-files/" in repo_rel_path:
        return True
    return language in {
        "autotools",
        "awk",
        "bazel",
        "boot_script",
        "cmake",
        "config",
        "dockerfile",
        "lex",
        "linker_script",
        "meson",
        "packer",
        "protobuf",
        "sql",
        "starlark",
        "terraform",
        "text",
        "xml",
        "yacc",
        *SOURCE_LANGUAGES,
    }


def classification_text(classified: JsonObject, key: str) -> str:
    value = classified.get(key)
    if not isinstance(value, str):
        raise TypeError(f"classification field {key} must be a string")
    return value


def classification_bool(classified: JsonObject, key: str) -> bool:
    value = classified.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"classification field {key} must be a boolean")
    return value


def make_snapshot(root: Path, repo: str, collection: str) -> Snapshot:
    repo_root = root / repo
    commit = run_git_required(repo_root, ["rev-parse", "HEAD"])
    base_tree_sha = run_git_required(repo_root, ["rev-parse", f"{commit}^{{tree}}"])
    commit_time = run_git_required(repo_root, ["log", "-1", "--format=%cI", commit])
    dirty_fingerprint = git_dirty_fingerprint(repo_root)
    dirty_paths = git_dirty_paths(repo_root) if dirty_fingerprint else set[str]()
    tree_sha = base_tree_sha
    if dirty_fingerprint:
        tree_sha = f"{base_tree_sha}:dirty:{dirty_fingerprint[:16]}"
    return Snapshot(
        collection=collection,
        repo=repo,
        repo_role=repo_role_for(repo),
        branch=run_git(repo_root, ["branch", "--show-current"]),
        commit_sha=commit,
        tree_sha=tree_sha,
        dirty=bool(dirty_fingerprint),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "chunker_version": CHUNKER_VERSION,
            "parser_version": PARSER_VERSION,
            "profile_name": profile_context.active_profile.name,
            "profile_version": profile_context.active_profile.version,
            "base_tree_sha": base_tree_sha,
            "commit_time": commit_time,
            "dirty_fingerprint": dirty_fingerprint,
            "dirty_paths": sorted(dirty_paths),
        },
    )


def discover_files(
    root: Path,
    snapshot: Snapshot,
    max_file_bytes: int,
    *,
    reuse: DiscoveryReuse | None = None,
) -> list[IntelFile]:
    repo_root = root / snapshot.repo
    files: list[IntelFile] = []
    reuse_context = reuse or DiscoveryReuse(previous_files={})
    dirty_paths = git_dirty_paths(repo_root)
    for git_blob_sha, rel_path in sorted(git_ls_files(repo_root), key=itemgetter(1)):
        abs_path = repo_root / rel_path
        if not abs_path.is_file():
            continue
        source_path = source_path_for(snapshot.repo, rel_path)
        if (prev := reusable_previous_file(source_path, rel_path, git_blob_sha, reuse=reuse_context)) is not None:
            files.append(file_from_previous_state(snapshot, rel_path, abs_path, prev))
            continue
        reason, data, size_bytes, read_ok = inspect_inventory_file(abs_path, max_file_bytes)
        language = language_for_read_file(rel_path, data, read_ok=read_ok)
        if reason is None and not (
            should_parse_text(rel_path, language, None)
            or profile_context.active_profile.should_parse_text(rel_path, language, None)
        ):
            reason = "unsupported_file_type"
        text = data.decode("utf-8", errors="replace") if read_ok and reason is None else None
        classified = profile_context.active_profile.classify_file(rel_path, language, classify_file(rel_path, language))
        file_md: JsonObject = {"path_parts": rel_path.split("/")[:8]}
        if language_has_metadata(language):
            file_md.update(language_metadata_for_file(rel_path, language, text))
        file_md.update(profile_context.active_profile.file_metadata(rel_path, language, classified))
        files.append(
            IntelFile(
                collection=snapshot.collection,
                repo=snapshot.repo,
                repo_role=snapshot.repo_role,
                branch=snapshot.branch,
                commit_sha=snapshot.commit_sha,
                tree_sha=snapshot.tree_sha,
                source_path=source_path,
                repo_rel_path=rel_path,
                abs_path=abs_path,
                git_blob_sha=git_blob_sha,
                file_sha256=sha256_bytes(data) if read_ok else None,
                size_bytes=size_bytes,
                language=language,
                skipped_reason=reason,
                metadata={key: value for key, value in file_md.items() if value},
                file_role=classification_text(classified, "file_role"),
                content_class=classification_text(classified, "content_class"),
                is_generated=classification_bool(classified, "is_generated"),
                is_vendor=classification_bool(classified, "is_vendor"),
                is_test=classification_bool(classified, "is_test"),
                is_source=classification_bool(classified, "is_source"),
                is_build=classification_bool(classified, "is_build"),
                is_config=classification_bool(classified, "is_config"),
                is_doc=classification_bool(classified, "is_doc"),
                is_untracked=False,
                indexed_dirty=rel_path in dirty_paths,
            )
        )
    for rel_path in sorted(git_untracked_files(repo_root)):
        abs_path = repo_root / rel_path
        if not abs_path.is_file():
            continue
        source_path = source_path_for(snapshot.repo, rel_path)
        reason, data, size_bytes, read_ok = inspect_inventory_file(abs_path, max_file_bytes)
        language = language_for_read_file(rel_path, data, read_ok=read_ok)
        if reason is None and not (
            should_parse_text(rel_path, language, None)
            or profile_context.active_profile.should_parse_text(rel_path, language, None)
        ):
            reason = "unsupported_file_type"
        text = data.decode("utf-8", errors="replace") if read_ok and reason is None else None
        classified = profile_context.active_profile.classify_file(rel_path, language, classify_file(rel_path, language))
        md: JsonObject = {"path_parts": rel_path.split("/")[:8]}
        if language_has_metadata(language):
            md.update(language_metadata_for_file(rel_path, language, text))
        md.update(profile_context.active_profile.file_metadata(rel_path, language, classified))
        files.append(
            IntelFile(
                collection=snapshot.collection,
                repo=snapshot.repo,
                repo_role=snapshot.repo_role,
                branch=snapshot.branch,
                commit_sha=snapshot.commit_sha,
                tree_sha=snapshot.tree_sha,
                source_path=source_path,
                repo_rel_path=rel_path,
                abs_path=abs_path,
                git_blob_sha=None,
                file_sha256=sha256_bytes(data) if read_ok else None,
                size_bytes=size_bytes,
                language=language,
                skipped_reason=reason,
                metadata={key: value for key, value in md.items() if value},
                file_role=classification_text(classified, "file_role"),
                content_class=classification_text(classified, "content_class"),
                is_generated=classification_bool(classified, "is_generated"),
                is_vendor=classification_bool(classified, "is_vendor"),
                is_test=classification_bool(classified, "is_test"),
                is_source=classification_bool(classified, "is_source"),
                is_build=classification_bool(classified, "is_build"),
                is_config=classification_bool(classified, "is_config"),
                is_doc=classification_bool(classified, "is_doc"),
                is_untracked=True,
                indexed_dirty=True,
            )
        )
    return files


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
