"""Generic code-intelligence ingestion profile.

Profiles keep repository-specific interpretation out of the core ingester.
The base profile should remain useful for ordinary C, Rust, Go, Python, shell,
docs, config, and build repositories without assuming a product-specific layout.

When a profile change alters emitted records, metadata, classifications, or
security eligibility, bump that profile's version so incremental ingestion does
not silently reuse stale records.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, TypeVar

if TYPE_CHECKING:
    from typing_extensions import override

    from project_code_intelligence.models import IntelEdge, JsonObject
else:
    _T = TypeVar("_T")

    def override(method: _T) -> _T:
        return method

    JsonObject = dict[str, object]
    IntelEdge = object


SecurityPattern = tuple[str, str, str, str, str]


class ProfileRecord(TypedDict, total=False):
    record_type: str
    record_id: str
    title: str
    summary: str
    body: str
    line_start: int | None
    line_end: int | None
    symbol: str | None
    symbol_kind: str | None
    parent_record_id: str | None
    confidence_kind: str
    confidence: float | None
    tool: str | None
    rule_id: str | None
    severity: str | None
    analyzer: str | None
    analyzer_version: str | None
    metadata: JsonObject


COMMON_SECURITY_PATTERNS: list[SecurityPattern] = [
    (r"\bstrcpy\s*\(", "unsafe_string_copy", "medium", "heuristic_candidate", "strcpy without visible bound"),
    (r"\bstrcat\s*\(", "unsafe_string_concat", "medium", "heuristic_candidate", "strcat without visible bound"),
    (r"\bsprintf\s*\(", "unsafe_sprintf", "medium", "heuristic_candidate", "sprintf without visible bound"),
    (r"\bvsprintf\s*\(", "unsafe_vsprintf", "medium", "heuristic_candidate", "vsprintf without visible bound"),
    (r"\bgets\s*\(", "unsafe_gets", "high", "heuristic_candidate", "gets is inherently unsafe"),
    (r"\bsystem\s*\(", "shell_command_execution", "medium", "heuristic_candidate", "system command execution"),
    (r"\bpopen\s*\(", "shell_command_execution", "medium", "heuristic_candidate", "popen command execution"),
    (r"\beval\s+", "shell_eval", "medium", "heuristic_candidate", "shell eval usage"),
    (r"`[^`]+`", "shell_backtick_execution", "low", "heuristic_candidate", "shell backtick command substitution"),
    (
        r"\bcopy_from_user\s*\(",
        "kernel_user_boundary",
        "medium",
        "heuristic_candidate",
        "kernel userspace copy boundary",
    ),
    (r"\bcopy_to_user\s*\(", "kernel_user_boundary", "medium", "heuristic_candidate", "kernel userspace copy boundary"),
    (r"\bnla_data\s*\(", "netlink_input_boundary", "medium", "heuristic_candidate", "netlink attribute input boundary"),
]


class CodeIntelProfile:
    """Project-specific hooks for code-intelligence ingestion."""

    name = "generic"
    version = "v1"
    default_repos = (".",)

    def repo_role(self, repo: str) -> str:
        del repo
        return "project"

    def language_for_path(self, path: str) -> str | None:
        del path
        return None

    def classify_file(self, path: str, language: str, classification: JsonObject) -> JsonObject:
        del path, language
        return classification

    def file_metadata(self, path: str, language: str, classification: JsonObject) -> JsonObject:
        del path, language, classification
        return {}

    def should_parse_text(self, path: str, language: str, skipped_reason: str | None) -> bool:
        del path, language, skipped_reason
        return False

    def make_metadata(self, path: str, text: str) -> JsonObject:
        del path, text
        return {}

    def make_block_record(
        self, kind: str, name: str | None, path: str, body: str
    ) -> tuple[str, str | None, str | None]:
        del kind, name, path, body
        return "code_chunk", None, None

    def shell_service_records(self, repo_rel_path: str, source_path: str, text: str) -> list[ProfileRecord]:
        del repo_rel_path, source_path, text
        return []

    def extra_records(
        self, path: str, source_path: str, language: str, text: str
    ) -> tuple[list[ProfileRecord], list[IntelEdge]]:
        """Return profile-specific record specs and edge objects.

        The core ingester converts each record spec with make_record(). This is
        intentionally additive; profiles should use it for records that are
        project concepts rather than replacement parsers. Examples include
        product feature records, Kubernetes YAML objects, NGINX locations,
        protobuf services, or Cargo package metadata.
        """
        del path, source_path, language, text
        return [], []

    def security_patterns(self) -> list[SecurityPattern]:
        return COMMON_SECURITY_PATTERNS

    def should_scan_security(self, path: str, language: str, file_role: str) -> bool:
        del path, file_role
        return language in {"c", "shell", "lua", "ucode", "make", "patch", "go", "rust", "python"}

    def security_context(self, path: str, language: str, file_role: str, content_class: str) -> JsonObject:
        contexts: list[str] = []
        boundaries: list[str] = []
        if content_class == "patch" or path.endswith((".patch", ".diff")):
            contexts.append("patch_payload")
        if content_class == "build":
            contexts.append("build_time")
        if content_class == "config":
            contexts.append("configuration")
            boundaries.append("config_input")
        if content_class == "test":
            contexts.append("test_code")
        if file_role == "runtime-service" or "/init.d/" in path:
            contexts.append("runtime_service")
            boundaries.append("service_entrypoint")
        if language in {"c", "go", "rust", "python", "shell", "lua", "ucode"} and not contexts:
            contexts.append("source_code")
        if not contexts:
            contexts.append(file_role)
        return {
            "security_contexts": sorted(set(contexts)),
            "boundary_candidates": sorted(set(boundaries)),
        }

    def embedding_metadata_keys(self) -> list[str]:
        return [
            "symbol",
            "symbol_kind",
            "target",
            "subtarget",
            "config_symbols",
            "symbols_defined",
            "symbols_referenced",
            "includes",
            "dts_compatibles",
            "security_sensitive_apis",
            "security_contexts",
            "boundary_candidates",
            "log_error_messages",
        ]

    def report_metadata_keys(self) -> list[str]:
        return self.embedding_metadata_keys()

    def sarif_globs(self, repos: list[str]) -> list[str]:
        del repos
        return []


class GenericProfile(CodeIntelProfile):
    name = "generic"
    version = "v1"

    @override
    def classify_file(self, path: str, language: str, classification: JsonObject) -> JsonObject:
        del language
        updated = dict(classification)
        parts = path.split("/")
        name = Path(path).name
        if parts and parts[0] in {"src", "source", "lib", "cmd", "pkg", "internal"} and updated["is_source"]:
            updated["file_role"] = "source"
        elif parts and parts[0] in {"include", "headers"}:
            updated["file_role"] = "source-include"
        elif parts and parts[0] in {"scripts", "tools"}:
            updated["file_role"] = "tooling"
        elif name in {"Cargo.toml", "go.mod", "pyproject.toml", "package.json"}:
            updated["file_role"] = "project-manifest"
        return updated
