"""Shared code-intelligence data structures.

The ingester, query server, and tests all work with the same small set of
records. Keeping those records here makes the ingestion pipeline easier to
reason about without introducing a larger framework.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from pathlib import Path

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Sequence["JsonValue"] | Mapping[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

SCHEMA_VERSION = "code-intel-schema-v2"
CHUNKER_VERSION = "code-intel-v1"
PARSER_VERSION = "stdlib-heuristic-v3"
SOURCE_TYPE = "code_intel"

DEFAULT_EMBED_RECORD_TYPES = {
    "code_chunk",
    "package_definition",
    "config_symbol",
    "patch_hunk",
    "dts_node",
    "service_entrypoint",
    "security_pattern",
    "static_finding",
    "doc_section",
}

TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".rs",
    ".go",
    ".py",
    ".java",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".cs",
    ".php",
    ".rb",
    ".s",
    ".S",
    ".mk",
    ".in",
    ".sh",
    ".lua",
    ".uc",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".dts",
    ".dtsi",
    ".patch",
    ".diff",
    ".conf",
    ".config",
    ".seed",
    ".service",
    ".init",
    ".md",
    ".rst",
    ".txt",
}

TEXT_NAMES = {
    "Makefile",
    "Kconfig",
    "Config.in",
    "Config-defaults.in",
    "config",
    "inittab",
    "rc.common",
    "sysupgrade",
}

BINARY_SUFFIXES = {
    ".a",
    ".bin",
    ".bmp",
    ".bz2",
    ".dtb",
    ".elf",
    ".gif",
    ".gz",
    ".img",
    ".ipk",
    ".jar",
    ".jpg",
    ".jpeg",
    ".ko",
    ".o",
    ".pdf",
    ".png",
    ".so",
    ".tar",
    ".tgz",
    ".woff",
    ".xz",
    ".zip",
    ".zst",
}

SOURCE_LANGUAGES = {
    "asm",
    "c",
    "csharp",
    "go",
    "java",
    "javascript",
    "lua",
    "php",
    "python",
    "ruby",
    "rust",
    "shell",
    "typescript",
    "ucode",
    "dts",
}


@dataclass
class Snapshot:
    collection: str
    repo: str
    repo_role: str
    branch: str | None
    commit_sha: str
    tree_sha: str
    dirty: bool
    metadata: JsonObject = field(default_factory=dict)


@dataclass
class IntelFile:
    collection: str
    repo: str
    repo_role: str
    branch: str | None
    commit_sha: str
    tree_sha: str
    source_path: str
    repo_rel_path: str
    abs_path: Path
    git_blob_sha: str | None
    file_sha256: str | None
    size_bytes: int
    language: str
    file_role: str
    content_class: str
    is_generated: bool
    is_vendor: bool
    is_test: bool
    is_source: bool
    is_build: bool
    is_config: bool
    is_doc: bool
    skipped_reason: str | None
    metadata: JsonObject = field(default_factory=dict)


@dataclass
class IntelRecord:
    collection: str
    source_path: str
    language: str
    file_role: str
    content_class: str
    record_type: str
    record_id: str
    title: str
    summary: str
    embedding_text: str
    display_content: str
    line_start: int | None = None
    line_end: int | None = None
    symbol: str | None = None
    symbol_kind: str | None = None
    parent_record_id: str | None = None
    confidence_kind: str = "high_confidence_fact"
    confidence: float | None = None
    tool: str | None = None
    rule_id: str | None = None
    severity: str | None = None
    analyzer: str | None = None
    analyzer_version: str | None = None
    parser: str | None = None
    parser_version: str = PARSER_VERSION
    chunker_version: str = CHUNKER_VERSION
    metadata: JsonObject = field(default_factory=dict)
    embedding: str | None = None


@dataclass
class IntelEdge:
    source_record_id: str
    edge_type: str
    target_record_id: str | None = None
    source_symbol: str | None = None
    target_symbol: str | None = None
    source_path: str | None = None
    target_path: str | None = None
    confidence_kind: str = "approximate_fact"
    metadata: JsonObject = field(default_factory=dict)


@dataclass
class RepoIngest:
    snapshot: Snapshot
    files: list[IntelFile]
    records: list[IntelRecord]
    edges: list[IntelEdge]
    parser_failures: list[JsonObject]
    mode: str = "full"
    previous_snapshot_id: int | None = None
    changed_paths: set[str] = field(default_factory=set)
    unchanged_paths: set[str] = field(default_factory=set)
    deleted_paths: set[str] = field(default_factory=set)


@dataclass
class StaticRule:
    rule_id: str
    name: str | None = None
    short_description: str | None = None
    full_description: str | None = None
    default_level: str | None = None
    help_uri: str | None = None
    properties: JsonObject = field(default_factory=dict)
    metadata: JsonObject = field(default_factory=dict)


@dataclass
class StaticLocation:
    ordinal: int
    location_kind: str
    source_path: str | None
    uri: str | None
    message: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    column_start: int | None = None
    column_end: int | None = None
    snippet: str | None = None
    properties: JsonObject = field(default_factory=dict)


@dataclass
class StaticCodeFlowStep:
    flow_index: int
    thread_index: int
    step_index: int
    source_path: str | None
    uri: str | None
    message: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    column_start: int | None = None
    column_end: int | None = None
    importance: str | None = None
    properties: JsonObject = field(default_factory=dict)


@dataclass
class StaticFinding:
    finding_key: str
    rule_id: str
    message: str
    rule_index: int | None = None
    level: str | None = None
    kind: str | None = None
    baseline_state: str | None = None
    primary_source_path: str | None = None
    primary_uri: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    column_start: int | None = None
    column_end: int | None = None
    fingerprints: JsonObject = field(default_factory=dict)
    suppressions: list[JsonValue] = field(default_factory=list)
    properties: JsonObject = field(default_factory=dict)
    raw_result: JsonObject = field(default_factory=dict)
    locations: list[StaticLocation] = field(default_factory=list)
    code_flows: list[StaticCodeFlowStep] = field(default_factory=list)


@dataclass
class StaticRun:
    repo: str
    sarif_path: str
    sarif_sha256: str
    run_index: int
    tool_name: str
    tool_version: str | None = None
    semantic_version: str | None = None
    information_uri: str | None = None
    automation_id: str | None = None
    metadata: JsonObject = field(default_factory=dict)
    rules: list[StaticRule] = field(default_factory=list)
    findings: list[StaticFinding] = field(default_factory=list)


@dataclass(frozen=True)
class SarifPathResolution:
    source_path: str | None
    repo: str | None
    path_mapping: str


@dataclass
class SarifIngest:
    runs: list[StaticRun]
    records_by_repo: dict[str, list[IntelRecord]]
    failures: list[JsonObject]
