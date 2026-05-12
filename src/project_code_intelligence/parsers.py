"""Language-specific code record extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from project_code_intelligence import profile_context
from project_code_intelligence.inventory import read_text
from project_code_intelligence.parser_cfamily import c_records, go_records, rust_records
from project_code_intelligence.parser_core import (
    LanguageParser,
    SymbolChunkSpec,
    bounded_brace_body,
    first_sentence,
    make_profile_record,
    make_symbol_chunk,
    string_items,
)
from project_code_intelligence.parser_patch import patch_records
from project_code_intelligence.parser_project import (
    doc_parser,
    dts_records,
    json_like_records,
    kconfig_parser,
    make_records,
    shell_records,
)
from project_code_intelligence.parser_python import (
    iter_python_definitions,
    python_node_start_lineno,
    python_records,
)
from project_code_intelligence.parser_security import (
    security_api_refs,
    security_context,
    security_pattern_anchor,
    security_records,
)
from project_code_intelligence.records import line_window_records

if TYPE_CHECKING:
    from project_code_intelligence.models import IntelEdge, IntelFile, IntelRecord, JsonObject

__all__ = [
    "LanguageParser",
    "SymbolChunkSpec",
    "bounded_brace_body",
    "c_records",
    "doc_parser",
    "dts_records",
    "first_sentence",
    "go_records",
    "iter_python_definitions",
    "json_like_records",
    "kconfig_parser",
    "make_profile_record",
    "make_records",
    "make_symbol_chunk",
    "parse_file",
    "parse_language_records",
    "patch_parser",
    "python_node_start_lineno",
    "python_records",
    "rust_records",
    "security_api_refs",
    "security_context",
    "security_pattern_anchor",
    "security_records",
    "shell_records",
    "string_items",
]


def patch_parser(
    intel_file: IntelFile,
    text: str,
    _max_chars: int,
    _overlap_lines: int,
) -> tuple[list[IntelRecord], list[IntelEdge]]:
    return patch_records(intel_file, text)


LANGUAGE_PARSERS: dict[str, LanguageParser] = {
    "c": c_records,
    "go": go_records,
    "rust": rust_records,
    "python": python_records,
    "kconfig": kconfig_parser,
    "make": make_records,
    "patch": patch_parser,
    "dts": dts_records,
    "shell": shell_records,
    "doc": doc_parser,
    "json": json_like_records,
    "toml": json_like_records,
    "yaml": json_like_records,
}


def parse_language_records(
    intel_file: IntelFile, text: str, max_chars: int, overlap_lines: int
) -> tuple[list[IntelRecord], list[IntelEdge]]:
    parser = LANGUAGE_PARSERS.get(intel_file.language)
    if parser is not None:
        return parser(intel_file, text, max_chars, overlap_lines)
    return line_window_records(intel_file, text, max_chars, overlap_lines), []


def parse_file(
    intel_file: IntelFile, max_chars: int, overlap_lines: int
) -> tuple[list[IntelRecord], list[IntelEdge], list[JsonObject]]:
    failures: list[JsonObject] = []
    if intel_file.skipped_reason:
        return [], [], failures
    try:
        text = read_text(intel_file.abs_path)
    except OSError as exc:
        return (
            [],
            [],
            [
                {
                    "source_path": intel_file.source_path,
                    "language": intel_file.language,
                    "parser": "read",
                    "error": str(exc),
                }
            ],
        )
    try:
        records, edges = parse_language_records(intel_file, text, max_chars, overlap_lines)
        extra_record_specs, extra_edges = profile_context.active_profile.extra_records(
            intel_file.repo_rel_path,
            intel_file.source_path,
            intel_file.language,
            text,
        )
        for spec in extra_record_specs:
            records.append(make_profile_record(intel_file, spec))
        edges.extend(extra_edges)
        records.extend(security_records(intel_file, text))
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - parser boundary
        failures.append({
            "source_path": intel_file.source_path,
            "language": intel_file.language,
            "parser": intel_file.language,
            "error": str(exc),
        })
        return line_window_records(intel_file, text, max_chars, overlap_lines), [], failures
    else:
        return records, edges, failures
