"""Language-specific code record extraction."""

from __future__ import annotations

from project_code_intelligence.parsers.cfamily import c_records, go_records, rust_records
from project_code_intelligence.parsers.core import (
    LanguageParser,
    SymbolChunkSpec,
    bounded_brace_body,
    first_sentence,
    make_profile_record,
    make_symbol_chunk,
    string_items,
)
from project_code_intelligence.parsers.patch import patch_records
from project_code_intelligence.parsers.project import (
    doc_parser,
    dts_records,
    json_like_records,
    kconfig_parser,
    make_records,
    shell_records,
)
from project_code_intelligence.parsers.python import iter_python_definitions, python_node_start_lineno, python_records
from project_code_intelligence.parsers.registry import (
    LANGUAGE_PARSERS,
    parse_file,
    parse_language_records,
    patch_parser,
)
from project_code_intelligence.parsers.security import (
    security_api_refs,
    security_context,
    security_pattern_anchor,
    security_records,
)

__all__ = [
    "LANGUAGE_PARSERS",
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
    "patch_records",
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
