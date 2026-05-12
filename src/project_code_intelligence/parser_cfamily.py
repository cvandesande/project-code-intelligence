"""C-like language parser helpers."""

from __future__ import annotations

import re

from project_code_intelligence.models import IntelEdge, IntelFile, IntelRecord
from project_code_intelligence.parser_core import (
    SymbolChunkSpec,
    bounded_brace_body,
    make_symbol_chunk,
    string_items,
)
from project_code_intelligence.parser_security import security_api_refs
from project_code_intelligence.records import (
    RecordSpec,
    common_extracts,
    extract_referenced_symbols,
    line_for_offset_with_index,
    line_offsets,
    line_window_records,
    make_record,
)

C_SIGNATURE_LOOKAHEAD_LINES = 5


def iter_c_function_candidates(lines: list[str]) -> list[tuple[str, int, int, str, bool]]:
    candidates: list[tuple[str, int, int, str, bool]] = []
    skip_prefixes = ("if", "for", "while", "switch", "return", "sizeof")
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//", "/*", "*")):
            idx += 1
            continue
        if "(" not in stripped:
            idx += 1
            continue
        start = idx
        buffer = stripped
        lookahead = idx
        while (
            "{" not in buffer
            and ";" not in buffer
            and lookahead + 1 < len(lines)
            and lookahead - start < C_SIGNATURE_LOOKAHEAD_LINES
        ):
            lookahead += 1
            buffer += " " + lines[lookahead].strip()
        if "{" not in buffer or ";" in buffer.split("{", 1)[0]:
            idx += 1
            continue
        prefix = buffer.split("(", 1)[0].strip()
        tokens = [match.group(0) for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", prefix)]
        if not tokens:
            idx += 1
            continue
        name = tokens[-1]
        if name in skip_prefixes:
            idx += 1
            continue
        line_end, body, truncated = bounded_brace_body(lines, start, max_lines=260, max_chars=5200)
        candidates.append((name, start + 1, line_end, body, truncated))
        idx = max(idx + 1, line_end)
    return candidates


def c_records(
    intel_file: IntelFile, text: str, max_chars: int, overlap_lines: int
) -> tuple[list[IntelRecord], list[IntelEdge]]:
    records: list[IntelRecord] = []
    edges: list[IntelEdge] = []
    lines = text.splitlines()
    includes = string_items(common_extracts(text).get("includes"))
    for include in includes:
        record_id = f"{intel_file.source_path}::include::{include}"
        records.append(
            make_record(
                intel_file,
                RecordSpec(
                    record_type="include_edge",
                    record_id=record_id,
                    title=f"{intel_file.source_path} includes {include}",
                    summary=f"Include edge from {intel_file.source_path} to {include}",
                    body=f"#include <{include}>",
                    line_start=None,
                    line_end=None,
                    metadata={"include": include},
                    confidence_kind="high_confidence_fact",
                ),
            )
        )
        edges.append(
            IntelEdge(
                source_record_id=record_id,
                edge_type="include",
                source_path=intel_file.source_path,
                target_path=include,
                confidence_kind="high_confidence_fact",
                metadata={"include": include},
            )
        )

    for name, line_start, line_end, body, truncated in iter_c_function_candidates(lines):
        refs = extract_referenced_symbols(body)
        metadata = {
            **common_extracts(body),
            "symbols_defined": [name],
            "symbols_referenced": refs,
            "security_sensitive_apis": security_api_refs(body),
            "bounded_function_parser": True,
            "body_truncated": truncated,
        }
        record_id = f"{intel_file.source_path}::function::{name}::{line_start:06d}"
        records.extend((
            make_record(
                intel_file,
                RecordSpec(
                    record_type="symbol_definition",
                    record_id=record_id,
                    title=f"{name} in {intel_file.source_path}:{line_start}-{line_end}",
                    summary=f"C function definition {name} in {intel_file.source_path}",
                    body=body,
                    line_start=line_start,
                    line_end=line_end,
                    symbol=name,
                    symbol_kind="function",
                    metadata=metadata,
                    confidence_kind="approximate_fact",
                ),
            ),
            make_record(
                intel_file,
                RecordSpec(
                    record_type="code_chunk",
                    record_id=f"{intel_file.source_path}::function_chunk::{name}::{line_start:06d}",
                    title=f"{name} body in {intel_file.source_path}:{line_start}-{line_end}",
                    summary=f"Function chunk for {name}",
                    body=body,
                    line_start=line_start,
                    line_end=line_end,
                    symbol=name,
                    symbol_kind="function",
                    metadata=metadata,
                    parent_record_id=record_id,
                    confidence_kind="approximate_fact",
                ),
            ),
        ))
        edges.extend(
            (
                IntelEdge(
                    source_record_id=record_id,
                    edge_type="call_candidate",
                    source_symbol=name,
                    target_symbol=ref,
                    source_path=intel_file.source_path,
                    confidence_kind="heuristic_candidate",
                )
            )
            for ref in refs[:80]
        )

    offsets = line_offsets(text)
    for match in re.finditer(r"(?m)^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s*\(([^)]*)\))?.*$", text):
        name = match.group(1)
        line_start = line_for_offset_with_index(offsets, match.start())
        line = match.group(0)
        records.append(
            make_record(
                intel_file,
                RecordSpec(
                    record_type="symbol_definition",
                    record_id=f"{intel_file.source_path}::macro::{name}::{line_start:06d}",
                    title=f"macro {name} in {intel_file.source_path}:{line_start}",
                    summary=f"C macro definition {name}",
                    body=line,
                    line_start=line_start,
                    line_end=line_start,
                    symbol=name,
                    symbol_kind="macro",
                    metadata={"symbols_defined": [name], **common_extracts(line)},
                ),
            )
        )

    if not any(record.record_type == "code_chunk" for record in records):
        records.extend(line_window_records(intel_file, text, max_chars, overlap_lines))
    return records, edges


def go_import_paths(text: str) -> list[str]:
    imports: list[str] = []
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if in_block:
            if stripped.startswith(")"):
                in_block = False
                continue
            match = re.match(r'(?:[A-Za-z_][A-Za-z0-9_]*|\.)?\s*"([^"]+)"', stripped)
            if match:
                imports.append(match.group(1))
            continue
        if re.match(r"import\s*\(", stripped):
            in_block = True
            continue
        match = re.match(r'import\s+(?:"([^"]+)"|(?:[A-Za-z_][A-Za-z0-9_]*|\.)\s+"([^"]+)")', stripped)
        if match:
            imports.append(match.group(1) or match.group(2))
    return sorted(set(imports))


def go_records(
    intel_file: IntelFile, text: str, max_chars: int, overlap_lines: int
) -> tuple[list[IntelRecord], list[IntelEdge]]:
    records: list[IntelRecord] = []
    edges: list[IntelEdge] = []
    lines = text.splitlines()
    offsets = line_offsets(text)
    package_match = re.search(r"(?m)^\s*package\s+([A-Za-z_][A-Za-z0-9_]*)", text)
    imports_flat = go_import_paths(text)
    for idx, line in enumerate(lines):
        match = re.match(
            r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)(?:\[[^\]]+\])?\s*\(",
            line,
        )
        if not match:
            continue
        name = match.group(1)
        line_end, body, truncated = bounded_brace_body(lines, idx)
        symbol, chunk, symbol_edges = make_symbol_chunk(
            intel_file,
            SymbolChunkSpec(
                language_label="Go",
                name=name,
                kind="function",
                line_start=idx + 1,
                line_end=line_end,
                body=body,
                metadata={
                    "package_name": package_match.group(1) if package_match else None,
                    "imports": imports_flat[:80],
                    "body_truncated": truncated,
                },
                confidence_kind="approximate_fact",
            ),
        )
        records.extend([symbol, chunk])
        edges.extend(symbol_edges)
    for match in re.finditer(r"(?m)^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\s+(struct|interface)\b", text):
        line = line_for_offset_with_index(offsets, match.start())
        line_end, body, truncated = bounded_brace_body(lines, line - 1)
        symbol, chunk, _symbol_edges = make_symbol_chunk(
            intel_file,
            SymbolChunkSpec(
                language_label="Go",
                name=match.group(1),
                kind=match.group(2),
                line_start=line,
                line_end=line_end,
                body=body,
                metadata={
                    "package_name": package_match.group(1) if package_match else None,
                    "body_truncated": truncated,
                },
                confidence_kind="approximate_fact",
            ),
        )
        records.extend([symbol, chunk])
    if not records:
        records.extend(line_window_records(intel_file, text, max_chars, overlap_lines))
    return records, edges


def rust_records(
    intel_file: IntelFile, text: str, max_chars: int, overlap_lines: int
) -> tuple[list[IntelRecord], list[IntelEdge]]:
    records: list[IntelRecord] = []
    edges: list[IntelEdge] = []
    lines = text.splitlines()
    pattern = re.compile(
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+|unsafe\s+|const\s+)?"
        r"(fn|struct|enum|trait|impl)\s+([A-Za-z_][A-Za-z0-9_]*)?"
    )
    for idx, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            continue
        kind = match.group(1)
        name = match.group(2) or f"impl_at_{idx + 1}"
        line_end, body, truncated = bounded_brace_body(lines, idx)
        symbol, chunk, symbol_edges = make_symbol_chunk(
            intel_file,
            SymbolChunkSpec(
                language_label="Rust",
                name=name,
                kind=kind,
                line_start=idx + 1,
                line_end=line_end,
                body=body,
                metadata={"body_truncated": truncated},
                confidence_kind="approximate_fact",
            ),
        )
        records.extend([symbol, chunk])
        edges.extend(symbol_edges)
    if not records:
        records.extend(line_window_records(intel_file, text, max_chars, overlap_lines))
    return records, edges
