"""Shared parser helpers and record construction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from project_code_intelligence.models import IntelEdge, IntelFile, IntelRecord, JsonObject
from project_code_intelligence.parsers.security import security_api_refs
from project_code_intelligence.records import (
    RecordSpec,
    common_extracts,
    extract_referenced_symbols,
    make_record,
)

if TYPE_CHECKING:
    from project_code_intelligence.code_profiles.base import ProfileRecord

LanguageParser = Callable[[IntelFile, str, int, int], tuple[list[IntelRecord], list[IntelEdge]]]


@dataclass(frozen=True)
class SymbolChunkSpec:
    language_label: str
    name: str
    kind: str
    line_start: int
    line_end: int
    body: str
    metadata: JsonObject | None = None
    confidence_kind: str = "approximate_fact"
    # Names that should not produce call_candidate edges (e.g. Go builtins).
    # Skips edges where target_symbol is in this set, because the heuristic
    # SQL resolver binds by name and would bind builtins to any same-named
    # user symbol in the codebase.
    non_resolvable_targets: frozenset[str] = frozenset()


def make_profile_record(intel_file: IntelFile, spec: ProfileRecord, *, default_body: str = "") -> IntelRecord:
    record_type = spec.get("record_type")
    record_id = spec.get("record_id")
    title = spec.get("title")
    summary = spec.get("summary")
    if not isinstance(record_type, str) or not isinstance(record_id, str):
        raise TypeError("profile records require string record_type and record_id")
    if not isinstance(title, str) or not isinstance(summary, str):
        raise TypeError("profile records require string title and summary")
    return make_record(
        intel_file,
        RecordSpec(
            record_type=record_type,
            record_id=record_id,
            title=title,
            summary=summary,
            body=spec.get("body", default_body),
            line_start=spec.get("line_start"),
            line_end=spec.get("line_end"),
            symbol=spec.get("symbol"),
            symbol_kind=spec.get("symbol_kind"),
            parent_record_id=spec.get("parent_record_id"),
            confidence_kind=spec.get("confidence_kind", "high_confidence_fact"),
            confidence=spec.get("confidence"),
            tool=spec.get("tool"),
            rule_id=spec.get("rule_id"),
            severity=spec.get("severity"),
            analyzer=spec.get("analyzer"),
            analyzer_version=spec.get("analyzer_version"),
            metadata=spec.get("metadata", {}),
        ),
    )


def string_items(value: object) -> list[str]:
    if isinstance(value, list):
        return [item for item in cast("list[object]", value) if isinstance(item, str)]
    return []


def advance_block_comment(line: str, idx: int) -> tuple[int, bool]:
    if line.startswith("*/", idx):
        return idx + 2, False
    return idx + 1, True


def advance_quote(idx: int, char: str, quote: str, *, escape: bool) -> tuple[int, str | None, bool]:
    if escape:
        return idx + 1, quote, False
    if char == "\\" and quote != "`":
        return idx + 1, quote, True
    if char == quote:
        return idx + 1, None, False
    return idx + 1, quote, False


def brace_delta(line: str, *, in_block_comment: bool) -> tuple[int, bool, bool]:
    delta = 0
    saw_open = False
    idx = 0
    quote: str | None = None
    escape = False
    while idx < len(line):
        char = line[idx]
        nxt = line[idx + 1] if idx + 1 < len(line) else ""
        if in_block_comment:
            idx, in_block_comment = advance_block_comment(line, idx)
            continue
        if quote:
            idx, quote, escape = advance_quote(idx, char, quote, escape=escape)
            continue
        if char == "/" and nxt == "*":
            in_block_comment = True
            idx += 2
            continue
        if char == "/" and nxt == "/":
            break
        if char in {'"', "'", "`"}:
            quote = char
        elif char == "{":
            delta += 1
            saw_open = True
        elif char == "}":
            delta -= 1
        idx += 1
    return delta, in_block_comment, saw_open


def bounded_brace_body(
    lines: list[str], start_idx: int, max_lines: int = 180, max_chars: int = 5200
) -> tuple[int, str, bool]:
    depth = 0
    saw_open = False
    end_idx = min(len(lines) - 1, start_idx + max_lines - 1)
    in_block_comment = False
    for idx in range(start_idx, min(len(lines), start_idx + max_lines)):
        line = lines[idx]
        delta, in_block_comment, line_saw_open = brace_delta(line, in_block_comment=in_block_comment)
        depth += delta
        if line_saw_open:
            saw_open = True
        end_idx = idx
        if saw_open and depth <= 0:
            break
    body = "\n".join(lines[start_idx : end_idx + 1])
    truncated = end_idx >= start_idx + max_lines - 1 or len(body) > max_chars
    if len(body) > max_chars:
        body = body[: max_chars - 38].rstrip() + "\n/* symbol candidate truncated */"
    return end_idx + 1, body, truncated


def make_symbol_chunk(intel_file: IntelFile, spec: SymbolChunkSpec) -> tuple[IntelRecord, IntelRecord, list[IntelEdge]]:
    refs = [ref for ref in extract_referenced_symbols(spec.body) if ref not in spec.non_resolvable_targets]
    meta = {
        **common_extracts(spec.body),
        "symbols_defined": [spec.name],
        "symbols_referenced": refs,
        "security_sensitive_apis": security_api_refs(spec.body),
        **(spec.metadata or {}),
    }
    record_id = f"{intel_file.source_path}::{spec.kind}::{spec.name}::{spec.line_start:06d}"
    symbol_record = make_record(
        intel_file,
        RecordSpec(
            record_type="symbol_definition",
            record_id=record_id,
            title=f"{spec.name} in {intel_file.source_path}:{spec.line_start}-{spec.line_end}",
            summary=f"{spec.language_label} {spec.kind} definition {spec.name} in {intel_file.source_path}",
            body=spec.body,
            line_start=spec.line_start,
            line_end=spec.line_end,
            symbol=spec.name,
            symbol_kind=spec.kind,
            metadata=meta,
            confidence_kind=spec.confidence_kind,
        ),
    )
    chunk_record = make_record(
        intel_file,
        RecordSpec(
            record_type="code_chunk",
            record_id=f"{intel_file.source_path}::{spec.kind}_chunk::{spec.name}::{spec.line_start:06d}",
            title=f"{spec.name} body in {intel_file.source_path}:{spec.line_start}-{spec.line_end}",
            summary=f"{spec.language_label} {spec.kind} chunk for {spec.name}",
            body=spec.body,
            line_start=spec.line_start,
            line_end=spec.line_end,
            symbol=spec.name,
            symbol_kind=spec.kind,
            metadata=meta,
            parent_record_id=record_id,
            confidence_kind=(
                "approximate_fact" if spec.confidence_kind == "high_confidence_fact" else spec.confidence_kind
            ),
        ),
    )
    edges = [
        IntelEdge(
            source_record_id=record_id,
            edge_type="call_candidate",
            source_symbol=spec.name,
            target_symbol=ref,
            source_path=intel_file.source_path,
            confidence_kind="heuristic_candidate",
        )
        for ref in refs[:80]
    ]
    return symbol_record, chunk_record, edges


def first_sentence(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().strip("#").strip()
        if stripped:
            return stripped[:700]
    return fallback[:700]
