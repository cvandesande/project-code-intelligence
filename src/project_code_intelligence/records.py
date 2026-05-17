"""Record-building primitives used by language parsers and SARIF ingestion."""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

from project_code_intelligence import profile_context
from project_code_intelligence.language_profiles import language_file_only_metadata_keys
from project_code_intelligence.models import IntelFile, IntelRecord, JsonObject

MIN_LINE_WINDOW_CHARS = 100

# File-level metadata keys (sibling lists of functions/imports/etc) that
# language profiles produce. These belong on the file row; copying them onto
# every record from that file duplicates the same payload N times.
_FILE_ONLY_METADATA_KEYS = language_file_only_metadata_keys()


@dataclass(frozen=True)
class RecordSpec:
    record_type: str
    record_id: str
    title: str
    summary: str
    body: str
    line_start: int | None
    line_end: int | None
    symbol: str | None = None
    symbol_kind: str | None = None
    metadata: JsonObject | None = None
    confidence_kind: str = "high_confidence_fact"
    confidence: float | None = None
    tool: str | None = None
    rule_id: str | None = None
    severity: str | None = None
    analyzer: str | None = None
    analyzer_version: str | None = None
    parent_record_id: str | None = None
    file_role: str | None = None
    content_class: str | None = None


def line_offsets(text: str) -> list[int]:
    return [0, *[match.end() for match in re.finditer(r"\n", text)]]


def line_for_offset_with_index(offsets: list[int], offset: int) -> int:
    return bisect_right(offsets, offset)


def line_window_records(intel_file: IntelFile, text: str, max_chars: int, overlap_lines: int) -> list[IntelRecord]:
    if max_chars < MIN_LINE_WINDOW_CHARS:
        raise ValueError(f"line window max_chars must be at least {MIN_LINE_WINDOW_CHARS}")
    records: list[IntelRecord] = []
    lines = text.splitlines()
    current: list[tuple[int, str]] = []
    current_chars = 0
    ordinal = 0
    for lineno, line in enumerate(lines, 1):
        chunk_line = line
        if len(chunk_line) > max_chars:
            chunk_line = chunk_line[: max_chars - 22].rstrip() + " [line truncated]"
        add_chars = len(chunk_line) + 1
        if current and current_chars + add_chars > max_chars:
            ordinal += 1
            records.append(make_code_record(intel_file, current, ordinal, "fallback line window"))
            current = current[-overlap_lines:] if overlap_lines else []
            current_chars = sum(len(item[1]) + 1 for item in current)
        current.append((lineno, chunk_line))
        current_chars += add_chars
    if current:
        ordinal += 1
        records.append(make_code_record(intel_file, current, ordinal, "fallback line window"))
    return records


def common_extracts(text: str) -> JsonObject:
    configs = sorted({match.group(0) for match in re.finditer(r"\bCONFIG_[A-Za-z0-9_]+\b", text)})
    includes = sorted({
        match.group(1) for match in re.finditer(r"^\s*#\s*include\s+[<\"]([^>\"]+)[>\"]", text, re.MULTILINE)
    })
    strings = sorted({
        match.group(1) for match in re.finditer(r'"([^"\n]{4,160})"', text) if not match.group(1).startswith("$(")
    })[:80]
    log_messages = sorted({
        match.group(1)
        for match in re.finditer(
            r"\b(?:pr_err|pr_warn|pr_info|dev_err|dev_warn|fprintf|printf|syslog|ulog)\s*\([^\"\n]*\"([^\"\n]{4,180})\"",
            text,
        )
    })[:80]
    return {
        "config_symbols": configs[:120],
        "includes": includes[:80],
        "string_literals": strings,
        "log_error_messages": log_messages,
    }


def make_embedding_text(record_type: str, title: str, summary: str, metadata: JsonObject, body: str) -> str:
    parts = [f"type: {record_type}", f"title: {title}", f"summary: {summary}"]
    for key in profile_context.active_profile.embedding_metadata_keys():
        value = metadata.get(key)
        if value:
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value[:40])
            parts.append(f"{key}: {value}")
    if body:
        parts.append("content:\n" + body[:4000])
    return "\n".join(parts)


def markdown_fence_for(body: str) -> str:
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", body)), default=0)
    return "`" * max(3, longest + 1)


def make_record(intel_file: IntelFile, spec: RecordSpec) -> IntelRecord:
    file_role = spec.file_role or intel_file.file_role
    content_class = spec.content_class or intel_file.content_class
    metadata = dict(spec.metadata or {})
    metadata.update({
        key: value
        for key, value in intel_file.metadata.items()
        if key not in metadata and key not in _FILE_ONLY_METADATA_KEYS
    })
    if spec.symbol and "symbol" not in metadata:
        metadata["symbol"] = spec.symbol
    if spec.symbol_kind and "symbol_kind" not in metadata:
        metadata["symbol_kind"] = spec.symbol_kind
    display = [
        f"# {spec.title}",
        "",
        f"- Repo: `{intel_file.repo}`",
        f"- Role: `{intel_file.repo_role}`",
        f"- File role: `{file_role}`",
        f"- Content class: `{content_class}`",
    ]
    if spec.line_start is not None:
        display.append(f"- Lines: {spec.line_start}-{spec.line_end}")
    if spec.symbol:
        display.append(f"- Symbol: `{spec.symbol}`")
    if spec.rule_id:
        display.append(f"- Rule: `{spec.rule_id}`")
    fence = intel_file.language if intel_file.language not in {"doc", "text"} else ""
    fence_marker = markdown_fence_for(spec.body)
    display_content = "\n".join(display) + f"\n\n{fence_marker}{fence}\n{spec.body}\n{fence_marker}"
    embedding_text = make_embedding_text(spec.record_type, spec.title, spec.summary, metadata, spec.body)
    return IntelRecord(
        collection=intel_file.collection,
        source_path=intel_file.source_path,
        language=intel_file.language,
        file_role=file_role,
        content_class=content_class,
        record_type=spec.record_type,
        record_id=spec.record_id,
        parent_record_id=spec.parent_record_id,
        title=spec.title,
        summary=spec.summary,
        embedding_text=embedding_text,
        display_content=display_content,
        line_start=spec.line_start,
        line_end=spec.line_end,
        symbol=spec.symbol,
        symbol_kind=spec.symbol_kind,
        confidence_kind=spec.confidence_kind,
        confidence=spec.confidence,
        tool=spec.tool,
        rule_id=spec.rule_id,
        severity=spec.severity,
        analyzer=spec.analyzer,
        analyzer_version=spec.analyzer_version,
        parser=intel_file.language,
        metadata=metadata,
    )


def make_code_record(intel_file: IntelFile, lines: list[tuple[int, str]], ordinal: int, reason: str) -> IntelRecord:
    line_start = lines[0][0]
    line_end = lines[-1][0]
    body = "\n".join(line for _lineno, line in lines)
    extracts = common_extracts(body)
    symbols = extract_referenced_symbols(body)
    metadata: JsonObject = {
        **extracts,
        "symbols_referenced": symbols[:120],
        "fallback_reason": reason,
        "chunk_ordinal": ordinal,
    }
    title = f"{intel_file.source_path}:{line_start}-{line_end}"
    summary = f"{intel_file.language} {reason} in {intel_file.source_path}:{line_start}-{line_end}"
    return make_record(
        intel_file,
        RecordSpec(
            record_type="code_chunk",
            record_id=f"{intel_file.source_path}::chunk::{line_start:06d}-{line_end:06d}",
            title=title,
            summary=summary,
            body=body,
            line_start=line_start,
            line_end=line_end,
            metadata=metadata,
            confidence_kind="approximate_fact" if reason.startswith("fallback") else "high_confidence_fact",
        ),
    )


def extract_referenced_symbols(text: str) -> list[str]:
    refs = [match.group(1) for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)]
    keywords = {
        # C/C++
        "if",
        "for",
        "while",
        "switch",
        "return",
        "sizeof",
        "defined",
        "do",
        "else",
        "case",
        "catch",
        # Go anonymous function literals: `func() { ... }`
        "func",
        # JS/TS: `function() { ... }`, typeof(x), new Foo(), delete(x), void(0), throw(e), await(p)
        "function",
        "typeof",
        "instanceof",
        "new",
        "delete",
        "void",
        "throw",
        "yield",
        "await",
        # Rust
        "fn",
        "match",
        "loop",
        # Python
        "lambda",
    }
    return sorted({ref for ref in refs if ref not in keywords})[:160]
