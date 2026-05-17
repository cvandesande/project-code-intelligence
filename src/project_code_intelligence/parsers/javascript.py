"""Heuristic JavaScript and TypeScript record extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass

from project_code_intelligence.models import IntelEdge, IntelFile, IntelRecord, JsonObject
from project_code_intelligence.parsers.core import advance_block_comment, advance_quote, bounded_brace_body, brace_delta
from project_code_intelligence.parsers.security import security_api_refs
from project_code_intelligence.records import (
    RecordSpec,
    common_extracts,
    line_window_records,
    make_code_record,
    make_record,
)

JS_IDENTIFIER = r"[$A-Za-z_][$A-Za-z0-9_$]*"
MIN_MEMBER_CALL_PARTS = 2
MIN_INSTANCE_MEMBER_CALL_PARTS = 3
JS_FUNCTION_DEF_RE = re.compile(
    rf"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+({JS_IDENTIFIER})"
    r"(?:<[^>{}\n]+>)?\s*\("
)
JS_CLASS_DEF_RE = re.compile(rf"^\s*(?:export\s+)?(?:default\s+)?class\s+({JS_IDENTIFIER})\b")
JS_ARROW_DEF_RE = re.compile(
    rf"^\s*(?:export\s+)?(?:const|let|var)\s+({JS_IDENTIFIER})\s*(?::[^=]+)?=\s*"
    rf"(?:async\s*)?(?:<[^>\n]+>\s*)?(?:\([^)]*\)|{JS_IDENTIFIER})\s*=>"
)
JS_FUNCTION_VALUE_RE = re.compile(
    rf"^\s*(?:export\s+)?(?:const|let|var)\s+({JS_IDENTIFIER})\s*(?::[^=]+)?=\s*"
    r"(?:async\s+)?function\b"
)
JS_EXPORTED_CONST_RE = re.compile(rf"^\s*export\s+(?:const|let|var)\s+({JS_IDENTIFIER})\b")
TS_INTERFACE_RE = re.compile(rf"^\s*(?:export\s+)?interface\s+({JS_IDENTIFIER})\b")
TS_TYPE_RE = re.compile(rf"^\s*(?:export\s+)?type\s+({JS_IDENTIFIER})\b")
TS_ENUM_RE = re.compile(rf"^\s*(?:export\s+)?enum\s+({JS_IDENTIFIER})\b")
JS_CALL_RE = re.compile(rf"(?<![A-Za-z0-9_$.])({JS_IDENTIFIER})\s*(?:<[^>\n]+>)?\(")
JS_MEMBER_CALL_RE = re.compile(rf"(?<![A-Za-z0-9_$])({JS_IDENTIFIER}(?:\.{JS_IDENTIFIER})+)\s*(?:<[^>\n]+>)?\(")
JS_OWNER_TARGET_METHODS = frozenset({"init"})
JS_UNRESOLVED_MEMBER_METHODS = frozenset({"run"})

JS_NOISE_CALLS = frozenset({
    "Array",
    "Boolean",
    "Date",
    "Error",
    "JSON",
    "Map",
    "Math",
    "Number",
    "Object",
    "Promise",
    "Reflect",
    "RegExp",
    "Set",
    "String",
    "Symbol",
    "WeakMap",
    "WeakSet",
    "afterAll",
    "afterEach",
    "assert",
    "await",
    "beforeAll",
    "beforeEach",
    "catch",
    "console",
    "constructor",
    "delete",
    "describe",
    "do",
    "else",
    "expect",
    "filter",
    "finally",
    "for",
    "forEach",
    "function",
    "get",
    "if",
    "instanceof",
    "it",
    "log",
    "map",
    "new",
    "reduce",
    "require",
    "return",
    "set",
    "switch",
    "test",
    "then",
    "throw",
    "typeof",
    "void",
    "while",
    "yield",
})


@dataclass(frozen=True)
class JsSymbolSpec:
    name: str
    kind: str
    line_start: int
    line_end: int
    body: str
    emit_edges: bool


@dataclass(frozen=True)
class JsReference:
    symbol: str
    metadata: JsonObject


@dataclass
class JsFunctionSignatureScan:
    in_block_comment: bool = False
    quote: str | None = None
    escape: bool = False
    paren_depth: int = 0
    saw_open_paren: bool = False
    params_closed: bool = False
    last_significant: str | None = None


def statement_body(lines: list[str], start_idx: int, *, max_lines: int = 80, max_chars: int = 4000) -> tuple[int, str]:
    collected: list[str] = []
    for idx in range(start_idx, min(len(lines), start_idx + max_lines)):
        line = lines[idx]
        collected.append(line)
        if ";" in line:
            break
        if idx > start_idx and not line.strip():
            break
    body = "\n".join(collected)
    if len(body) > max_chars:
        body = body[: max_chars - 37].rstrip() + "\n/* symbol candidate truncated */"
    return start_idx + len(collected), body


def line_has_statement_terminator(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped and not stripped.startswith("//") and ";" in stripped)


def has_open_brace_before_terminator(lines: list[str], start_idx: int, *, max_signature_lines: int = 40) -> bool:
    in_block_comment = False
    for idx in range(start_idx, min(len(lines), start_idx + max_signature_lines)):
        line = lines[idx]
        _delta, in_block_comment, saw_open = brace_delta(line, in_block_comment=in_block_comment)
        if saw_open:
            return True
        if line_has_statement_terminator(line):
            return False
        if idx > start_idx and not line.strip():
            return False
    return False


def function_body_is_ready(last_significant: str | None) -> bool:
    return last_significant not in {None, ":", "=", "("}


def advance_js_non_code(line: str, char_idx: int, state: JsFunctionSignatureScan) -> tuple[int, bool] | None:
    char = line[char_idx]
    nxt = line[char_idx + 1] if char_idx + 1 < len(line) else ""
    if state.in_block_comment:
        next_idx, state.in_block_comment = advance_block_comment(line, char_idx)
        return next_idx, False
    if state.quote:
        next_idx, state.quote, state.escape = advance_quote(char_idx, char, state.quote, escape=state.escape)
        return next_idx, False
    if char == "/" and nxt == "*":
        state.in_block_comment = True
        return char_idx + 2, False
    if char == "/" and nxt == "/":
        return len(line), True
    if char in {'"', "'", "`"}:
        state.quote = char
        return char_idx + 1, False
    return None


def update_js_signature_state(char: str, state: JsFunctionSignatureScan) -> bool:
    found_body_open = False
    if char == "(" and not state.params_closed:
        state.paren_depth += 1
        state.saw_open_paren = True
    elif char == ")" and state.saw_open_paren and not state.params_closed:
        state.paren_depth = max(0, state.paren_depth - 1)
        state.params_closed = state.paren_depth == 0
    elif (
        char == "{"
        and (state.params_closed or not state.saw_open_paren)
        and function_body_is_ready(state.last_significant)
    ):
        found_body_open = True
    if char.strip():
        state.last_significant = char
    return found_body_open


def find_js_function_body_open(lines: list[str], start_idx: int, *, max_lines: int = 260) -> tuple[int, int] | None:
    state = JsFunctionSignatureScan()
    for idx in range(start_idx, min(len(lines), start_idx + max_lines)):
        line = lines[idx]
        char_idx = 0
        while char_idx < len(line):
            advanced = advance_js_non_code(line, char_idx, state)
            if advanced is not None:
                char_idx, stop_line = advanced
                if stop_line:
                    break
                continue
            if update_js_signature_state(line[char_idx], state):
                return idx, char_idx
            char_idx += 1
    return None


def bounded_brace_body_from_open(
    lines: list[str],
    start_idx: int,
    open_location: tuple[int, int],
    *,
    max_lines: int = 260,
    max_chars: int = 7200,
) -> tuple[int, str]:
    depth = 0
    end_idx = min(len(lines) - 1, start_idx + max_lines - 1)
    in_block_comment = False
    open_idx, open_col = open_location
    for idx in range(open_idx, min(len(lines), start_idx + max_lines)):
        line = lines[idx][open_col:] if idx == open_idx else lines[idx]
        delta, in_block_comment, _saw_open = brace_delta(line, in_block_comment=in_block_comment)
        depth += delta
        end_idx = idx
        if depth <= 0:
            break
    body = "\n".join(lines[start_idx : end_idx + 1])
    if len(body) > max_chars:
        body = body[: max_chars - 38].rstrip() + "\n/* symbol candidate truncated */"
    return end_idx + 1, body


def js_function_body(
    lines: list[str], start_idx: int, *, max_lines: int = 260, max_chars: int = 7200
) -> tuple[int, str]:
    body_open = find_js_function_body_open(lines, start_idx, max_lines=max_lines)
    if body_open is None:
        return statement_body(lines, start_idx)
    return bounded_brace_body_from_open(lines, start_idx, body_open, max_lines=max_lines, max_chars=max_chars)


def js_symbol_body(lines: list[str], start_idx: int, *, kind: str) -> tuple[int, str]:
    if kind == "function":
        return js_function_body(lines, start_idx)
    if kind in {"class", "enum"} or has_open_brace_before_terminator(lines, start_idx):
        line_end, body, _truncated = bounded_brace_body(lines, start_idx, max_lines=260, max_chars=7200)
        return line_end, body
    return statement_body(lines, start_idx)


def member_call_resolvable(parts: list[str]) -> bool:
    member = parts[-1]
    if member in JS_UNRESOLVED_MEMBER_METHODS:
        return False
    return not (len(parts) >= MIN_INSTANCE_MEMBER_CALL_PARTS and member[:1].islower())


def js_member_call_candidates(body: str) -> list[JsReference]:
    candidates: dict[str, JsReference] = {}
    for match in JS_MEMBER_CALL_RE.finditer(body):
        parts = [part.strip() for part in match.group(1).split(".") if part.strip()]
        if len(parts) < MIN_MEMBER_CALL_PARTS:
            continue
        qualifier = ".".join(parts[:-1])
        full_symbol = match.group(1)
        owner = parts[-2]
        member = parts[-1]
        if member in JS_NOISE_CALLS or member in JS_OWNER_TARGET_METHODS:
            if owner.startswith("$") or owner[:1].isupper():
                _ = candidates.setdefault(
                    owner,
                    JsReference(
                        symbol=owner,
                        metadata={"call_kind": "member_owner_call", "member": member, "qualifier": qualifier},
                    ),
                )
            continue
        metadata: JsonObject = {"call_kind": "member_call", "member": member, "qualifier": qualifier}
        if not member_call_resolvable(parts):
            metadata["target_resolvable"] = False
            metadata["full_symbol"] = full_symbol
        _ = candidates.setdefault(member, JsReference(symbol=member, metadata=metadata))
    return sorted(candidates.values(), key=lambda item: item.symbol)


def js_referenced_symbols(body: str, defined_symbol: str) -> list[JsReference]:
    refs: dict[str, JsReference] = {}
    for match in JS_CALL_RE.finditer(body):
        symbol = match.group(1)
        if symbol != defined_symbol and symbol not in JS_NOISE_CALLS:
            _ = refs.setdefault(symbol, JsReference(symbol=symbol, metadata={}))
    for ref in js_member_call_candidates(body):
        if ref.symbol != defined_symbol and ref.symbol not in JS_NOISE_CALLS:
            _ = refs.setdefault(ref.symbol, ref)
    return sorted(refs.values(), key=lambda item: item.symbol)[:120]


def covered_code_ranges(records: list[IntelRecord]) -> list[tuple[int, int]]:
    return [
        (record.line_start, record.line_end)
        for record in records
        if record.record_type == "code_chunk" and record.line_start is not None and record.line_end is not None
    ]


def coverage_line_window_records(
    intel_file: IntelFile,
    text: str,
    max_chars: int,
    overlap_lines: int,
    covered_ranges: list[tuple[int, int]],
) -> list[IntelRecord]:
    covered = {lineno for start, end in covered_ranges for lineno in range(start, end + 1)}
    lines = text.splitlines()
    records: list[IntelRecord] = []
    current: list[tuple[int, str]] = []
    current_chars = 0
    ordinal = 0
    for lineno, line in enumerate(lines, 1):
        if lineno in covered:
            if current:
                ordinal += 1
                records.append(make_code_record(intel_file, current, ordinal, "coverage line window"))
                current = []
                current_chars = 0
            continue
        if not line.strip() and not current:
            continue
        chunk_line = line
        if len(chunk_line) > max_chars:
            chunk_line = chunk_line[: max_chars - 22].rstrip() + " [line truncated]"
        add_chars = len(chunk_line) + 1
        if current and current_chars + add_chars > max_chars:
            ordinal += 1
            records.append(make_code_record(intel_file, current, ordinal, "coverage line window"))
            current = current[-overlap_lines:] if overlap_lines else []
            current_chars = sum(len(item[1]) + 1 for item in current)
        current.append((lineno, chunk_line))
        current_chars += add_chars
    if current:
        ordinal += 1
        records.append(make_code_record(intel_file, current, ordinal, "coverage line window"))
    return records


def js_symbol_records(
    intel_file: IntelFile,
    spec: JsSymbolSpec,
) -> tuple[list[IntelRecord], list[IntelEdge]]:
    refs = js_referenced_symbols(spec.body, spec.name)
    ref_symbols = [ref.symbol for ref in refs]
    metadata: JsonObject = {
        **common_extracts(spec.body),
        "symbols_defined": [spec.name],
        "symbols_referenced": ref_symbols,
        "security_sensitive_apis": security_api_refs(spec.body),
        "bounded_symbol_parser": True,
    }
    record_id = f"{intel_file.source_path}::{spec.kind}::{spec.name}::{spec.line_start:06d}"
    records = [
        make_record(
            intel_file,
            RecordSpec(
                record_type="symbol_definition",
                record_id=record_id,
                title=f"{spec.name} in {intel_file.source_path}:{spec.line_start}-{spec.line_end}",
                summary=f"{intel_file.language} {spec.kind} definition {spec.name} in {intel_file.source_path}",
                body=spec.body,
                line_start=spec.line_start,
                line_end=spec.line_end,
                symbol=spec.name,
                symbol_kind=spec.kind,
                metadata=metadata,
                confidence_kind="approximate_fact",
            ),
        ),
        make_record(
            intel_file,
            RecordSpec(
                record_type="code_chunk",
                record_id=f"{intel_file.source_path}::{spec.kind}_chunk::{spec.name}::{spec.line_start:06d}",
                title=f"{spec.name} body in {intel_file.source_path}:{spec.line_start}-{spec.line_end}",
                summary=f"{intel_file.language} {spec.kind} chunk for {spec.name}",
                body=spec.body,
                line_start=spec.line_start,
                line_end=spec.line_end,
                symbol=spec.name,
                symbol_kind=spec.kind,
                metadata=metadata,
                parent_record_id=record_id,
                confidence_kind="approximate_fact",
            ),
        ),
    ]
    edges = [
        IntelEdge(
            source_record_id=record_id,
            edge_type="call_candidate",
            source_symbol=spec.name,
            target_symbol=ref.symbol,
            source_path=intel_file.source_path,
            confidence_kind="heuristic_candidate",
            metadata=ref.metadata,
        )
        for ref in refs[:80]
    ]
    return records, edges if spec.emit_edges else []


def javascript_records(
    intel_file: IntelFile, text: str, max_chars: int, overlap_lines: int
) -> tuple[list[IntelRecord], list[IntelEdge]]:
    records: list[IntelRecord] = []
    edges: list[IntelEdge] = []
    lines = text.splitlines()
    seen: set[tuple[str, str, int]] = set()
    for idx, line in enumerate(lines):
        candidates = (
            (JS_FUNCTION_DEF_RE, "function", True),
            (JS_CLASS_DEF_RE, "class", True),
            (JS_ARROW_DEF_RE, "function", True),
            (JS_FUNCTION_VALUE_RE, "function", True),
            (TS_INTERFACE_RE, "interface", False),
            (TS_TYPE_RE, "type", False),
            (TS_ENUM_RE, "enum", False),
            (JS_EXPORTED_CONST_RE, "constant", True),
        )
        for pattern, kind, emit_edges in candidates:
            match = pattern.match(line)
            if not match:
                continue
            name = match.group(1)
            key = (kind, name, idx)
            if key in seen:
                break
            seen.add(key)
            line_end, body = js_symbol_body(lines, idx, kind=kind)
            symbol_records, symbol_edges = js_symbol_records(
                intel_file,
                JsSymbolSpec(
                    name=name,
                    kind=kind,
                    line_start=idx + 1,
                    line_end=line_end,
                    body=body,
                    emit_edges=emit_edges,
                ),
            )
            records.extend(symbol_records)
            edges.extend(symbol_edges)
            break
    if not any(record.record_type == "code_chunk" for record in records):
        records.extend(line_window_records(intel_file, text, max_chars, overlap_lines))
    else:
        records.extend(
            coverage_line_window_records(intel_file, text, max_chars, overlap_lines, covered_code_ranges(records))
        )
    return records, edges
