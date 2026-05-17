"""C-like language parser helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from operator import itemgetter

from project_code_intelligence.models import IntelEdge, IntelFile, IntelRecord, JsonObject
from project_code_intelligence.parsers.core import (
    SymbolChunkSpec,
    bounded_brace_body,
    make_symbol_chunk,
    string_items,
)
from project_code_intelligence.parsers.security import security_api_refs
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
            "security_sensitive_apis": security_api_refs(body, language=intel_file.language),
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


# Go builtins (https://pkg.go.dev/builtin). Edges to these names are never
# resolvable to a definition in this codebase — and a user-defined symbol with
# the same name (e.g. a routeRuleErrors.append method) would otherwise capture
# every call to the builtin via the name-based edge resolver.
GO_BUILTIN_NAMES: frozenset[str] = frozenset({
    "append",
    "cap",
    "clear",
    "close",
    "complex",
    "copy",
    "delete",
    "imag",
    "len",
    "make",
    "max",
    "min",
    "new",
    "panic",
    "print",
    "println",
    "real",
    "recover",
})


def go_records(
    intel_file: IntelFile, text: str, max_chars: int, overlap_lines: int
) -> tuple[list[IntelRecord], list[IntelEdge]]:
    records: list[IntelRecord] = []
    edges: list[IntelEdge] = []
    lines = text.splitlines()
    offsets = line_offsets(text)
    # Package name and import list live on the file row via the language profile
    # (go_package / go_imports). Per-record metadata stays minimal.
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
                metadata={"body_truncated": truncated},
                confidence_kind="approximate_fact",
                non_resolvable_targets=GO_BUILTIN_NAMES,
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
                metadata={"body_truncated": truncated},
                confidence_kind="approximate_fact",
                non_resolvable_targets=GO_BUILTIN_NAMES,
            ),
        )
        records.extend([symbol, chunk])
    if not records:
        records.extend(line_window_records(intel_file, text, max_chars, overlap_lines))
    return records, edges


# Rust keywords + common Option/Result/&str/String/Iter methods. Edges to these
# names are pure noise — they appear in nearly every function body and the
# name-based SQL resolver would otherwise bind them to any user-defined symbol
# that happens to share the name.
RUST_NON_RESOLVABLE_NAMES: frozenset[str] = frozenset({
    # Keywords
    "as",
    "async",
    "await",
    "break",
    "const",
    "continue",
    "crate",
    "dyn",
    "else",
    "enum",
    "extern",
    "false",
    "fn",
    "for",
    "if",
    "impl",
    "in",
    "let",
    "loop",
    "match",
    "mod",
    "move",
    "mut",
    "pub",
    "ref",
    "return",
    "self",
    "Self",
    "static",
    "struct",
    "super",
    "trait",
    "true",
    "type",
    "unsafe",
    "use",
    "where",
    "while",
    # Common Option/Result methods
    "and_then",
    "as_deref",
    "as_deref_mut",
    "err",
    "expect",
    "flatten",
    "get_or_insert",
    "get_or_insert_with",
    "insert",
    "inspect",
    "is_err",
    "is_none",
    "is_ok",
    "is_some",
    "map",
    "map_err",
    "map_or",
    "map_or_else",
    "None",
    "ok",
    "ok_or",
    "ok_or_else",
    "Err",
    "Ok",
    "or",
    "or_else",
    "replace",
    "Some",
    "take",
    "transpose",
    "unwrap",
    "unwrap_or",
    "unwrap_or_default",
    "unwrap_or_else",
    # Common iterator / collection methods
    "collect",
    "contains",
    "ends_with",
    "filter",
    "filter_map",
    "find",
    "fold",
    "for_each",
    "into_iter",
    "iter",
    "iter_mut",
    "len",
    "next",
    "push",
    "pop",
    "starts_with",
    # Common conversion methods
    "as_mut",
    "as_ref",
    "as_str",
    "clone",
    "from",
    "into",
    "to_owned",
    "to_str",
    "to_string",
    # Common predicate / size methods
    "is_empty",
})
RUST_SYMBOL_MAX_LINES = 1200
RUST_SYMBOL_MAX_BODY_CHARS = 24000
RUST_SUBCHUNK_MIN_LINES = 80
RUST_IMPL_HEADER_RE = re.compile(
    r"^\s*(?:unsafe\s+)?impl(?:\s*<[^>{}]+>)?\s+"
    r"(?:(?P<trait>[A-Za-z_][A-Za-z0-9_:]*(?:<[^>{}]+>)?)\s+for\s+)?"
    r"(?P<owner>[A-Za-z_][A-Za-z0-9_:]*(?:<[^>{}]+>)?)"
)
RUST_ITEM_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+|unsafe\s+|const\s+)?"
    r"(?P<kind>fn|struct|enum|trait|impl)\b(?P<rest>.*)"
)
RUST_QUALIFIED_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:<[^>\n(){};]+>)?(?:::[A-Za-z_][A-Za-z0-9_]*)+)\s*\(")
RUST_SELF_METHOD_CALL_RE = re.compile(r"\bself\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")


@dataclass(frozen=True)
class RustSubchunkContext:
    intel_file: IntelFile
    parent_record_id: str
    symbol: str
    symbol_kind: str
    max_chars: int
    overlap_lines: int
    file_role: str | None


@dataclass(frozen=True)
class RustParseContext:
    intel_file: IntelFile
    lines: list[str]
    max_chars: int
    overlap_lines: int
    test_ranges: list[tuple[int, int]]
    impl_ranges: list[tuple[int, int, str | None, str | None]]


def rust_strip_generic_args(value: str) -> str:
    return value.split("<", 1)[0].strip()


def rust_impl_parts(line: str) -> tuple[str | None, str | None]:
    match = RUST_IMPL_HEADER_RE.match(line.split("{", 1)[0])
    if not match:
        return None, None
    trait = match.group("trait")
    owner = match.group("owner")
    return (trait.strip() if trait else None, rust_strip_generic_args(owner))


def rust_item_name(kind: str, rest: str, line: str, fallback_line: int) -> str:
    if kind == "impl":
        trait, owner = rust_impl_parts(line)
        return trait or owner or f"impl_at_{fallback_line}"
    match = re.match(r"\s+([A-Za-z_][A-Za-z0-9_]*)", rest)
    return match.group(1) if match else f"{kind}_at_{fallback_line}"


def rust_has_brace_body(lines: list[str], start_idx: int, *, max_lookahead: int = 25) -> bool:
    buffer = ""
    for line in lines[start_idx : min(len(lines), start_idx + max_lookahead)]:
        buffer += line
        brace = buffer.find("{")
        semicolon = buffer.find(";")
        if brace >= 0:
            return semicolon < 0 or brace < semicolon
        if semicolon >= 0:
            return False
    return False


def rust_test_attribute_before(lines: list[str], idx: int) -> bool:
    cursor = idx - 1
    while cursor >= 0:
        stripped = lines[cursor].strip()
        if not stripped:
            cursor -= 1
            continue
        if not stripped.startswith("#"):
            return False
        if re.match(r"#\s*\[\s*(?:[A-Za-z_][A-Za-z0-9_]*::)?test\b", stripped):
            return True
        cursor -= 1
    return False


def rust_cfg_test_ranges(lines: list[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for idx, line in enumerate(lines):
        if not re.search(r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]", line):
            continue
        cursor = idx + 1
        while cursor < len(lines) and (not lines[cursor].strip() or lines[cursor].strip().startswith("#")):
            cursor += 1
        if cursor >= len(lines) or not re.match(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\b", lines[cursor]):
            continue
        if not rust_has_brace_body(lines, cursor):
            continue
        line_end, _body, _truncated = bounded_brace_body(lines, cursor, max_lines=RUST_SYMBOL_MAX_LINES)
        ranges.append((cursor + 1, line_end))
    return ranges


def rust_line_in_ranges(line: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= line <= end for start, end in ranges)


def rust_impl_ranges(lines: list[str]) -> list[tuple[int, int, str | None, str | None]]:
    ranges: list[tuple[int, int, str | None, str | None]] = []
    for idx, line in enumerate(lines):
        item = RUST_ITEM_RE.match(line)
        if not item or item.group("kind") != "impl" or not rust_has_brace_body(lines, idx):
            continue
        trait, owner = rust_impl_parts(line)
        line_end, _body, _truncated = bounded_brace_body(lines, idx, max_lines=RUST_SYMBOL_MAX_LINES)
        ranges.append((idx + 1, line_end, trait, owner))
    return ranges


def rust_enclosing_impl(
    line: int, ranges: list[tuple[int, int, str | None, str | None]]
) -> tuple[str | None, str | None]:
    matches = [item for item in ranges if item[0] < line <= item[1]]
    if not matches:
        return None, None
    _start, _end, trait, owner = max(matches, key=itemgetter(0))
    return trait, owner


def rust_qualified_method_symbol(name: str, *, impl_trait: str | None, impl_owner: str | None) -> str:
    qualifier = impl_trait or impl_owner
    return f"{qualifier}::{name}" if qualifier else name


def rust_referenced_symbols(body: str, *, self_type: str | None) -> list[str]:
    qualified: set[str] = set()
    for match in RUST_QUALIFIED_CALL_RE.finditer(body):
        symbol = str(match.group(1))
        if self_type and symbol.startswith("Self::"):
            symbol = f"{self_type}{symbol.removeprefix('Self')}"
        qualified.add(symbol)
    if self_type:
        for match in RUST_SELF_METHOD_CALL_RE.finditer(body):
            method = str(match.group(1))
            if method not in RUST_NON_RESOLVABLE_NAMES:
                qualified.add(f"{self_type}::{method}")
    qualified_bare = {symbol.rsplit("::", 1)[-1] for symbol in qualified}
    bare = {
        symbol
        for symbol in extract_referenced_symbols(body)
        if symbol not in qualified_bare and symbol not in RUST_NON_RESOLVABLE_NAMES
    }
    return sorted((qualified | bare) - RUST_NON_RESOLVABLE_NAMES)[:160]


def rust_symbol_subchunks(
    context: RustSubchunkContext,
    line_start: int,
    line_end: int,
    lines: list[str],
) -> list[IntelRecord]:
    body_lines = [(line_no, lines[line_no - 1]) for line_no in range(line_start, line_end + 1)]
    if not body_lines or (
        line_end - line_start + 1 <= RUST_SUBCHUNK_MIN_LINES
        and sum(len(line) + 1 for _no, line in body_lines) <= context.max_chars
    ):
        return []
    records: list[IntelRecord] = []
    current: list[tuple[int, str]] = []
    current_chars = 0
    ordinal = 0
    for line_no, line in body_lines:
        chunk_line = (
            line[: context.max_chars - 22].rstrip() + " [line truncated]" if len(line) > context.max_chars else line
        )
        add_chars = len(chunk_line) + 1
        if current and current_chars + add_chars > context.max_chars:
            ordinal += 1
            records.append(rust_symbol_subchunk_record(context, current, ordinal))
            current = current[-context.overlap_lines :] if context.overlap_lines else []
            current_chars = sum(len(item[1]) + 1 for item in current)
        current.append((line_no, chunk_line))
        current_chars += add_chars
    if current:
        ordinal += 1
        records.append(rust_symbol_subchunk_record(context, current, ordinal))
    return records


def rust_symbol_subchunk_record(
    context: RustSubchunkContext,
    lines: list[tuple[int, str]],
    ordinal: int,
) -> IntelRecord:
    line_start = lines[0][0]
    line_end = lines[-1][0]
    body = "\n".join(line for _line_no, line in lines)
    self_type = rust_strip_generic_args(context.symbol.rsplit("::", 1)[0]) if "::" in context.symbol else None
    refs = rust_referenced_symbols(body, self_type=self_type)
    metadata = {
        **common_extracts(body),
        "symbols_defined": [context.symbol],
        "symbols_referenced": refs,
        "rust_symbol_subchunk": True,
        "chunk_ordinal": ordinal,
    }
    return make_record(
        context.intel_file,
        RecordSpec(
            record_type="code_chunk",
            record_id=(
                f"{context.intel_file.source_path}::rust_symbol_chunk::{context.symbol}::{line_start:06d}-{line_end:06d}"
            ),
            title=f"{context.symbol} chunk {ordinal} in {context.intel_file.source_path}:{line_start}-{line_end}",
            summary=f"Rust symbol chunk for {context.symbol}",
            body=body,
            line_start=line_start,
            line_end=line_end,
            symbol=context.symbol,
            symbol_kind=context.symbol_kind,
            metadata=metadata,
            parent_record_id=context.parent_record_id,
            confidence_kind="approximate_fact",
            file_role=context.file_role,
        ),
    )


def rust_item_records(
    context: RustParseContext, idx: int, line: str, match: re.Match[str]
) -> tuple[list[IntelRecord], list[IntelEdge]]:
    kind = match.group("kind")
    if kind == "fn" and not rust_has_brace_body(context.lines, idx):
        return [], []
    name = rust_item_name(kind, match.group("rest"), line, idx + 1)
    line_end, body, truncated = (
        bounded_brace_body(context.lines, idx, max_lines=RUST_SYMBOL_MAX_LINES, max_chars=RUST_SYMBOL_MAX_BODY_CHARS)
        if rust_has_brace_body(context.lines, idx)
        else (idx + 1, line, False)
    )
    impl_trait, impl_owner = rust_enclosing_impl(idx + 1, context.impl_ranges) if kind == "fn" else (None, None)
    bare_name = name
    if kind == "fn" and (impl_trait or impl_owner):
        name = rust_qualified_method_symbol(name, impl_trait=impl_trait, impl_owner=impl_owner)
        kind = "method"
    test_record = rust_test_attribute_before(context.lines, idx) or rust_line_in_ranges(idx + 1, context.test_ranges)
    metadata: JsonObject = {
        "body_truncated": truncated,
        "qualified_symbol": name,
        "rust_symbol_name": bare_name,
    }
    if impl_owner:
        metadata["impl_owner"] = impl_owner
    if impl_trait:
        metadata["impl_trait"] = impl_trait
    if test_record:
        metadata["rust_test"] = True
    refs = rust_referenced_symbols(body, self_type=impl_owner)
    symbol, chunk, symbol_edges = make_symbol_chunk(
        context.intel_file,
        SymbolChunkSpec(
            language_label="Rust",
            name=name,
            kind=kind,
            line_start=idx + 1,
            line_end=line_end,
            body=body,
            metadata=metadata,
            confidence_kind="approximate_fact",
            non_resolvable_targets=RUST_NON_RESOLVABLE_NAMES,
            referenced_symbols=refs,
            file_role="test" if test_record else None,
        ),
    )
    records = [symbol, chunk]
    records.extend(
        rust_symbol_subchunks(
            RustSubchunkContext(
                intel_file=context.intel_file,
                parent_record_id=symbol.record_id,
                symbol=name,
                symbol_kind=kind,
                max_chars=context.max_chars,
                overlap_lines=context.overlap_lines,
                file_role="test" if test_record else None,
            ),
            idx + 1,
            line_end,
            context.lines,
        )
    )
    return records, symbol_edges


def rust_records(
    intel_file: IntelFile, text: str, max_chars: int, overlap_lines: int
) -> tuple[list[IntelRecord], list[IntelEdge]]:
    records: list[IntelRecord] = []
    edges: list[IntelEdge] = []
    lines = text.splitlines()
    context = RustParseContext(
        intel_file=intel_file,
        lines=lines,
        max_chars=max_chars,
        overlap_lines=overlap_lines,
        test_ranges=rust_cfg_test_ranges(lines),
        impl_ranges=rust_impl_ranges(lines),
    )
    for idx, line in enumerate(lines):
        match = RUST_ITEM_RE.match(line)
        if not match:
            continue
        symbol_records, symbol_edges = rust_item_records(context, idx, line, match)
        records.extend(symbol_records)
        edges.extend(symbol_edges)
    if not records:
        records.extend(line_window_records(intel_file, text, max_chars, overlap_lines))
    return records, edges
