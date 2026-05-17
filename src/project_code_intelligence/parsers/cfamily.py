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


def _comment_padding(char: str) -> str:
    return "\n" if char == "\n" else " "


def _append_block_comment_padding(text: str, idx: int, out: list[str]) -> tuple[int, bool]:
    char = text[idx]
    nxt = text[idx + 1] if idx + 1 < len(text) else ""
    if char == "*" and nxt == "/":
        out.extend((" ", " "))
        return idx + 2, False
    out.append(_comment_padding(char))
    return idx + 1, True


def _append_line_comment_padding(text: str, idx: int, out: list[str]) -> int:
    out.extend((" ", " "))
    idx += 2
    while idx < len(text) and text[idx] != "\n":
        out.append(" ")
        idx += 1
    return idx


def _append_quoted_char(
    text: str,
    idx: int,
    out: list[str],
    quote: str,
    *,
    escape: bool,
) -> tuple[int, str | None, bool]:
    char = text[idx]
    out.append(char)
    if escape:
        return idx + 1, quote, False
    if char == "\\" and quote != "`":
        return idx + 1, quote, True
    if char == quote:
        return idx + 1, None, False
    return idx + 1, quote, False


def strip_c_like_comments(text: str) -> str:
    """Remove // and /* */ comments while preserving strings and line positions."""
    out: list[str] = []
    idx = 0
    in_block_comment = False
    quote: str | None = None
    escape = False
    while idx < len(text):
        char = text[idx]
        nxt = text[idx + 1] if idx + 1 < len(text) else ""
        if in_block_comment:
            idx, in_block_comment = _append_block_comment_padding(text, idx, out)
            continue
        if quote:
            idx, quote, escape = _append_quoted_char(text, idx, out, quote, escape=escape)
            continue
        if char == "/" and nxt == "*":
            out.extend((" ", " "))
            idx += 2
            in_block_comment = True
            continue
        if char == "/" and nxt == "/":
            idx = _append_line_comment_padding(text, idx, out)
            continue
        if char in {'"', "'", "`"}:
            quote = char
        out.append(char)
        idx += 1
    return "".join(out)


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
        refs = extract_referenced_symbols(strip_c_like_comments(body))
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


GO_PACKAGE_RE = re.compile(r"(?m)^\s*package\s+([A-Za-z_][A-Za-z0-9_]*)")
GO_FUNC_RE = re.compile(
    r"^\s*func\s+(?:\((?P<receiver>[^)]*)\)\s*)?"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:\[[^\]]+\])?\s*\("
)
GO_RECEIVER_RE = re.compile(
    r"^\s*(?:(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+)?"
    r"(?P<pointer>\*)?\s*(?:(?:[A-Za-z_][A-Za-z0-9_]*)\.)?"
    r"(?P<type>[A-Za-z_][A-Za-z0-9_]*)"
)


def go_package_name(intel_file: IntelFile, text: str) -> str | None:
    value = intel_file.metadata.get("go_package")
    if isinstance(value, str) and value:
        return value
    match = GO_PACKAGE_RE.search(text)
    return match.group(1) if match else None


def go_receiver_metadata(receiver: str, name: str) -> JsonObject:
    match = GO_RECEIVER_RE.match(receiver)
    if not match:
        return {"go_receiver": receiver, "go_symbol_name": name, "qualified_symbol": name}
    receiver_type = match.group("type")
    qualified_symbol = f"{receiver_type}.{name}"
    metadata: JsonObject = {
        "go_receiver_type": receiver_type,
        "go_receiver_pointer": bool(match.group("pointer")),
        "go_symbol_name": name,
        "qualified_symbol": qualified_symbol,
    }
    receiver_name = match.group("name")
    if receiver_name:
        metadata["go_receiver_name"] = receiver_name
    return metadata


def go_function_symbol_spec(
    lines: list[str],
    idx: int,
    *,
    package: str | None,
) -> SymbolChunkSpec | None:
    match = GO_FUNC_RE.match(lines[idx])
    if not match:
        return None
    name = match.group("name")
    receiver = match.group("receiver")
    line_end, body, truncated = bounded_brace_body(lines, idx)
    metadata: JsonObject = {"body_truncated": truncated, "qualified_symbol": name, "go_symbol_name": name}
    kind = "function"
    if package:
        metadata["go_package"] = package
    if receiver:
        kind = "method"
        metadata.update(go_receiver_metadata(receiver, name))
    return SymbolChunkSpec(
        language_label="Go",
        name=name,
        kind=kind,
        line_start=idx + 1,
        line_end=line_end,
        body=body,
        metadata=metadata,
        confidence_kind="approximate_fact",
        non_resolvable_targets=GO_BUILTIN_NAMES,
        referenced_symbols=extract_referenced_symbols(strip_c_like_comments(body)),
    )


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
    package = go_package_name(intel_file, text)
    for idx, _line in enumerate(lines):
        spec = go_function_symbol_spec(lines, idx, package=package)
        if spec is None:
            continue
        symbol, chunk, symbol_edges = make_symbol_chunk(
            intel_file,
            spec,
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
    "all",
    "any",
    "as",
    "async",
    "await",
    "break",
    "cfg",
    "cfg_attr",
    "const",
    "continue",
    "crate",
    "doc",
    "dyn",
    "else",
    "enum",
    "extern",
    "false",
    "feature",
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
RUST_RECEIVER_METHOD_CALL_RE = re.compile(
    r"\b(?!self\b|Self\b)[A-Za-z_][A-Za-z0-9_]*\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
RUST_FN_PARAM_START_RE = re.compile(r"\bfn\s+[A-Za-z_][A-Za-z0-9_]*(?:\s*<[^>{};()]+>)?\s*\(")
RUST_CLOSURE_PARAMS_RE = re.compile(r"(?:^|[\s=({[,])(?:move\s+)?\|(?P<params>[^|\n{};]{0,240})\|", re.MULTILINE)
RUST_LOCAL_CLOSURE_BINDING_RE = re.compile(
    r"\blet\s+(?:mut\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=;]+)?=\s*(?:move\s+)?\|"
)
RUST_BINDING_IDENTIFIER_RE = re.compile(r"\b(?:r#)?([A-Za-z_][A-Za-z0-9_]*)\b")
RUST_DELIMITER_CLOSE_FOR = {"(": ")", "[": "]", "{": "}", "<": ">"}
RUST_DOC_FENCE_COLLAPSED = "[rustdoc example collapsed]"
RUST_DOC_FENCE_MARKERS = ("```", "~~~")


@dataclass(frozen=True)
class RustSubchunkContext:
    intel_file: IntelFile
    parent_record_id: str
    symbol: str
    symbol_kind: str
    impl_qualifier: str | None
    local_methods: frozenset[str]
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
    impl_methods: dict[tuple[int, int], frozenset[str]]


@dataclass(frozen=True)
class RustImplReferenceContext:
    trait: str | None
    owner: str | None
    qualifier: str | None
    local_methods: frozenset[str]


def rust_doc_comment_parts(line: str, *, in_block_doc: bool) -> tuple[str, str] | None:
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]
    if stripped.startswith(("///", "//!")):
        return f"{indent}{stripped[:3]}", stripped[3:].lstrip()
    if stripped.startswith(("/**", "/*!")):
        return f"{indent}{stripped[:3]}", stripped[3:].lstrip()
    if in_block_doc:
        if stripped.startswith("*"):
            return f"{indent}*", stripped[1:].lstrip()
        return indent.rstrip(), stripped
    return None


def rust_doc_fence_boundary(content: str) -> bool:
    stripped = content.strip()
    return any(stripped.startswith(marker) for marker in RUST_DOC_FENCE_MARKERS)


def rust_doc_marker_line(prefix: str) -> str:
    return f"{prefix} {RUST_DOC_FENCE_COLLAPSED}" if prefix else RUST_DOC_FENCE_COLLAPSED


def rust_collapse_doc_example_lines(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    collapsed: list[tuple[int, str]] = []
    in_block_doc = False
    in_doc_fence = False
    for line_no, line in lines:
        stripped = line.lstrip()
        doc_parts = rust_doc_comment_parts(line, in_block_doc=in_block_doc)
        if doc_parts is not None and rust_doc_fence_boundary(doc_parts[1]):
            collapsed.append((line_no, line))
            if not in_doc_fence:
                collapsed.append((line_no, rust_doc_marker_line(doc_parts[0])))
            in_doc_fence = not in_doc_fence
        elif in_doc_fence and doc_parts is not None:
            pass
        else:
            collapsed.append((line_no, line))

        starts_block_doc = stripped.startswith(("/**", "/*!"))
        if starts_block_doc and "*/" not in stripped:
            in_block_doc = True
        if in_block_doc and "*/" in stripped:
            in_block_doc = False
    return collapsed


def rust_collapse_doc_examples(body: str) -> str:
    lines = list(enumerate(body.splitlines(), 1))
    return "\n".join(line for _line_no, line in rust_collapse_doc_example_lines(lines))


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


def rust_impl_range_at_start(
    line: int, ranges: list[tuple[int, int, str | None, str | None]]
) -> tuple[int, int, str | None, str | None] | None:
    for item in ranges:
        if item[0] == line:
            return item
    return None


def rust_enclosing_impl_range(
    line: int, ranges: list[tuple[int, int, str | None, str | None]]
) -> tuple[int, int, str | None, str | None] | None:
    matches = [item for item in ranges if item[0] < line <= item[1]]
    if not matches:
        return None
    return max(matches, key=itemgetter(0))


def rust_impl_method_names(
    lines: list[str],
    impl_ranges: list[tuple[int, int, str | None, str | None]],
) -> dict[tuple[int, int], frozenset[str]]:
    methods_by_range: dict[tuple[int, int], set[str]] = {
        (start, end): set() for start, end, _trait, _owner in impl_ranges
    }
    for idx, line in enumerate(lines):
        match = RUST_ITEM_RE.match(line)
        if not match or match.group("kind") != "fn" or not rust_has_brace_body(lines, idx):
            continue
        impl_range = rust_enclosing_impl_range(idx + 1, impl_ranges)
        if impl_range is None:
            continue
        start, end, _trait, _owner = impl_range
        methods_by_range[start, end].add(rust_item_name("fn", match.group("rest"), line, idx + 1))
    return {key: frozenset(value) for key, value in methods_by_range.items()}


def rust_impl_reference_context(context: RustParseContext, idx: int, kind: str) -> RustImplReferenceContext:
    impl_range = (
        rust_impl_range_at_start(idx + 1, context.impl_ranges)
        if kind == "impl"
        else rust_enclosing_impl_range(idx + 1, context.impl_ranges)
    )
    if impl_range is None:
        return RustImplReferenceContext(trait=None, owner=None, qualifier=None, local_methods=frozenset())
    start, end, trait, owner = impl_range
    return RustImplReferenceContext(
        trait=trait,
        owner=owner,
        qualifier=trait or owner,
        local_methods=context.impl_methods.get((start, end), frozenset()),
    )


def rust_qualified_method_symbol(name: str, *, impl_trait: str | None, impl_owner: str | None) -> str:
    qualifier = impl_trait or impl_owner
    return f"{qualifier}::{name}" if qualifier else name


def rust_advance_quote_state(char: str, quote: str, *, escape: bool) -> tuple[str | None, bool]:
    if escape:
        return quote, False
    if char == "\\" and quote != "`":
        return quote, True
    if char == quote:
        return None, False
    return quote, False


def rust_lifetime_at(text: str, idx: int) -> bool:
    if idx + 1 >= len(text) or text[idx] != "'":
        return False
    if not (text[idx + 1].isalpha() or text[idx + 1] == "_"):
        return False
    cursor = idx + 2
    while cursor < len(text) and (text[cursor].isalnum() or text[cursor] == "_"):
        cursor += 1
    return cursor >= len(text) or text[cursor] != "'"


def rust_quote_at(text: str, idx: int) -> str | None:
    char = text[idx]
    if char in {'"', "`"}:
        return char
    if char == "'" and not rust_lifetime_at(text, idx):
        return char
    return None


def rust_delimited_text(text: str, open_idx: int, open_char: str, close_char: str) -> str | None:
    if open_idx >= len(text) or text[open_idx] != open_char:
        return None
    depth = 0
    quote: str | None = None
    escape = False
    for idx in range(open_idx, len(text)):
        char = text[idx]
        if quote:
            quote, escape = rust_advance_quote_state(char, quote, escape=escape)
            continue
        if next_quote := rust_quote_at(text, idx):
            quote = next_quote
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : idx]
    return None


def rust_update_delimiter_stack(char: str, stack: list[str]) -> None:
    closer = RUST_DELIMITER_CLOSE_FOR.get(char)
    if closer:
        stack.append(closer)
    elif stack and char == stack[-1]:
        _ = stack.pop()


def rust_split_top_level_commas(text: str) -> list[str]:
    items: list[str] = []
    start = 0
    delimiter_stack: list[str] = []
    quote: str | None = None
    escape = False
    for idx, char in enumerate(text):
        if quote:
            quote, escape = rust_advance_quote_state(char, quote, escape=escape)
            continue
        if next_quote := rust_quote_at(text, idx):
            quote = next_quote
        elif char == "," and not delimiter_stack:
            items.append(text[start:idx])
            start = idx + 1
        else:
            rust_update_delimiter_stack(char, delimiter_stack)
    items.append(text[start:])
    return items


def rust_pattern_binding_names(pattern: str) -> set[str]:
    names: set[str] = set()
    for match in RUST_BINDING_IDENTIFIER_RE.finditer(pattern):
        name = match.group(1)
        if name in RUST_NON_RESOLVABLE_NAMES or name[:1].isupper():
            continue
        names.add(name)
    return names


def rust_parameter_binding_names(params: str, *, include_untyped: bool) -> set[str]:
    names: set[str] = set()
    for param in rust_split_top_level_commas(params):
        value = param.strip()
        if not value:
            continue
        if ":" in value:
            pattern = value.split(":", 1)[0]
        elif include_untyped:
            pattern = value
        else:
            continue
        names.update(rust_pattern_binding_names(pattern))
    return names


def rust_function_parameter_names(body: str) -> set[str]:
    names: set[str] = set()
    for match in RUST_FN_PARAM_START_RE.finditer(body):
        params = rust_delimited_text(body, match.end() - 1, "(", ")")
        if params is not None:
            names.update(rust_parameter_binding_names(params, include_untyped=False))
    return names


def rust_closure_parameter_names(body: str) -> set[str]:
    names: set[str] = set()
    for match in RUST_CLOSURE_PARAMS_RE.finditer(body):
        params = match.group("params")
        if params.strip():
            names.update(rust_parameter_binding_names(params, include_untyped=True))
    return names


def rust_local_closure_binding_names(body: str) -> set[str]:
    return {
        match.group("name")
        for match in RUST_LOCAL_CLOSURE_BINDING_RE.finditer(body)
        if not match.group("name")[:1].isupper()
    }


def rust_local_callable_names(body: str) -> frozenset[str]:
    return frozenset(
        rust_function_parameter_names(body)
        | rust_closure_parameter_names(body)
        | rust_local_closure_binding_names(body)
    )


def _padding_text(text: str) -> str:
    return "".join("\n" if char == "\n" else " " for char in text)


def rust_attribute_bracket_delta(line: str) -> int:
    return line.count("[") - line.count("]")


def strip_rust_attributes(text: str) -> str:
    out: list[str] = []
    in_attribute = False
    bracket_depth = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        starts_attribute = stripped.startswith(("#[", "#!["))
        if in_attribute or starts_attribute:
            out.append(_padding_text(line))
            bracket_depth += rust_attribute_bracket_delta(line)
            in_attribute = bracket_depth > 0
        else:
            out.append(line)
    return "".join(out)


def rust_receiver_method_names(body: str) -> frozenset[str]:
    return frozenset(match.group(1) for match in RUST_RECEIVER_METHOD_CALL_RE.finditer(body))


def rust_referenced_symbols(
    body: str,
    *,
    self_type: str | None,
    impl_qualifier: str | None = None,
    local_methods: frozenset[str] | None = None,
    defined_symbol: str | None = None,
) -> list[str]:
    body = strip_rust_attributes(strip_c_like_comments(body))
    method_names = local_methods or frozenset()
    defined_bare = defined_symbol.rsplit("::", 1)[-1] if defined_symbol else None
    local_callable_names = rust_local_callable_names(body)
    receiver_method_names = rust_receiver_method_names(body)
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
    bare: set[str] = set()
    for symbol in extract_referenced_symbols(body):
        if (
            symbol in qualified_bare
            or symbol == defined_bare
            or symbol in local_callable_names
            or symbol in receiver_method_names
        ):
            continue
        if impl_qualifier and symbol in method_names:
            qualified.add(f"{impl_qualifier}::{symbol}")
            continue
        if symbol not in RUST_NON_RESOLVABLE_NAMES:
            bare.add(symbol)
    return sorted((qualified | bare) - RUST_NON_RESOLVABLE_NAMES)[:160]


def rust_symbol_subchunks(
    context: RustSubchunkContext,
    line_start: int,
    line_end: int,
    lines: list[str],
) -> list[IntelRecord]:
    body_lines = [(line_no, lines[line_no - 1]) for line_no in range(line_start, line_end + 1)]
    body_lines = rust_collapse_doc_example_lines(body_lines)
    if not body_lines or (
        len(body_lines) <= RUST_SUBCHUNK_MIN_LINES
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
    display_body = rust_collapse_doc_examples(body)
    self_type = rust_strip_generic_args(context.symbol.rsplit("::", 1)[0]) if "::" in context.symbol else None
    refs = rust_referenced_symbols(
        body,
        self_type=self_type,
        impl_qualifier=context.impl_qualifier,
        local_methods=context.local_methods,
        defined_symbol=context.symbol,
    )
    metadata = {
        **common_extracts(display_body),
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
            body=display_body,
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
    impl_context = rust_impl_reference_context(context, idx, kind)
    bare_name = name
    if kind == "fn" and (impl_context.trait or impl_context.owner):
        name = rust_qualified_method_symbol(name, impl_trait=impl_context.trait, impl_owner=impl_context.owner)
        kind = "method"
    test_record = rust_test_attribute_before(context.lines, idx) or rust_line_in_ranges(idx + 1, context.test_ranges)
    metadata: JsonObject = {
        "body_truncated": truncated,
        "qualified_symbol": name,
        "rust_symbol_name": bare_name,
    }
    if impl_context.owner:
        metadata["impl_owner"] = impl_context.owner
    if impl_context.trait:
        metadata["impl_trait"] = impl_context.trait
    if test_record:
        metadata["rust_test"] = True
    refs = rust_referenced_symbols(
        body,
        self_type=impl_context.owner,
        impl_qualifier=impl_context.qualifier,
        local_methods=impl_context.local_methods,
        defined_symbol=name,
    )
    display_body = rust_collapse_doc_examples(body)
    symbol, chunk, symbol_edges = make_symbol_chunk(
        context.intel_file,
        SymbolChunkSpec(
            language_label="Rust",
            name=name,
            kind=kind,
            line_start=idx + 1,
            line_end=line_end,
            body=display_body,
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
                impl_qualifier=impl_context.qualifier,
                local_methods=impl_context.local_methods,
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
    impl_ranges = rust_impl_ranges(lines)
    context = RustParseContext(
        intel_file=intel_file,
        lines=lines,
        max_chars=max_chars,
        overlap_lines=overlap_lines,
        test_ranges=rust_cfg_test_ranges(lines),
        impl_ranges=impl_ranges,
        impl_methods=rust_impl_method_names(lines, impl_ranges),
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
