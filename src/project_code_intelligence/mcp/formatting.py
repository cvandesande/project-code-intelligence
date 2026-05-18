"""Record/edge response formatters for MCP tool handlers.

Internal to the mcp package. The compact formatters drop fields that are
constant across results in a single-snapshot query (snapshot_id, collection,
repo, etc.), fields where the default value carries no signal (boolean False),
and heavy metadata that would balloon responses. Verbose mode returns the full
row.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from project_code_intelligence import db

MIN_CENTERED_SNIPPET_TERM_CHARS = 3
DEFAULT_SNIPPET_LENGTH = 300

_SNIPPET_FENCE_RE = re.compile(r"`{3,}[^\n]*\n")
_SNIPPET_CLOSE_FENCE_RE = re.compile(r"\n`{3,}[^\n]*$")

# Fields stripped in compact mode — per-result snapshot/git/repo metadata that is
# constant across all results in a single-snapshot query and redundant with the
# response envelope. Verbose mode (verbose=true) returns them.
_COMPACT_RECORD_STRIP = frozenset({
    "id",
    "snapshot_id",
    "collection",
    "repo",
    "repo_role",
    "branch",
    "commit_sha",
    "tree_sha",
    "created_at",
    "updated_at",
    "confidence",
    "match_score",
    "distance",
    "quality_penalty",
    "tool",
    "rule_id",
    "severity",
    # embedding_text duplicates display_content minus the markdown frame and is
    # truncated mid-body — useful for debugging embedding similarity, noise for
    # navigation. Verbose mode keeps it.
    "embedding_text",
    "embedding_text_truncated",
})
_COMPACT_EDGE_KEYS = (
    "edge_type",
    "direction",
    "confidence_kind",
    "source_symbol",
    "target_symbol",
    "source_record_id",
    "target_record_id",
    "source_path",
    "target_path",
    "source_line_start",
    "source_line_end",
    "target_line_start",
    "target_line_end",
    "target_resolved",
    "target_kind",
    "edge_source",
)

# Per-record metadata fields that are valuable but heavy enough to balloon a
# compact response. doc_links in particular carries every URL in a README, which
# easily dwarfs the rest of the record. Verbose mode keeps these.
_HEAVY_METADATA_KEYS = frozenset({"doc_links"})

# Boolean fields where False is the uninteresting default. Stripped from compact
# responses so the absence of the key implies False. has_embedding stays — both
# True and False carry useful signal (False == not findable via semantic search).
_STRIP_WHEN_FALSE = frozenset({
    "is_test",
    "is_doc",
    "is_generated",
    "is_vendor",
    "is_source",
    "is_build",
    "is_config",
    "is_untracked",
    "indexed_dirty",
    "display_content_truncated",
    "embedding_text_truncated",
    "content_omitted",
})

_RECORD_TYPE_DEDUP_PRIORITY: dict[str, int] = {"code_chunk": 0, "symbol_definition": 1}


def _is_compact_noise(key: str, value: object) -> bool:
    if value is None:
        return True
    if value in ([], {}):
        return True
    return value is False and key in _STRIP_WHEN_FALSE


def dedup_by_location(rows: list[db.DbRow]) -> list[db.DbRow]:
    """Keep one record per (source_path, line_start, line_end), preferring code_chunk.

    Records without line numbers are never deduplicated.
    Two passes: first find the winning record_type per location, then filter to keep
    only the first occurrence of the winner (preserving rank order).
    """
    best: dict[tuple[object, object, object], str] = {}
    for row in rows:
        line_start = row.get("line_start")
        if line_start is None:
            continue
        key = (row.get("source_path"), line_start, row.get("line_end"))
        rtype = str(row.get("record_type") or "")
        prev = best.get(key)
        if prev is None or _RECORD_TYPE_DEDUP_PRIORITY.get(rtype, 99) < _RECORD_TYPE_DEDUP_PRIORITY.get(prev, 99):
            best[key] = rtype
    seen: set[tuple[object, object, object]] = set()
    result: list[db.DbRow] = []
    for row in rows:
        line_start = row.get("line_start")
        if line_start is None:
            result.append(row)
            continue
        key = (row.get("source_path"), line_start, row.get("line_end"))
        rtype = str(row.get("record_type") or "")
        if key not in seen and rtype == best[key]:
            seen.add(key)
            result.append(row)
    return result


def _display_content_body(raw: str | None) -> str | None:
    if not raw:
        return None
    m = _SNIPPET_FENCE_RE.search(raw)
    if m:
        return _SNIPPET_CLOSE_FENCE_RE.sub("", raw[m.end() :]).rstrip() or None
    return None


def _first_snippet_match(code: str, terms: tuple[str, ...]) -> int | None:
    if not terms:
        return None
    lower_code = code.casefold()
    preferred_terms = [term for term in terms if len(term) >= MIN_CENTERED_SNIPPET_TERM_CHARS] or list(terms)
    positions = [lower_code.find(term.casefold()) for term in preferred_terms if term]
    matches = [position for position in positions if position >= 0]
    return min(matches) if matches else None


def _centered_text_window(text: str, center: int | None, length: int) -> str | None:
    if center is None or len(text) <= length:
        return text[:length].rstrip()
    start = max(0, center - (length // 2))
    end = min(len(text), start + length)
    start = max(0, end - length)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    body_length = max(0, length - len(prefix) - len(suffix))
    if body_length != end - start:
        start = max(0, min(center - (body_length // 2), len(text) - body_length))
        end = min(len(text), start + body_length)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end].rstrip()}{suffix}" or None


def _extract_snippet(
    raw: str | None,
    length: int = DEFAULT_SNIPPET_LENGTH,
    terms: tuple[str, ...] = (),
) -> str | None:
    """Return a bounded code-body snippet, centered on a matched search term when available."""
    code = _display_content_body(raw)
    if code is None:
        return None
    return _centered_text_window(code, _first_snippet_match(code, terms), length)


def _row_text(row: db.DbRow, key: str) -> str | None:
    value = row.get(key)
    return value if isinstance(value, str) else None


def compact_record(
    row: db.DbRow,
    snippet_length: int = DEFAULT_SNIPPET_LENGTH,
    snippet_terms: tuple[str, ...] = (),
    *,
    include_metadata: bool | None = True,
) -> dict[str, object]:
    """Compact-format a row. include_metadata=None is treated the same as True;
    callers that want to drop metadata pass False explicitly.
    """
    snippet = _extract_snippet(_row_text(row, "snippet_raw"), snippet_length, snippet_terms)
    out: dict[str, object] = {
        k: v
        for k, v in row.items()
        if not _is_compact_noise(k, v) and k not in _COMPACT_RECORD_STRIP and k != "snippet_raw"
    }
    if include_metadata is not False:
        metadata = out.get("metadata")
        if isinstance(metadata, dict):
            metadata_dict = cast("dict[str, object]", metadata)
            trimmed = {
                k: v for k, v in metadata_dict.items() if k not in _HEAVY_METADATA_KEYS and not _is_compact_noise(k, v)
            }
            if trimmed != metadata_dict:
                if trimmed:
                    out["metadata"] = trimmed
                else:
                    del out["metadata"]
    else:
        _ = out.pop("metadata", None)
    if snippet:
        out["snippet"] = snippet
    return out


def _verbose_record(
    row: db.DbRow,
    snippet_length: int = DEFAULT_SNIPPET_LENGTH,
    snippet_terms: tuple[str, ...] = (),
) -> dict[str, object]:
    snippet = _extract_snippet(_row_text(row, "snippet_raw"), snippet_length, snippet_terms)
    out = {k: v for k, v in row.items() if k not in {"snippet_raw", "match_score", "quality_penalty"}}
    if snippet:
        out["snippet"] = snippet
    return out


def compact_file(row: db.DbRow) -> dict[str, object]:
    return {k: v for k, v in row.items() if not _is_compact_noise(k, v)}


def format_records(
    rows: list[db.DbRow],
    *,
    verbose: bool,
    snippet_length: int = DEFAULT_SNIPPET_LENGTH,
    snippet_terms: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    fmt = _verbose_record if verbose else compact_record
    return [fmt(row, snippet_length, snippet_terms) for row in rows]


def _compact_edge(row: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key in _COMPACT_EDGE_KEYS:
        value = row.get(key)
        if value is not None and not _is_compact_noise(key, value):
            out[key] = value
    return out


def format_edges(rows: Sequence[Mapping[str, object]], *, verbose: bool) -> list[dict[str, object]]:
    if verbose:
        return [dict(row) for row in rows]
    return [_compact_edge(row) for row in rows]
