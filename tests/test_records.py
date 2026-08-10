"""Unit tests for `project_code_intelligence.records`.

Covers the line-window chunker, IntelRecord construction via make_record,
common_extracts regex extractors, line offset utilities, and the symbol
reference extractor.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import cast

from project_code_intelligence.models import IntelFile, IntelRecord, JsonObject, JsonValue
from project_code_intelligence.records import (
    MIN_LINE_WINDOW_CHARS,
    RecordSpec,
    common_extracts,
    extract_referenced_symbols,
    line_for_offset_with_index,
    line_offsets,
    line_window_records,
    make_code_record,
    make_embedding_text,
    make_record,
    markdown_fence_for,
    module_records,
)


def _make_intel_file(
    *,
    source_path: str = "src/foo.py",
    language: str = "python",
    file_role: str = "source",
    content_class: str = "source",
    metadata: JsonObject | None = None,
) -> IntelFile:
    return IntelFile(
        collection="test",
        repo="example",
        repo_role="project",
        branch="main",
        commit_sha="0" * 40,
        tree_sha="1" * 40,
        source_path=source_path,
        repo_rel_path=source_path,
        abs_path=Path(tempfile.gettempdir()) / source_path,
        git_blob_sha=None,
        file_sha256=None,
        size_bytes=1024,
        language=language,
        file_role=file_role,
        content_class=content_class,
        is_generated=False,
        is_vendor=False,
        is_test=False,
        is_source=True,
        is_build=False,
        is_config=False,
        is_doc=False,
        skipped_reason=None,
        metadata=metadata if metadata is not None else {},
    )


class LineOffsetsTests(unittest.TestCase):
    def test_line_offsets_for_empty_string_is_single_zero(self) -> None:
        self.assertEqual(line_offsets(""), [0])

    def test_line_offsets_records_each_newline_position(self) -> None:
        # "ab\ncd\nef" → newlines at indices 2 and 5; offsets at 0, 3, 6.
        self.assertEqual(line_offsets("ab\ncd\nef"), [0, 3, 6])

    def test_line_offsets_handles_trailing_newline(self) -> None:
        # Trailing newline yields one more offset.
        self.assertEqual(line_offsets("a\nb\n"), [0, 2, 4])

    def test_line_for_offset_with_index_returns_one_based_line(self) -> None:
        offsets = line_offsets("ab\ncd\nef")  # [0, 3, 6]
        # Byte offsets within line 1
        self.assertEqual(line_for_offset_with_index(offsets, 0), 1)
        self.assertEqual(line_for_offset_with_index(offsets, 1), 1)
        # Offset 3 lands on the start of line 2 (bisect_right with 3 returns 2)
        self.assertEqual(line_for_offset_with_index(offsets, 3), 2)
        # Offset 6 lands on the start of line 3
        self.assertEqual(line_for_offset_with_index(offsets, 6), 3)


class MarkdownFenceForTests(unittest.TestCase):
    def test_default_fence_is_three_backticks_when_body_has_none(self) -> None:
        self.assertEqual(markdown_fence_for("plain text"), "```")

    def test_fence_grows_when_body_contains_triple_backticks(self) -> None:
        body = "before ``` after"
        self.assertEqual(markdown_fence_for(body), "````")

    def test_fence_uses_longest_run_plus_one(self) -> None:
        body = "x ` y `` z ```` end"
        # Longest run is 4 backticks → fence must be 5.
        self.assertEqual(markdown_fence_for(body), "`````")


class CommonExtractsTests(unittest.TestCase):
    def test_common_extracts_collects_config_symbols(self) -> None:
        result = common_extracts("uses CONFIG_FOO and CONFIG_BAR\nelse CONFIG_FOO again")
        configs = cast("list[str]", result["config_symbols"])
        # Deduplicated and sorted.
        self.assertEqual(configs, ["CONFIG_BAR", "CONFIG_FOO"])

    def test_common_extracts_collects_includes(self) -> None:
        text = '#include <stdio.h>\n#include "local.h"\n# include <stdlib.h>\n'
        result = common_extracts(text)
        includes = cast("list[str]", result["includes"])
        self.assertEqual(includes, ["local.h", "stdio.h", "stdlib.h"])

    def test_common_extracts_keeps_string_literals_above_four_chars(self) -> None:
        # "abc" is below 4-char minimum; "hello world" qualifies. Newlines
        # between the two `"..."` segments stop the regex from spanning them.
        result = common_extracts('msg = "abc"\nlabel = "hello world"\n')
        strings = cast("list[str]", result["string_literals"])
        self.assertIn("hello world", strings)
        self.assertNotIn("abc", strings)

    def test_common_extracts_collects_log_error_messages(self) -> None:
        text = 'pr_err("boot failed: %d", err);\nprintf("ok");\n'
        result = common_extracts(text)
        logs = cast("list[str]", result["log_error_messages"])
        # The "ok" string from printf is shorter than 4 chars → ignored.
        self.assertEqual(logs, ["boot failed: %d"])

    def test_common_extracts_skips_strings_starting_with_dollar_paren(self) -> None:
        # Strings that look like shell expansions are excluded from string_literals.
        result = common_extracts('var = "$(date +%Y)" and "normal string here"')
        strings = cast("list[str]", result["string_literals"])
        self.assertIn("normal string here", strings)
        self.assertNotIn("$(date +%Y)", strings)


class ExtractReferencedSymbolsTests(unittest.TestCase):
    def test_extract_referenced_symbols_finds_call_sites(self) -> None:
        symbols = extract_referenced_symbols("foo() and bar(x) plus baz(1, 2)")
        self.assertEqual(symbols, ["bar", "baz", "foo"])

    def test_extract_referenced_symbols_skips_keywords(self) -> None:
        text = "if (cond) { for (i = 0; i < n; i++) { return foo(); } }"
        symbols = extract_referenced_symbols(text)
        # "if", "for", "return" are keywords; "foo" is not.
        self.assertEqual(symbols, ["foo"])

    def test_extract_referenced_symbols_deduplicates(self) -> None:
        symbols = extract_referenced_symbols("a() a() a()")
        self.assertEqual(symbols, ["a"])

    def test_extract_referenced_symbols_caps_at_160_entries(self) -> None:
        text = " ".join(f"sym{i}()" for i in range(200))
        symbols = extract_referenced_symbols(text)
        self.assertLessEqual(len(symbols), 160)


class MakeEmbeddingTextTests(unittest.TestCase):
    def test_make_embedding_text_includes_type_title_summary(self) -> None:
        text = make_embedding_text("code_chunk", "T", "S", {}, "body")
        self.assertIn("type: code_chunk", text)
        self.assertIn("title: T", text)
        self.assertIn("summary: S", text)
        self.assertIn("content:\n", text)
        self.assertIn("body", text)

    def test_make_embedding_text_truncates_body_to_4000_chars(self) -> None:
        body = "x" * 5000
        text = make_embedding_text("code_chunk", "T", "S", {}, body)
        # Only the first 4000 chars of body appear; nothing past it.
        self.assertIn("x" * 4000, text)
        self.assertNotIn("x" * 4001, text)

    def test_make_embedding_text_omits_content_block_for_empty_body(self) -> None:
        text = make_embedding_text("code_chunk", "T", "S", {}, "")
        self.assertNotIn("content:", text)


class LineWindowRecordsTests(unittest.TestCase):
    def test_line_window_records_rejects_max_chars_below_minimum(self) -> None:
        intel_file = _make_intel_file()
        with self.assertRaises(ValueError):
            _ = line_window_records(intel_file, "abc", max_chars=MIN_LINE_WINDOW_CHARS - 1, overlap_lines=0)

    def test_line_window_records_empty_text_returns_no_records(self) -> None:
        intel_file = _make_intel_file()
        records = line_window_records(intel_file, "", max_chars=200, overlap_lines=0)
        self.assertEqual(records, [])

    def test_line_window_records_produces_one_chunk_for_short_text(self) -> None:
        intel_file = _make_intel_file()
        text = "line one\nline two\nline three\n"
        records = line_window_records(intel_file, text, max_chars=200, overlap_lines=0)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.line_start, 1)
        self.assertEqual(record.line_end, 3)
        self.assertEqual(record.record_type, "code_chunk")
        # The fallback line-window chunker emits approximate confidence.
        self.assertEqual(record.confidence_kind, "approximate_fact")

    def test_line_window_records_splits_long_text_into_multiple_chunks(self) -> None:
        intel_file = _make_intel_file()
        # 20 lines of ~20 chars (~420 chars total). max_chars=100 yields multiple chunks.
        text = "\n".join(f"this is line {i:02d}" for i in range(1, 21))
        records = line_window_records(intel_file, text, max_chars=100, overlap_lines=0)
        self.assertGreaterEqual(len(records), 2)
        # First chunk must start at line 1; chunks must cover contiguous ranges.
        self.assertEqual(records[0].line_start, 1)
        prev_end = 0
        for record in records:
            self.assertIsNotNone(record.line_start)
            self.assertIsNotNone(record.line_end)
            line_start = record.line_start or 0
            line_end = record.line_end or 0
            self.assertGreater(line_start, prev_end)
            self.assertGreaterEqual(line_end, line_start)
            prev_end = line_end

    def test_line_window_records_overlap_lines_repeats_tail_in_next_chunk(self) -> None:
        intel_file = _make_intel_file()
        text = "\n".join(f"l{i}" for i in range(1, 31))
        records = line_window_records(intel_file, text, max_chars=100, overlap_lines=2)
        self.assertGreaterEqual(len(records), 2)
        self.assertIsNotNone(records[0].line_end)
        self.assertIsNotNone(records[1].line_start)
        first_end = records[0].line_end or 0
        second_start = records[1].line_start or 0
        # With overlap_lines=2, the second chunk replays up to 2 prior lines.
        self.assertLessEqual(second_start, first_end + 1)
        self.assertGreaterEqual(second_start, first_end - 1)

    def test_line_window_records_truncates_very_long_single_line(self) -> None:
        intel_file = _make_intel_file()
        long_line = "x" * 500
        records = line_window_records(intel_file, long_line, max_chars=200, overlap_lines=0)
        self.assertEqual(len(records), 1)
        record = records[0]
        # The truncated marker must appear in the body (via display_content).
        self.assertIn("[line truncated]", record.display_content)

    def test_line_window_records_records_carry_chunk_ordinal_metadata(self) -> None:
        intel_file = _make_intel_file()
        text = "\n".join(f"line {i}" for i in range(1, 41))
        records = line_window_records(intel_file, text, max_chars=100, overlap_lines=0)
        for index, record in enumerate(records, 1):
            self.assertEqual(record.metadata.get("chunk_ordinal"), index)
            self.assertEqual(record.metadata.get("fallback_reason"), "fallback line window")


class MakeRecordTests(unittest.TestCase):
    def test_make_record_uses_file_role_and_content_class_when_spec_unset(self) -> None:
        intel_file = _make_intel_file(file_role="test", content_class="test")
        spec = RecordSpec(
            record_type="code_chunk",
            record_id="r1",
            title="Title",
            summary="Summary",
            body="print('hi')",
            line_start=1,
            line_end=1,
        )
        record = make_record(intel_file, spec)
        self.assertEqual(record.file_role, "test")
        self.assertEqual(record.content_class, "test")

    def test_make_record_spec_overrides_file_role_and_content_class(self) -> None:
        intel_file = _make_intel_file(file_role="source", content_class="source")
        spec = RecordSpec(
            record_type="code_chunk",
            record_id="r1",
            title="T",
            summary="S",
            body="x = 1",
            line_start=None,
            line_end=None,
            file_role="config",
            content_class="config",
        )
        record = make_record(intel_file, spec)
        self.assertEqual(record.file_role, "config")
        self.assertEqual(record.content_class, "config")

    def test_make_record_merges_intel_file_metadata_excluding_file_only_keys(self) -> None:
        # symbol_kind in metadata should win over auto-derived from spec.symbol_kind.
        intel_file = _make_intel_file(metadata={"shared_key": "from_file", "language_version": "3.11"})
        spec = RecordSpec(
            record_type="code_chunk",
            record_id="r1",
            title="T",
            summary="S",
            body="",
            line_start=None,
            line_end=None,
            metadata={"spec_only": "v"},
            symbol="my_symbol",
            symbol_kind="function",
        )
        record = make_record(intel_file, spec)
        self.assertEqual(record.metadata["spec_only"], "v")
        # Non-conflicting file metadata is merged in.
        self.assertEqual(record.metadata.get("shared_key"), "from_file")
        # Symbol/symbol_kind auto-attached when missing.
        self.assertEqual(record.metadata.get("symbol"), "my_symbol")
        self.assertEqual(record.metadata.get("symbol_kind"), "function")

    def test_make_record_spec_metadata_wins_over_file_metadata_for_shared_keys(self) -> None:
        intel_file = _make_intel_file(metadata={"shared": "from_file"})
        spec = RecordSpec(
            record_type="code_chunk",
            record_id="r1",
            title="T",
            summary="S",
            body="",
            line_start=None,
            line_end=None,
            metadata={"shared": "from_spec"},
        )
        record = make_record(intel_file, spec)
        self.assertEqual(record.metadata["shared"], "from_spec")

    def test_make_record_display_content_contains_header_and_fenced_body(self) -> None:
        intel_file = _make_intel_file()
        spec = RecordSpec(
            record_type="code_chunk",
            record_id="r1",
            title="Hello",
            summary="S",
            body="print('hi')",
            line_start=5,
            line_end=5,
            symbol="hello",
            rule_id="RULE-1",
        )
        record = make_record(intel_file, spec)
        display = record.display_content
        self.assertIn("# Hello", display)
        self.assertIn("- Repo: `example`", display)
        self.assertIn("- Lines: 5-5", display)
        self.assertIn("- Symbol: `hello`", display)
        self.assertIn("- Rule: `RULE-1`", display)
        # The body is wrapped in a python-tagged fence block.
        self.assertIn("```python", display)
        self.assertIn("print('hi')", display)
        self.assertIn("```", display)

    def test_make_record_doc_language_omits_language_tag_on_fence(self) -> None:
        intel_file = _make_intel_file(language="doc")
        spec = RecordSpec(
            record_type="doc_section",
            record_id="r1",
            title="Doc",
            summary="S",
            body="# heading",
            line_start=1,
            line_end=1,
        )
        record = make_record(intel_file, spec)
        # No "```doc" tag.
        self.assertNotIn("```doc", record.display_content)


def _definition_record(intel_file: IntelFile, symbol: str, line_start: int, line_end: int) -> IntelRecord:
    return make_record(
        intel_file,
        RecordSpec(
            record_type="symbol_definition",
            record_id=f"{intel_file.source_path}::function::{symbol}::{line_start:06d}",
            title=f"{symbol} def",
            summary="S",
            body="body",
            line_start=line_start,
            line_end=line_end,
            symbol=symbol,
            symbol_kind="function",
        ),
    )


class ModuleRecordsTests(unittest.TestCase):
    # A file whose function body (lines 4-6) is captured by a symbol_definition,
    # leaving imports (1-2) and a module-level registry wiring (line 9) uncovered.
    _TEXT = (
        "import os\n"
        "from pkg import LanguageProfile\n"
        "\n"
        "def my_builder(path):\n"
        "    return os.path.basename(path)\n"
        "\n"
        "\n"
        "PROFILE = LanguageProfile(\n"
        "    file_metadata=my_builder,\n"
        ")\n"
    )

    @staticmethod
    def _definitions(intel_file: IntelFile) -> list[IntelRecord]:
        # my_builder spans lines 4-5 (def + body); everything else is module level.
        return [_definition_record(intel_file, "my_builder", 4, 5)]

    def test_module_records_capture_reference_made_at_module_level(self) -> None:
        intel_file = _make_intel_file()
        records = self._definitions(intel_file)
        module_recs, _edges = module_records(intel_file, self._TEXT, records, max_chars=200)
        self.assertTrue(module_recs)
        self.assertTrue(all(rec.record_type == "module_chunk" for rec in module_recs))
        joined = "\n".join(rec.display_content for rec in module_recs)
        # The by-reference wiring (no call parens) is now visible as record text,
        # which is exactly what call-candidate edges alone cannot capture.
        self.assertIn("my_builder", joined)
        self.assertIn("PROFILE", joined)
        # The function's own body (line 5, os.path.basename) stays out of the
        # module chunk because the definition record already covers it.
        self.assertNotIn("basename", joined)

    def test_module_records_emit_call_candidate_edges_for_module_level_calls(self) -> None:
        intel_file = _make_intel_file()
        records = self._definitions(intel_file)
        module_recs, edges = module_records(intel_file, self._TEXT, records, max_chars=200)
        targets = {edge.target_symbol for edge in edges}
        # `LanguageProfile(` is a module-level call → a call_candidate edge.
        self.assertIn("LanguageProfile", targets)
        self.assertTrue(all(edge.edge_type == "call_candidate" for edge in edges))
        module_ids = {rec.record_id for rec in module_recs}
        self.assertTrue(all(edge.source_record_id in module_ids for edge in edges))

    def test_module_records_empty_without_symbol_definitions(self) -> None:
        # No definition records (e.g. a line-window-fallback file) → nothing to
        # add; the fallback chunks already cover the whole file.
        intel_file = _make_intel_file()
        self.assertEqual(module_records(intel_file, self._TEXT, [], max_chars=200), ([], []))

    def test_module_records_empty_when_definitions_cover_all_content(self) -> None:
        intel_file = _make_intel_file()
        text = "def only():\n    return 1\n"
        records = [_definition_record(intel_file, "only", 1, 2)]
        self.assertEqual(module_records(intel_file, text, records, max_chars=200), ([], []))

    def test_module_records_carry_line_ranges_and_ordinals(self) -> None:
        intel_file = _make_intel_file()
        records = self._definitions(intel_file)
        module_recs, _edges = module_records(intel_file, self._TEXT, records, max_chars=200)
        for index, rec in enumerate(module_recs, 1):
            self.assertEqual(rec.metadata.get("chunk_ordinal"), index)
            self.assertTrue(rec.metadata.get("module_level"))
            self.assertIsNotNone(rec.line_start)
            self.assertIsNotNone(rec.line_end)
        self.assertTrue(all(rec.confidence_kind == "high_confidence_fact" for rec in module_recs))


class MakeCodeRecordTests(unittest.TestCase):
    def test_make_code_record_uses_ordinal_in_record_id(self) -> None:
        intel_file = _make_intel_file()
        lines = [(1, "alpha()"), (2, "beta()"), (3, "gamma()")]
        record = make_code_record(intel_file, lines, ordinal=2, reason="fallback line window")
        self.assertEqual(record.line_start, 1)
        self.assertEqual(record.line_end, 3)
        # record_id encodes the line range.
        self.assertIn("::chunk::000001-000003", record.record_id)
        # Approximate confidence for fallback reason.
        self.assertEqual(record.confidence_kind, "approximate_fact")
        # Symbols referenced are extracted into metadata.
        symbols = cast("list[JsonValue]", record.metadata.get("symbols_referenced") or [])
        self.assertIn("alpha", symbols)
        self.assertIn("beta", symbols)
        self.assertIn("gamma", symbols)

    def test_make_code_record_non_fallback_reason_is_high_confidence(self) -> None:
        intel_file = _make_intel_file()
        record = make_code_record(intel_file, [(1, "a()")], ordinal=1, reason="parser block")
        self.assertEqual(record.confidence_kind, "high_confidence_fact")


if __name__ == "__main__":
    _ = unittest.main()
