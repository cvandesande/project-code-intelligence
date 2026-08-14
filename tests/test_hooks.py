from __future__ import annotations

import io
import json
import os
import sys
import unittest
from collections.abc import Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import mock

from project_code_intelligence import analyze
from project_code_intelligence.exceptions import DatabaseConnectionError
from project_code_intelligence.hooks import cli as hooks_cli
from project_code_intelligence.hooks import detect, install, runtime, similar
from project_code_intelligence.hooks.opencode_assets import OPENCODE_FILES


class DetectTests(unittest.TestCase):
    def test_removed_definition_on_delete(self) -> None:
        self.assertEqual(detect.removed_definitions("def render_text(x):\n    return 1\n", ""), ["render_text"])

    def test_in_place_edit_removes_nothing(self) -> None:
        old = "def render_text(x):\n    return 1\n"
        new = "def render_text(x):\n    return 2\n"
        self.assertEqual(detect.removed_definitions(old, new), [])

    def test_rename_reports_old_name(self) -> None:
        self.assertEqual(detect.removed_definitions("def foo():\n pass\n", "def bar():\n pass\n"), ["foo"])

    def test_detects_class_func_type_and_shell(self) -> None:
        blob = "class A:\n    pass\nfunc Go() {}\ntype T struct{}\nsh_fn() {\n  :\n}\n"
        self.assertEqual(detect.defined_names(blob), {"A", "Go", "T", "sh_fn"})

    def test_source_path_gate(self) -> None:
        self.assertTrue(detect.is_source_path("a/b.py"))
        self.assertFalse(detect.is_source_path("notes.md"))

    def test_detects_rust_fn_through_every_modifier_order(self) -> None:
        """.rs was in SOURCE_EXT while `fn` matched nothing, so Rust definitions were
        invisible to both sides of the hook. Methods indent inside impl blocks."""
        blob = (
            "fn plain() {}\n"
            "pub fn public() {}\n"
            "pub(crate) fn scoped() {}\n"
            "pub async fn public_async() {}\n"
            "const fn constant() {}\n"
            'pub unsafe extern "C" fn foreign() {}\n'
            "impl Thing {\n"
            "    pub fn method(&self) {}\n"
            "}\n"
        )
        self.assertEqual(
            detect.defined_names(blob),
            {"plain", "public", "scoped", "public_async", "constant", "foreign", "method"},
        )

    def test_rust_removal_is_a_removal(self) -> None:
        self.assertEqual(detect.removed_definitions("pub fn gone(a: u32) -> u32 {\n    a\n}\n", ""), ["gone"])

    def test_gate_patterns_match_the_opencode_copy_verbatim(self) -> None:
        """detect.py and LIB_JS carry the same patterns and nothing enforced it until now --
        the opencode plugin uses its JS copy as the gate, so drift silently drops events."""
        lib_js = OPENCODE_FILES["lib/pci-evidence-logic.js"]
        for source in detect.DEF_SOURCES:
            self.assertIn(source, lib_js, f"pattern missing from LIB_JS: {source}")
        self.assertEqual(lib_js.count("/g,\n"), len(detect.DEF_SOURCES))

    def test_checked_in_opencode_assets_match_their_source(self) -> None:
        """.opencode/ holds this repo's own installed copies and they are tracked, so editing
        OPENCODE_FILES without rewriting them leaves a stale gate running here."""
        root = Path(__file__).resolve().parent.parent
        for rel, text in OPENCODE_FILES.items():
            installed = root / ".opencode" / rel
            if not installed.is_file():
                continue  # not installed in this checkout; install_opencode covers that path
            self.assertEqual(installed.read_text(encoding="utf-8"), text, f"stale copy: {rel}")

    def test_definition_slice_stops_before_the_next_definition(self) -> None:
        """The slice is a query vector, so a neighbouring signature must not bleed into it."""
        text = "def wanted(x):\n    y = x\n    return y\n\ndef other():\n    pass\n"
        self.assertEqual(
            detect.definition_slices(text, ["wanted"]), {"wanted": "def wanted(x):\n    y = x\n    return y"}
        )

    def test_definition_slice_handles_braces_and_misses(self) -> None:
        self.assertEqual(
            detect.definition_slices("sh_fn() {\n  echo hi\n}\n", ["sh_fn"]), {"sh_fn": "sh_fn() {\n  echo hi"}
        )
        self.assertEqual(detect.definition_slices("def a():\n pass\n", ["absent"]), {})

    def test_definition_slice_caps_runaway_bodies(self) -> None:
        text = "def big():\n" + "".join(f"    line{i} = {i}\n" for i in range(200))
        self.assertEqual(len(detect.definition_slices(text, ["big"], max_lines=10)["big"].splitlines()), 10)


class _StubCursor:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[dict[str, object]]:
        return self._rows


class _StubConn:
    """Enough of a connection for similar.nearest: one canned result set per query."""

    def __init__(self, batches: list[list[dict[str, object]]]) -> None:
        self.batches = batches
        self.queries = 0

    def execute(self, _sql: str, _params: object = None) -> _StubCursor:
        rows = self.batches[min(self.queries, len(self.batches) - 1)]
        self.queries += 1
        return _StubCursor(rows)

    def __enter__(self) -> _StubConn:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _one_snapshot(_conn: object) -> list[analyze.SnapshotRef]:
    return [analyze.SnapshotRef(snapshot_id=1, collection="c", repo="r")]


def _no_snapshots(_conn: object) -> list[analyze.SnapshotRef]:
    return []


def _fake_embedding(_text: str) -> tuple[str, int]:
    return "[0]", 1


def _one_hit(hit: similar.Hit) -> _Nearest:
    """Stub for similar.nearest returning a single canned hit."""

    def nearest(
        _slices: Mapping[str, str], *, language: str | None = None, file_path: str | None = None
    ) -> list[similar.Hit]:
        _ = language, file_path
        return [hit]

    return nearest


def _row(symbol: str, distance: float, *, line: object = 10, path: str = "repo/src/pkg/mod.py") -> dict[str, object]:
    return {"symbol": symbol, "source_path": path, "line_start": line, "distance": distance}


class SimilarTests(unittest.TestCase):
    """The query path in similar.nearest, with the index and the embedder stubbed out."""

    @staticmethod
    def _nearest(batches: list[list[dict[str, object]]]) -> list[similar.Hit]:
        conn = _StubConn(batches)
        with (
            mock.patch.object(similar.mcp_db, "connect", lambda: conn),
            mock.patch.object(similar.analyze, "latest_snapshots", _one_snapshot),
            mock.patch.object(similar.semantic, "query_embedding", _fake_embedding),
        ):
            return similar.nearest({"brand_new": "def brand_new():\n    pass"})

    def test_hits_beyond_the_gate_are_dropped(self) -> None:
        hits = SimilarTests._nearest([[_row("close_one", 0.20), _row("far_one", similar.GATE + 0.05)]])
        self.assertEqual([hit.symbol for hit in hits], ["close_one"])

    def test_self_match_is_not_prior_art(self) -> None:
        """An in-place rewrite matches its own indexed chunk; 'similar to itself' is noise."""
        hits = SimilarTests._nearest([[_row("Klass.brand_new", 0.01), _row("other", 0.10)]])
        self.assertEqual([hit.symbol for hit in hits], ["other"])

    def test_rust_path_self_match_is_dropped_and_near_miss_names_are_kept(self) -> None:
        """The split must handle :: paths; a longer name (verify_server_cert vs
        verify_server_certificate) is NOT a self-match and must survive."""
        rows = [_row("crate::mod::Trait::brand_new", 0.01), _row("Trait::brand_new_thing", 0.10)]
        hits = SimilarTests._nearest([rows])
        self.assertEqual([hit.symbol for hit in hits], ["Trait::brand_new_thing"])

    def test_trait_impl_methods_are_not_filtered_from_prior_art(self) -> None:
        """A near-verbatim copy of a rustls trait impl (distance 0.098) was invisible while
        the add-side SQL carried the audit's impl_trait exclusion. Names-audit noise
        reasoning does not transfer to a distance query; the gate handles siblings."""
        source = Path(similar.__file__).read_text(encoding="utf-8")
        query = source.split('_SQL = """', 1)[1].split('"""', 1)[0]
        self.assertNotIn("impl_trait", query)

    def test_split_chunks_of_one_function_render_once(self) -> None:
        """Long functions index as a whole-body chunk plus overlapping split chunks under
        the same symbol; one match must not render as several rows."""
        hits = SimilarTests._nearest([[_row("long_fn", 0.10), _row("long_fn", 0.12), _row("other", 0.15)]])
        self.assertEqual([(h.symbol, h.distance) for h in hits], [("long_fn", 0.10), ("other", 0.15)])

    def test_rows_with_unusable_columns_are_skipped_not_crashed(self) -> None:
        hits = SimilarTests._nearest([
            [_row("ok", 0.10, line=None), {"symbol": None, "source_path": None, "line_start": 1, "distance": 0.1}]
        ])
        self.assertEqual([(hit.symbol, hit.line_start) for hit in hits], [("ok", None)])

    def test_results_are_sorted_and_capped(self) -> None:
        rows = [_row(f"s{i}", 0.20 - i * 0.01) for i in range(5)]
        hits = SimilarTests._nearest([rows])
        self.assertEqual(len(hits), similar.MAX_HITS)
        self.assertEqual([hit.distance for hit in hits], sorted(hit.distance for hit in hits))

    def test_missing_snapshot_returns_nothing_rather_than_raising(self) -> None:
        conn = _StubConn([[]])
        with (
            mock.patch.object(similar.mcp_db, "connect", lambda: conn),
            mock.patch.object(similar.analyze, "latest_snapshots", _no_snapshots),
        ):
            self.assertEqual(similar.nearest({"a": "def a():\n    pass"}), [])

    def test_rust_gets_a_higher_gate_than_python(self) -> None:
        """One constant does not transfer: at Python's 0.25 the hook was silent on a Rust pair
        a blind validation had labelled duplicates (true hit at 0.26)."""
        self.assertGreater(similar.gate("rust"), similar.gate("python"))
        self.assertAlmostEqual(similar.gate("python"), similar.GATE)
        self.assertAlmostEqual(similar.gate("go"), similar.GATE)  # unmeasured -> base
        self.assertAlmostEqual(similar.gate(None), similar.GATE)

    def test_env_override_beats_the_language_default(self) -> None:
        """A repo that has calibrated its own value must win over the shipped table."""
        with mock.patch.dict(os.environ, {similar.GATE_ENV: "0.55"}):
            self.assertAlmostEqual(similar.gate("rust"), 0.55)

    def test_gate_env_override_is_read_and_bad_values_ignored(self) -> None:
        """The gate does not transfer across languages, so a repo must be able to retune it."""
        with mock.patch.dict(os.environ, {similar.GATE_ENV: "0.40"}):
            self.assertAlmostEqual(similar.gate(), 0.40)
        with mock.patch.dict(os.environ, {similar.GATE_ENV: "not-a-number"}):
            self.assertAlmostEqual(similar.gate(), similar.GATE)

    def test_snapshot_selection_follows_the_files_git_root(self) -> None:
        """A file under a nested clone must query THAT repo's snapshot, and must raise when
        no snapshot covers it -- a no-hit against the wrong repo's index carries near-zero
        information and must not render as a calibrated no-hit."""
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            (workspace / ".git").mkdir(parents=True)
            nested = workspace / "nested"
            (nested / ".git").mkdir(parents=True)
            snapshots = [
                analyze.SnapshotRef(snapshot_id=1, collection="c", repo="ws"),
                analyze.SnapshotRef(snapshot_id=2, collection="c", repo="nested"),
            ]
            cwd = Path.cwd()
            os.chdir(workspace)
            try:
                self.assertEqual(similar.snapshot_for(snapshots, "a.py").snapshot_id, 1)
                self.assertEqual(similar.snapshot_for(snapshots, str(nested / "src" / "b.rs")).snapshot_id, 2)
                with self.assertRaises(similar.UnindexedRepoError):
                    _ = similar.snapshot_for(snapshots[:1], str(nested / "src" / "b.rs"))
            finally:
                os.chdir(cwd)

    def test_snapshot_selection_resolves_a_worktree_file_to_the_main_repo(self) -> None:
        """A file edited inside a linked worktree must match the MAIN repo's snapshot
        (never a new repo keyed on the worktree's own directory name), while still
        preferring the snapshot on the worktree's OWN checkout branch."""
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            (workspace / ".git").mkdir(parents=True)
            worktree = Path(tmp) / "wt"
            worktree.mkdir()
            _ = (worktree / ".git").write_text(f"gitdir: {workspace}/.git/worktrees/wt\n", encoding="utf-8")
            snapshots = [
                analyze.SnapshotRef(snapshot_id=1, collection="c", repo="ws", branch="main"),
                analyze.SnapshotRef(snapshot_id=2, collection="c", repo="ws", branch="feature"),
            ]
            cwd = Path.cwd()
            os.chdir(workspace)
            try:
                with mock.patch.object(similar.analyze, "resolve_repo_branch", return_value="feature") as branch_mock:
                    selected = similar.snapshot_for(snapshots, str(worktree / "src" / "b.py"))
                branch_mock.assert_called_once_with(worktree.resolve())
                self.assertEqual(selected.snapshot_id, 2)
            finally:
                os.chdir(cwd)

    def test_render_strips_the_repo_prefix(self) -> None:
        hit = similar.Hit(added_name="new", symbol="old", source_path="repo/src/a.py", line_start=7, distance=0.123)
        self.assertEqual(hit.render("repo"), "  0.12  old  src/a.py:7  (vs your new)")


def _write_event(path: Path, content: str) -> dict[str, object]:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": "Write",
        "tool_input": {"file_path": str(path), "content": content},
    }


_Nearest = Callable[..., "list[similar.Hit]"]


def _no_hits(
    _slices: Mapping[str, str], *, language: str | None = None, file_path: str | None = None
) -> list[similar.Hit]:
    """Stub for similar.nearest: the query ran and nothing cleared the gate."""
    _ = language, file_path
    return []


class _StubReports:
    """Stand-in for evidence.render_symbol_reports that avoids the database."""

    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self.mapping = mapping
        self.calls: list[str] = []

    def __call__(self, symbol: str, **_: object) -> list[str]:
        self.calls.append(symbol)
        return self.mapping.get(symbol, [])


class EvidenceRuntimeTests(unittest.TestCase):  # noqa: PLR0904 - one shared event-runtime fixture
    def _run(self, agent: str, event: Mapping[str, object], reports: _StubReports) -> str:
        out = io.StringIO()
        with mock.patch.object(runtime.evidence, "render_symbol_reports", reports):
            code = runtime.run_evidence(agent, stdin=io.StringIO(json.dumps(event)), stdout=out)
        self.assertEqual(code, 0)
        return out.getvalue()

    def test_opencode_delete_injects_raw_block(self) -> None:
        reports = _StubReports({"gone": ["gone  function  a.py:1-3\ncallers (0)"]})
        event = {"filePath": "a.py", "oldString": "def gone():\n pass\n", "newString": ""}
        text = self._run("opencode", event, reports)
        self.assertIn("[pci blast-radius", text)
        self.assertIn("gone  function", text)
        self.assertEqual(reports.calls, ["gone"])

    def test_claude_delete_wraps_in_additional_context(self) -> None:
        reports = _StubReports({"gone": ["gone  function  a.py:1-3\ncallers (0)"]})
        event = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Edit",
            "tool_input": {"file_path": "a.py", "old_string": "def gone():\n x\n", "new_string": ""},
        }
        payload = cast("dict[str, object]", json.loads(self._run("claude", event, reports)))
        hook_out = cast("dict[str, object]", payload["hookSpecificOutput"])
        # Output event name echoes the firing event (preventive PreToolUse).
        self.assertEqual(hook_out["hookEventName"], "PreToolUse")
        self.assertIn("[pci blast-radius", cast("str", hook_out["additionalContext"]))

    def test_codex_apply_patch_delete_wraps_in_additional_context(self) -> None:
        reports = _StubReports({"gone": ["gone  function  a.py:1-3\ncallers (0)"]})
        event = {
            "hook_event_name": "PreToolUse",
            "tool_name": "apply_patch",
            "tool_input": {
                "command": "*** Begin Patch\n*** Update File: a.py\n@@\n-def gone():\n-    pass\n*** End Patch"
            },
        }
        payload = cast("dict[str, object]", json.loads(self._run("codex", event, reports)))
        hook_out = cast("dict[str, object]", payload["hookSpecificOutput"])
        self.assertEqual(hook_out["hookEventName"], "PreToolUse")
        self.assertIn("[pci blast-radius", cast("str", hook_out["additionalContext"]))
        self.assertEqual(reports.calls, ["gone"])

    def test_codex_multi_file_patch_checks_each_file(self) -> None:
        reports = _StubReports({
            "one": ["one  function  a.py:1-2\ncallers (0)"],
            "two": ["two  function  b.py:1-2\ncallers (0)"],
        })
        event = {
            "hook_event_name": "PreToolUse",
            "tool_input": {
                "command": (
                    "*** Begin Patch\n*** Update File: a.py\n@@\n-def one():\n-    pass\n"
                    "*** Update File: b.py\n@@\n-def two():\n-    pass\n*** End Patch"
                )
            },
        }
        text = self._run("codex", event, reports)
        self.assertIn("one  function", text)
        self.assertIn("two  function", text)
        self.assertEqual(reports.calls, ["one", "two"])

    def test_codex_delete_file_reads_existing_content(self) -> None:
        reports = _StubReports({"gone": ["gone  function  a.py:1-2\ncallers (0)"]})
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.py"
            _ = target.write_text("def gone():\n    pass\n", encoding="utf-8")
            event = {
                "hook_event_name": "PreToolUse",
                "tool_input": {"command": f"*** Begin Patch\n*** Delete File: {target}\n*** End Patch"},
            }
            text = self._run("codex", event, reports)
        self.assertIn("[pci blast-radius", text)
        self.assertEqual(reports.calls, ["gone"])

    def test_in_place_edit_is_silent(self) -> None:
        reports = _StubReports({"gone": ["should not appear"]})
        event = {"filePath": "a.py", "oldString": "def keep():\n return 1\n", "newString": "def keep():\n return 2\n"}
        self.assertEqual(self._run("opencode", event, reports), "")
        self.assertEqual(reports.calls, [])

    def _run_add(self, event: Mapping[str, object], nearest: _Nearest) -> str:
        """Add-branch run with the index query stubbed -- these tests must never depend on a
        reachable database or embedding endpoint."""
        with mock.patch.object(runtime.similar, "nearest", nearest):
            return self._run("opencode", event, _StubReports({}))

    def test_no_close_hit_is_silent(self) -> None:
        """Nothing actionable means no interruption; the banner carries the standing rule."""
        event = {"filePath": "a.py", "oldString": "", "newString": "def brand_new():\n x = 1\n return x\n"}
        self.assertEqual(self._run_add(event, _no_hits), "")

    def test_add_side_carries_practice_text_and_env_turns_it_off(self) -> None:
        hit = runtime.similar.Hit(
            added_name="brand_new",
            symbol="existing_helper",
            source_path="repo/src/pkg/mod.py",
            line_start=42,
            distance=0.18,
        )
        event = {"filePath": "a.py", "oldString": "", "newString": "def brand_new():\n x = 1\n return x\n"}
        with mock.patch.dict(os.environ, {}, clear=False):
            _ = os.environ.pop("PCI_HOOK_PRACTICE", None)
            self.assertIn("[pci practice", self._run_add(event, _one_hit(hit)))
        with mock.patch.dict(os.environ, {"PCI_HOOK_PRACTICE": "0"}):
            self.assertNotIn("[pci practice", self._run_add(event, _one_hit(hit)))

    def test_hook_disable_env_silences_everything(self) -> None:
        event = {"filePath": "a.py", "oldString": "", "newString": "def brand_new():\n x = 1\n return x\n"}
        with mock.patch.dict(os.environ, {"PCI_HOOK_DISABLE": "1"}):
            self.assertEqual(self._run_add(event, _no_hits), "")

    def test_added_definition_shows_prior_art_when_the_index_has_a_close_hit(self) -> None:
        hit = runtime.similar.Hit(
            added_name="brand_new",
            symbol="existing_helper",
            source_path="repo/src/pkg/mod.py",
            line_start=42,
            distance=0.18,
        )
        event = {"filePath": "a.py", "oldString": "", "newString": "def brand_new():\n x = 1\n return x\n"}
        text = self._run_add(event, _one_hit(hit))
        self.assertIn("existing_helper", text)
        self.assertIn("0.18", text)
        self.assertIn("evidence, not a finding", text)

    def test_added_definition_query_failure_warns_and_never_stays_silent(self) -> None:
        """A skipped check must not read as 'nothing found' -- that is the dangerous direction."""

        def explode(
            _slices: Mapping[str, str], *, language: str | None = None, file_path: str | None = None
        ) -> list[similar.Hit]:
            _ = language, file_path
            raise DatabaseConnectionError("could not connect to pci_x")

        event = {"filePath": "a.py", "oldString": "", "newString": "def brand_new():\n x = 1\n return x\n"}
        text = self._run_add(event, explode)
        self.assertIn("could not run", text)
        self.assertIn("could not connect to pci_x", text)
        self.assertIn("brand_new", text)

    def test_file_outside_indexed_repos_gets_the_distinct_message(self) -> None:
        """Not the calibrated no-hit: the index holds none of this repo's code."""

        def unindexed(
            _slices: Mapping[str, str], *, language: str | None = None, file_path: str | None = None
        ) -> list[similar.Hit]:
            _ = language, file_path
            raise similar.UnindexedRepoError("/ws/nested")

        event = {"filePath": "nested/a.py", "oldString": "", "newString": "def brand_new():\n x = 1\n return x\n"}
        text = self._run_add(event, unindexed)
        self.assertIn("outside the indexed repos", text)
        self.assertIn("/ws/nested", text)
        self.assertIn("brand_new", text)
        self.assertNotIn("similarity check ran", text)

    def test_added_definition_embeds_only_the_new_definition(self) -> None:
        """A Write payload is the whole file; the gate was calibrated on single definitions."""
        seen: list[dict[str, str]] = []
        languages: list[str | None] = []

        def capture(
            slices: Mapping[str, str], *, language: str | None = None, file_path: str | None = None
        ) -> list[similar.Hit]:
            _ = file_path
            seen.append(dict(slices))
            languages.append(language)
            return []

        payload = "import os\n\n\ndef untouched():\n return 1\n\n\ndef brand_new():\n x = 2\n return x\n"
        event = {"filePath": "a.py", "oldString": "def untouched():\n return 1\n", "newString": payload}
        _ = self._run_add(event, capture)
        self.assertEqual(list(seen[0]), ["brand_new"])
        self.assertNotIn("untouched", seen[0]["brand_new"])
        self.assertNotIn("import os", seen[0]["brand_new"])
        # The gate is language-dependent, so the edited file's language has to reach nearest.
        self.assertEqual(languages, ["python"])

    def test_added_test_case_gets_no_reminder(self) -> None:
        reports = _StubReports({})
        for path, name in (("tests/test_x.py", "test_thing"), ("pkg/helpers.py", "TestHarness")):
            event = {"filePath": path, "oldString": "", "newString": f"def {name}():\n pass\n"}
            self.assertEqual(self._run("opencode", event, reports), "", path)

    def test_removed_test_still_gets_blast_radius(self) -> None:
        """The exemption is add-only: deleting a test is a coverage loss worth reporting."""
        reports = _StubReports({"test_thing": ["test_thing  function  tests/test_x.py:1-2\ncallers (0)"]})
        event = {"filePath": "tests/test_x.py", "oldString": "def test_thing():\n x\n", "newString": ""}
        self.assertIn("[pci blast-radius", self._run("opencode", event, reports))

    def test_rename_prefers_blast_radius_over_reminder(self) -> None:
        reports = _StubReports({"old_name": ["old_name  function  a.py:1-3\ncallers (1)"]})
        event = {"filePath": "a.py", "oldString": "def old_name():\n x\n", "newString": "def new_name():\n x\n"}
        text = self._run("opencode", event, reports)
        self.assertIn("[pci blast-radius", text)
        self.assertNotIn("[pci add-side", text)

    def test_non_source_file_is_silent(self) -> None:
        reports = _StubReports({"gone": ["x"]})
        event = {"filePath": "notes.md", "oldString": "def gone(): pass", "newString": ""}
        self.assertEqual(self._run("opencode", event, reports), "")

    def test_empty_reports_stay_silent(self) -> None:
        reports = _StubReports({})  # symbol resolves to nothing in the index
        event = {"filePath": "a.py", "oldString": "def gone():\n x\n", "newString": ""}
        self.assertEqual(self._run("opencode", event, reports), "")

    def test_unreachable_index_warns_instead_of_silence(self) -> None:
        """A removal with an unreachable database must warn, not read as 'no callers'."""

        def raising(_symbol: str, **_: object) -> list[str]:
            raise runtime.DatabaseConnectionError("Could not connect to PostgreSQL/pgvector using PCI_PG_DB=x")

        out = io.StringIO()
        event = {"filePath": "a.py", "oldString": "def gone():\n x\n", "newString": ""}
        with mock.patch.object(runtime.evidence, "render_symbol_reports", raising):
            code = runtime.run_evidence("opencode", stdin=io.StringIO(json.dumps(event)), stdout=out)
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("blast-radius unavailable", text)
        self.assertIn("gone", text)
        self.assertIn("Could not connect", text)

    def test_claude_write_over_file_reports_removals(self) -> None:
        """A whole-file Write carries no old_string; the old side comes from disk."""
        reports = _StubReports({"gone": ["gone  function  a.py:1-3\ncallers (2)"]})
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.py"
            _ = target.write_text("def gone():\n    return 1\n", encoding="utf-8")
            event = _write_event(target, "# rewritten\n")
            payload = cast("dict[str, object]", json.loads(self._run("claude", event, reports)))
        hook_out = cast("dict[str, object]", payload["hookSpecificOutput"])
        self.assertIn("[pci blast-radius", cast("str", hook_out["additionalContext"]))
        self.assertEqual(reports.calls, ["gone"])

    def test_claude_write_keeping_definition_is_silent(self) -> None:
        reports = _StubReports({"gone": ["should not appear"]})
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.py"
            _ = target.write_text("def gone():\n    return 1\n", encoding="utf-8")
            event = _write_event(target, "def gone():\n    return 2\n")
            self.assertEqual(self._run("claude", event, reports), "")
        self.assertEqual(reports.calls, [])

    def test_claude_write_new_file_takes_the_add_branch(self) -> None:
        """A new file removes nothing, so it takes the add branch, never blast radius.

        similar.nearest is stubbed: the add branch does query the index now, and a unit test
        must not depend on a reachable database or embedding endpoint to decide the branch.
        """
        reports = _StubReports({"gone": ["should not appear"]})
        hit = runtime.similar.Hit(
            added_name="fresh", symbol="existing", source_path="repo/a.py", line_start=1, distance=0.2
        )
        with TemporaryDirectory() as tmp, mock.patch.object(runtime.similar, "nearest", _one_hit(hit)):
            event = _write_event(Path(tmp) / "new.py", "def fresh():\n    return 1\n")
            text = self._run("claude", event, reports)
        self.assertIn("[pci add-side", text)
        self.assertNotIn("[pci blast-radius", text)
        self.assertEqual(reports.calls, [])

    def test_rename_fires_blast_radius_for_the_old_name(self) -> None:
        """A rename removes the old name; the evidence hook must report it."""
        reports = _StubReports({"old_name": ["old_name  function  a.py:1-3\ncallers (1)"]})
        event = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Edit",
            "tool_input": {
                "file_path": "a.py",
                "old_string": "def old_name():\n    return build()\n",
                "new_string": "def new_name():\n    return build()\n",
            },
        }
        text = self._run("claude", event, reports)
        self.assertIn("[pci blast-radius", text)
        self.assertEqual(reports.calls, ["old_name"])


class InstallOpencodeTests(unittest.TestCase):
    def test_install_then_uninstall(self) -> None:
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            outcome = install.install_opencode(project, uninstall=False, dry_run=False)
            self.assertEqual(outcome.action, "installed")
            for rel in OPENCODE_FILES:
                self.assertTrue((project / ".opencode" / rel).is_file())

            again = install.install_opencode(project, uninstall=False, dry_run=False)
            self.assertEqual(again.action, "updated")

            removed = install.install_opencode(project, uninstall=True, dry_run=False)
            self.assertEqual(removed.action, "removed")
            for rel in OPENCODE_FILES:
                self.assertFalse((project / ".opencode" / rel).exists())

    def test_dry_run_writes_nothing(self) -> None:
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            outcome = install.install_opencode(project, uninstall=False, dry_run=True)
            self.assertEqual(outcome.action, "installed")
            self.assertFalse((project / ".opencode").exists())


def _read_json_file(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


class InstallClaudeTests(unittest.TestCase):
    def test_install_merges_and_is_idempotent(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = Path(tmp) / ".claude" / "settings.json"
            first = install.install_claude(settings, uninstall=False, dry_run=False)
            self.assertEqual(first.action, "installed")
            second = install.install_claude(settings, uninstall=False, dry_run=False)
            self.assertEqual(second.action, "updated")

            data = _read_json_file(settings)
            hooks = cast("dict[str, object]", data["hooks"])
            self.assertEqual(len(cast("list[object]", hooks["PreToolUse"])), 1)
            self.assertEqual(len(cast("list[object]", hooks["SessionStart"])), 1)
            self.assertNotIn("PostToolUse", hooks)  # evidence is preventive now
            self.assertNotIn("Stop", hooks)  # reindex moved to the git post-commit hook

    def test_uninstall_removes_the_banner_group(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = Path(tmp) / ".claude" / "settings.json"
            _ = install.install_claude(settings, uninstall=False, dry_run=False)
            _ = install.install_claude(settings, uninstall=True, dry_run=False)
            self.assertNotIn("hooks", _read_json_file(settings))

    def test_banner_runtime_wraps_session_start_context(self) -> None:
        out = io.StringIO()
        self.assertEqual(runtime.run_banner("claude", stdout=out), 0)
        payload = cast("dict[str, object]", json.loads(out.getvalue()))
        hook_out = cast("dict[str, object]", payload["hookSpecificOutput"])
        self.assertEqual(hook_out["hookEventName"], "SessionStart")
        self.assertIn("PCI RADAR MODE ACTIVE", cast("str", hook_out["additionalContext"]))

    def test_install_preserves_foreign_hooks(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = Path(tmp) / ".claude" / "settings.json"
            settings.parent.mkdir(parents=True)
            foreign = {
                "hooks": {"PostToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "lint"}]}]}
            }
            _ = settings.write_text(json.dumps(foreign), encoding="utf-8")

            _ = install.install_claude(settings, uninstall=False, dry_run=False)
            groups = cast("list[object]", cast("dict[str, object]", _read_json_file(settings)["hooks"])["PostToolUse"])
            commands = {
                cast("str", cast("dict[str, object]", handler).get("command"))
                for group in groups
                for handler in cast("list[object]", cast("dict[str, object]", group).get("hooks"))
            }
            self.assertIn("lint", commands)

            removed = install.install_claude(settings, uninstall=True, dry_run=False)
            self.assertEqual(removed.action, "removed")
            # Foreign hook survives uninstall; our handlers are gone.
            hooks_after = cast("dict[str, object]", _read_json_file(settings)["hooks"])
            groups_after = cast("list[object]", hooks_after["PostToolUse"])
            self.assertEqual(len(groups_after), 1)

    def test_legacy_agent_config_is_recognized_and_migrated(self) -> None:
        """Pre-consolidation installs wrote pci-hook + --agent; a reinstall must
        treat them as ours (update, not duplicate) and rewrite to the new spelling."""
        with TemporaryDirectory() as tmp:
            settings = Path(tmp) / ".claude" / "settings.json"
            settings.parent.mkdir(parents=True)
            legacy = {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Edit|Write",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "/old/.venv/bin/pci-hook",
                                    "args": ["run", "--agent", "claude", "--behavior", "evidence"],
                                }
                            ],
                        }
                    ]
                }
            }
            _ = settings.write_text(json.dumps(legacy), encoding="utf-8")
            outcome = install.install_claude(settings, uninstall=False, dry_run=False)
            self.assertEqual(outcome.action, "updated")
            groups = cast("list[object]", cast("dict[str, object]", _read_json_file(settings)["hooks"])["PreToolUse"])
            self.assertEqual(len(groups), 1)
            handler = cast("dict[str, object]", cast("list[object]", cast("dict[str, object]", groups[0])["hooks"])[0])
            # Claude Code ignores an "args" key: everything must be in the command string.
            self.assertNotIn("args", handler)
            self.assertIn("--target claude", cast("str", handler["command"]))

    def test_install_writes_single_command_string(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = Path(tmp) / ".claude" / "settings.json"
            _ = install.install_claude(settings, uninstall=False, dry_run=False)
            groups = cast("list[object]", cast("dict[str, object]", _read_json_file(settings)["hooks"])["PreToolUse"])
            handler = cast("dict[str, object]", cast("list[object]", cast("dict[str, object]", groups[0])["hooks"])[0])
            self.assertNotIn("args", handler)
            command = cast("str", handler["command"])
            self.assertIn("run --target claude --behavior evidence", command)
            # A reinstall recognizes the command-string spelling as ours.
            second = install.install_claude(settings, uninstall=False, dry_run=False)
            self.assertEqual(second.action, "updated")
            groups2 = cast("list[object]", cast("dict[str, object]", _read_json_file(settings)["hooks"])["PreToolUse"])
            self.assertEqual(len(groups2), 1)

    def test_uninstall_on_clean_config_is_unchanged(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = Path(tmp) / ".claude" / "settings.json"
            outcome = install.install_claude(settings, uninstall=True, dry_run=False)
            self.assertEqual(outcome.action, "unchanged")


def _prompt_scope_in(cwd: Path, reply: str) -> bool:
    with (
        mock.patch.object(Path, "cwd", return_value=cwd),
        mock.patch.object(hooks_cli.sys, "stdin", io.StringIO(reply)),
        mock.patch.object(hooks_cli.sys, "stderr", io.StringIO()),
    ):
        return hooks_cli.prompt_claude_scope()


class InstallCodexTests(unittest.TestCase):
    def test_install_merges_is_idempotent_and_uninstalls(self) -> None:
        with TemporaryDirectory() as tmp:
            hooks_path = Path(tmp) / ".codex" / "hooks.json"
            hooks_path.parent.mkdir(parents=True)
            foreign = {"hooks": {"PostToolUse": [{"hooks": [{"type": "command", "command": "lint"}]}]}}
            _ = hooks_path.write_text(json.dumps(foreign), encoding="utf-8")

            first = install.install_codex(hooks_path, uninstall=False, dry_run=False)
            self.assertEqual(first.action, "installed")
            second = install.install_codex(hooks_path, uninstall=False, dry_run=False)
            self.assertEqual(second.action, "updated")
            hooks = cast("dict[str, object]", _read_json_file(hooks_path)["hooks"])
            self.assertEqual(len(cast("list[object]", hooks["PreToolUse"])), 1)
            self.assertEqual(len(cast("list[object]", hooks["SessionStart"])), 1)
            group = cast("dict[str, object]", cast("list[object]", hooks["PreToolUse"])[0])
            handler = cast("dict[str, object]", cast("list[object]", group["hooks"])[0])
            self.assertIn("run --target codex --behavior evidence", cast("str", handler["command"]))

            removed = install.install_codex(hooks_path, uninstall=True, dry_run=False)
            self.assertEqual(removed.action, "removed")
            hooks_after = cast("dict[str, object]", _read_json_file(hooks_path)["hooks"])
            self.assertNotIn("PreToolUse", hooks_after)
            self.assertNotIn("SessionStart", hooks_after)
            self.assertIn("PostToolUse", hooks_after)

    def test_dry_run_writes_nothing(self) -> None:
        with TemporaryDirectory() as tmp:
            hooks_path = Path(tmp) / ".codex" / "hooks.json"
            outcome = install.install_codex(hooks_path, uninstall=False, dry_run=True)
            self.assertEqual(outcome.action, "installed")
            self.assertFalse(hooks_path.exists())


class PromptClaudeScopeTests(unittest.TestCase):
    def test_non_project_cwd_defaults_to_user_scope(self) -> None:
        with TemporaryDirectory() as tmp:
            self.assertTrue(_prompt_scope_in(Path(tmp), ""))

    def test_project_reply_p_selects_project_scope(self) -> None:
        with TemporaryDirectory() as tmp:
            (Path(tmp) / ".git").mkdir()
            self.assertFalse(_prompt_scope_in(Path(tmp), "p\n"))

    def test_project_default_reply_selects_user_scope(self) -> None:
        with TemporaryDirectory() as tmp:
            (Path(tmp) / ".git").mkdir()
            self.assertTrue(_prompt_scope_in(Path(tmp), "\n"))


def _post_commit_path(repo: Path) -> Path:
    return repo / ".git" / "hooks" / "post-commit"


def _post_merge_path(repo: Path) -> Path:
    return repo / ".git" / "hooks" / "post-merge"


class InstallGitTests(unittest.TestCase):
    def test_install_writes_executable_managed_block(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            outcome = install.install_git(repo, uninstall=False, dry_run=False)
            self.assertEqual(outcome.action, "installed")
            for hook in (_post_commit_path(repo), _post_merge_path(repo)):
                text = hook.read_text(encoding="utf-8")
                self.assertIn("pci-hook reindex (managed)", text)
                self.assertIn("--behavior reindex", text)
                self.assertTrue(os.access(hook, os.X_OK))

    def test_install_is_idempotent(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _ = install.install_git(repo, uninstall=False, dry_run=False)
            again = install.install_git(repo, uninstall=False, dry_run=False)
            self.assertEqual(again.action, "updated")
            for hook in (_post_commit_path(repo), _post_merge_path(repo)):
                text = hook.read_text(encoding="utf-8")
                self.assertEqual(text.count(">>> pci-hook reindex (managed) >>>"), 1)

    def test_uninstall_keeps_user_script(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            hook = _post_commit_path(repo)
            hook.parent.mkdir(parents=True)
            _ = hook.write_text("#!/bin/sh\necho mine\n", encoding="utf-8")
            _ = install.install_git(repo, uninstall=False, dry_run=False)
            self.assertIn("echo mine", hook.read_text(encoding="utf-8"))
            removed = install.install_git(repo, uninstall=True, dry_run=False)
            self.assertEqual(removed.action, "removed")
            surviving = hook.read_text(encoding="utf-8")
            self.assertIn("echo mine", surviving)
            self.assertNotIn("managed", surviving)
            # post-merge had no pre-existing user script, so it should be gone entirely.
            self.assertFalse(_post_merge_path(repo).exists())

    def test_uninstall_deletes_when_only_ours(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _ = install.install_git(repo, uninstall=False, dry_run=False)
            self.assertTrue(_post_commit_path(repo).exists())
            self.assertTrue(_post_merge_path(repo).exists())
            _ = install.install_git(repo, uninstall=True, dry_run=False)
            self.assertFalse(_post_commit_path(repo).exists())
            self.assertFalse(_post_merge_path(repo).exists())

    def test_find_nested_repos_skips_root_gitfiles_and_hidden_dirs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / ".git").mkdir()
            nested = root / "vendor" / "nested"
            (nested / ".git").mkdir(parents=True)
            deeper = nested / "deeper"
            (deeper / ".git").mkdir(parents=True)
            submodule = root / "submodule"
            submodule.mkdir()
            _ = (submodule / ".git").write_text("gitdir: ../.git/modules/submodule\n", encoding="utf-8")
            hidden = root / ".cache" / "repo"
            (hidden / ".git").mkdir(parents=True)
            self.assertEqual(install.find_nested_repos(root), [nested, deeper])

    def test_has_git_hook_sees_only_our_managed_block(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self.assertFalse(install.has_git_hook(repo))
            hook = _post_commit_path(repo)
            hook.parent.mkdir(parents=True)
            _ = hook.write_text("#!/bin/sh\necho mine\n", encoding="utf-8")
            self.assertFalse(install.has_git_hook(repo))
            _ = install.install_git(repo, uninstall=False, dry_run=False)
            self.assertTrue(install.has_git_hook(repo))


class InstallGitNestedTests(unittest.TestCase):
    @staticmethod
    def _workspace(root: Path) -> Path:
        (root / ".git").mkdir()
        (root / "nested" / ".git").mkdir(parents=True)
        return root / "nested"

    def test_non_interactive_lists_nested_repos_without_touching_them(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            nested = self._workspace(root)
            parsed = hooks_cli.HookNamespace(project=str(root))
            outcomes = hooks_cli._install_git_with_nested(parsed)  # pyright: ignore[reportPrivateUsage]
            self.assertEqual(len(outcomes), 1)
            self.assertFalse(_post_commit_path(nested).exists())
            self.assertIn(("nested repo", str(nested)), outcomes[0].rows)

    def test_interactive_yes_installs_into_nested_repos(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            nested = self._workspace(root)
            parsed = hooks_cli.HookNamespace(project=str(root))
            with (
                mock.patch.object(sys.stdin, "isatty", return_value=True),
                mock.patch.object(sys.stderr, "isatty", return_value=True),
                mock.patch.object(hooks_cli, "prompt_nested_repos", return_value=True),
            ):
                outcomes = hooks_cli._install_git_with_nested(parsed)  # pyright: ignore[reportPrivateUsage]
            self.assertEqual([o.action for o in outcomes], ["installed", "installed"])
            self.assertTrue(_post_commit_path(nested).exists())

    def test_uninstall_only_offers_nested_repos_with_our_hook(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            nested = self._workspace(root)
            _ = install.install_git(root, uninstall=False, dry_run=False)
            parsed = hooks_cli.HookNamespace(project=str(root), uninstall=True)
            outcomes = hooks_cli._install_git_with_nested(parsed)  # pyright: ignore[reportPrivateUsage]
            self.assertEqual(len(outcomes), 1)
            self.assertNotIn(("nested repo", str(nested)), outcomes[0].rows)


class ReindexMarkerTests(unittest.TestCase):
    @staticmethod
    def _repo(root: Path, name: str) -> Path:
        repo = root / name
        (repo / ".git").mkdir(parents=True)
        return repo

    def test_marker_round_trip_replays_workspace_invocation(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            repo_a = self._repo(workspace, "repo-a")
            repo_b = self._repo(workspace, "repo-b")
            with mock.patch.object(Path, "cwd", return_value=workspace):
                runtime.write_reindex_markers([repo_a, repo_b], "my-workspace")
            target = runtime.reindex_target(repo_b)
            if target is None:
                self.fail("expected a reindex target from a valid marker")
            self.assertEqual(target[0], workspace)
            self.assertEqual(target[1], ["--collection", "my-workspace", str(repo_a), str(repo_b)])

    def test_marker_without_collection_omits_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            repo = self._repo(workspace, "repo")
            with mock.patch.object(Path, "cwd", return_value=workspace):
                runtime.write_reindex_markers([repo], None)
            self.assertEqual(runtime.reindex_target(repo), (workspace, [str(repo)]))

    def test_missing_or_invalid_marker_skips(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = self._repo(Path(tmp).resolve(), "repo")
            self.assertIsNone(runtime.reindex_target(repo))
            _ = (repo / ".git" / "pci-reindex.json").write_text("not json", encoding="utf-8")
            self.assertIsNone(runtime.reindex_target(repo))

    def test_marker_with_vanished_workspace_skips(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = self._repo(Path(tmp).resolve(), "repo")
            payload = json.dumps({"cwd": str(Path(tmp) / "gone"), "repo_paths": [str(repo)], "collection": None})
            _ = (repo / ".git" / "pci-reindex.json").write_text(payload, encoding="utf-8")
            self.assertIsNone(runtime.reindex_target(repo))

    def test_git_worktree_has_no_marker_and_skips(self) -> None:
        # A worktree's .git is a file, so the marker read fails and no reindex runs.
        with TemporaryDirectory() as tmp:
            worktree = Path(tmp).resolve() / "feature-wt"
            worktree.mkdir()
            _ = (worktree / ".git").write_text("gitdir: /elsewhere/.git/worktrees/feature-wt\n", encoding="utf-8")
            self.assertIsNone(runtime.reindex_target(worktree))

    def test_worktree_replays_under_main_repos_marker(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            main_repo = self._repo(workspace, "main-repo")
            with mock.patch.object(Path, "cwd", return_value=workspace):
                runtime.write_reindex_markers([main_repo], "my-workspace")
            worktree = workspace / "feature-wt"
            worktree.mkdir()
            _ = (worktree / ".git").write_text(f"gitdir: {main_repo}/.git/worktrees/feature-wt\n", encoding="utf-8")
            target = runtime.reindex_target(worktree)
            if target is None:
                self.fail("expected a worktree reindex target")
            self.assertEqual(target[0], workspace)
            self.assertEqual(
                target[1],
                ["--collection", "my-workspace", "--worktree", f"{main_repo}={worktree.resolve()}"],
            )

    def test_worktree_with_malformed_gitdir_pointer_skips(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            worktree = workspace / "feature-wt"
            worktree.mkdir()
            # Doesn't end in /worktrees/<name> -- unparseable, must fall through to None.
            _ = (worktree / ".git").write_text("gitdir: /elsewhere/.git\n", encoding="utf-8")
            self.assertIsNone(runtime.reindex_target(worktree))

    def test_worktree_whose_main_repo_missing_from_marker_skips(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            main_repo = self._repo(workspace, "main-repo")
            other_repo = self._repo(workspace, "other-repo")
            with mock.patch.object(Path, "cwd", return_value=workspace):
                # Both get a marker (so main_repo's own marker read succeeds), but neither
                # marker's repo_paths lists main_repo -- it was never indexed on purpose,
                # so the worktree reindex must not replay against it.
                runtime.write_reindex_markers([main_repo, other_repo], "my-workspace")
            payload = json.dumps({"cwd": str(workspace), "repo_paths": [str(other_repo)], "collection": "my-workspace"})
            _ = (main_repo / ".git" / "pci-reindex.json").write_text(payload, encoding="utf-8")
            worktree = workspace / "feature-wt"
            worktree.mkdir()
            _ = (worktree / ".git").write_text(f"gitdir: {main_repo}/.git/worktrees/feature-wt\n", encoding="utf-8")
            self.assertIsNone(runtime.reindex_target(worktree))

    def test_run_reindex_without_marker_spawns_nothing(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = self._repo(Path(tmp).resolve(), "repo")
            with mock.patch.object(runtime.process, "run") as run_mock:
                self.assertEqual(runtime.run_reindex(repo), 0)
            run_mock.assert_not_called()


if __name__ == "__main__":
    _ = unittest.main()
