import json
import os
import unittest
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from typing_extensions import override

from project_code_intelligence import check
from project_code_intelligence.check_core import BaselineEntry, CheckFinding, Regression
from project_code_intelligence.models import StaticFinding, StaticRun


@contextmanager
def _chdir(path: Path) -> Generator[None]:
    """Basedpyright's configured `pythonVersion = "3.10"` predates `contextlib.chdir` (3.11)."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class ResolveCheckIdentityTests(unittest.TestCase):
    def test_infers_repo_and_collection_from_cwd(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "my-repo"
            root.mkdir()
            with mock.patch.object(check, "resolve_repo_branch", return_value="feature-branch"):
                identity = check.resolve_check_identity(root, collection=None, repo=None)
        self.assertEqual(identity.repo, "my-repo")
        self.assertEqual(identity.collection, "my-repo")
        self.assertEqual(identity.branch, "feature-branch")

    def test_explicit_overrides_win(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "my-repo"
            root.mkdir()
            with mock.patch.object(check, "resolve_repo_branch", return_value="main"):
                identity = check.resolve_check_identity(root, collection="coll", repo="repo")
        self.assertEqual(identity.repo, "repo")
        self.assertEqual(identity.collection, "coll")

    def test_detached_head_is_none_not_a_literal_head_branch(self) -> None:
        """None is a distinct baseline bucket, matching resolve_repo_branch/snapshot identity --
        not the string "HEAD", which would collide with a real branch named HEAD."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "my-repo"
            root.mkdir()
            with mock.patch.object(check, "resolve_repo_branch", return_value=None):
                identity = check.resolve_check_identity(root, collection=None, repo=None)
        self.assertIsNone(identity.branch)


class StaticFindingToCheckFindingTests(unittest.TestCase):
    def test_reduces_run_and_finding_to_check_finding(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _ = (root / "a.py").write_text("bad_call()\n", encoding="utf-8")
            run = StaticRun(repo=".", sarif_path="out.sarif", sarif_sha256="x", run_index=0, tool_name="semgrep")
            finding = StaticFinding(
                finding_key="k",
                rule_id="RULE1",
                message="bad",
                level="warning",
                primary_source_path="a.py",
                line_start=1,
            )
            check_finding = check.static_finding_to_check_finding(root, run, finding)
        self.assertEqual(check_finding.rule_id, "RULE1")
        self.assertEqual(check_finding.tool_name, "semgrep")
        self.assertEqual(check_finding.level, "warning")


class RenderRegressionsTests(unittest.TestCase):
    def test_empty_list_renders_empty_string(self) -> None:
        self.assertEqual(check.render_regressions([]), "")

    def test_one_regression_renders_one_line(self) -> None:
        finding = CheckFinding(
            fingerprint="fp1",
            rule_id="RULE1",
            level="error",
            tool_name="tool",
            message="bad",
            primary_source_path="a.py",
            line_start=1,
            line_end=1,
        )
        rendered = check.render_regressions([Regression(status="new", finding=finding)])
        self.assertEqual(rendered.count("\n"), 1)


class _FakeConnection:
    @staticmethod
    def commit() -> None:
        return None

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        return None


def _write_sarif(path: Path, *, rule_id: str, level: str, line: int, source_path: str = "a.py") -> None:
    payload = {
        "version": "2.1.0",
        "runs": [
            {
                "tool": {"driver": {"name": "demo-tool"}},
                "results": [
                    {
                        "ruleId": rule_id,
                        "level": level,
                        "message": {"text": "bad thing"},
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": source_path},
                                    "region": {"startLine": line},
                                }
                            }
                        ],
                    }
                ],
            }
        ],
    }
    _ = path.write_text(json.dumps(payload), encoding="utf-8")


class CheckMainEndToEndTests(unittest.TestCase):
    """`check_main` exit codes, storage mocked with an in-memory baseline dict so these run
    without Postgres -- `freeze_baseline`/`load_baseline` themselves are exercised against a
    real database in the manual verification described in the Phase 1 summary."""

    @override
    def setUp(self) -> None:
        self._store: dict[tuple[str, str, str | None], list[BaselineEntry]] = {}
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name) / "repo"
        self.repo.mkdir()
        _ = (self.repo / "a.py").write_text("x = 1\nbad_call()\ny = 2\n", encoding="utf-8")
        self._patches = [
            mock.patch.object(check, "resolve_repo_branch", return_value="main"),
            mock.patch.object(check, "ensure_schema"),
            mock.patch.object(check.db, "connect", return_value=_FakeConnection()),
            mock.patch.object(check, "freeze_baseline", side_effect=self._fake_freeze),
            mock.patch.object(check, "load_baseline", side_effect=self._fake_load),
        ]
        for patcher in self._patches:
            _ = patcher.start()
            self.addCleanup(patcher.stop)

    def _fake_freeze(
        self, _conn: object, *, collection: str, repo: str, branch: str | None, findings: list[CheckFinding]
    ) -> int:
        self._store[collection, repo, branch] = [
            BaselineEntry(fingerprint=f.fingerprint, rule_id=f.rule_id, level=f.level) for f in findings
        ]
        return len(findings)

    def _fake_load(
        self, _conn: object, *, collection: str, repo: str, branch: str | None
    ) -> list[BaselineEntry] | None:
        return self._store.get((collection, repo, branch))

    def test_baseline_then_clean_rerun_exits_0(self) -> None:
        sarif_path = self.repo / "out.sarif"
        _write_sarif(sarif_path, rule_id="RULE1", level="warning", line=2)
        with _chdir(self.repo):
            self.assertEqual(check.check_main(["--baseline", "out.sarif"]), 0)
            self.assertEqual(check.check_main(["out.sarif"]), 0)

    def test_new_finding_exits_1(self) -> None:
        sarif_path = self.repo / "out.sarif"
        _write_sarif(sarif_path, rule_id="RULE1", level="warning", line=2)
        with _chdir(self.repo):
            self.assertEqual(check.check_main(["--baseline", "out.sarif"]), 0)
            _write_sarif(sarif_path, rule_id="RULE2", level="warning", line=1)
            self.assertEqual(check.check_main(["out.sarif"]), 1)

    def test_missing_baseline_exits_1(self) -> None:
        """Documented in docs/PUBLIC_API.md: `pci check` requires an explicit
        `--baseline` bootstrap; it does not silently pass on a first run."""
        sarif_path = self.repo / "out.sarif"
        _write_sarif(sarif_path, rule_id="RULE1", level="warning", line=2)
        with _chdir(self.repo):
            self.assertEqual(check.check_main(["out.sarif"]), 1)


if __name__ == "__main__":
    _ = unittest.main()
