import io
import json
import os
import unittest
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from project_code_intelligence import rulepack_cli


@contextmanager
def _chdir(path: Path) -> Generator[None]:
    """Basedpyright's configured `pythonVersion = "3.10"` predates `contextlib.chdir` (3.11)."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


_VALID_MANIFEST = json.dumps({
    "name": "sample",
    "version": "1.0.0",
    "rules": [
        {
            "id": "T1",
            "tier": 1,
            "description": "desc",
            "rationale": "why",
            "producer": {"kind": "ast_grep"},
        }
    ],
})


class RulepackCliTests(unittest.TestCase):
    def test_list_with_no_rulepacks_reports_none_found(self) -> None:
        with (
            TemporaryDirectory() as tmp,
            _chdir(Path(tmp)),
            mock.patch.object(rulepack_cli.sys, "stdout", io.StringIO()) as out,
        ):
            code = rulepack_cli.rulepack_main(["list"])
        self.assertEqual(code, 0)
        self.assertIn("no rulepacks found", out.getvalue())

    def test_list_reports_pack_and_tier_counts(self) -> None:
        with TemporaryDirectory() as tmp, _chdir(Path(tmp)):
            manifest_path = Path(tmp) / ".pci" / "rulepacks" / "sample" / "rulepack.json"
            manifest_path.parent.mkdir(parents=True)
            _ = manifest_path.write_text(_VALID_MANIFEST, encoding="utf-8")
            with mock.patch.object(rulepack_cli.sys, "stdout", io.StringIO()) as out:
                code = rulepack_cli.rulepack_main(["list"])
        self.assertEqual(code, 0)
        self.assertIn("sample 1.0.0", out.getvalue())
        self.assertIn("tier 1: 1", out.getvalue())

    def test_validate_succeeds_on_valid_pack(self) -> None:
        with TemporaryDirectory() as tmp, _chdir(Path(tmp)):
            manifest_path = Path(tmp) / ".pci" / "rulepacks" / "sample" / "rulepack.json"
            manifest_path.parent.mkdir(parents=True)
            _ = manifest_path.write_text(_VALID_MANIFEST, encoding="utf-8")
            with mock.patch.object(rulepack_cli.sys, "stdout", io.StringIO()) as out:
                code = rulepack_cli.rulepack_main(["validate"])
        self.assertEqual(code, 0)
        self.assertIn("1 pack(s) valid", out.getvalue())

    def test_validate_fails_on_invalid_pack(self) -> None:
        bad_manifest = _VALID_MANIFEST.replace('"tier": 1', '"tier": 9')
        with TemporaryDirectory() as tmp, _chdir(Path(tmp)):
            manifest_path = Path(tmp) / ".pci" / "rulepacks" / "sample" / "rulepack.json"
            manifest_path.parent.mkdir(parents=True)
            _ = manifest_path.write_text(bad_manifest, encoding="utf-8")
            with mock.patch.object(rulepack_cli.sys, "stderr", io.StringIO()) as err:
                code = rulepack_cli.rulepack_main(["validate"])
        self.assertEqual(code, 1)
        self.assertIn("unknown tier", err.getvalue())

    def test_validate_fails_on_unparseable_pack(self) -> None:
        with TemporaryDirectory() as tmp, _chdir(Path(tmp)):
            manifest_path = Path(tmp) / ".pci" / "rulepacks" / "broken" / "rulepack.json"
            manifest_path.parent.mkdir(parents=True)
            _ = manifest_path.write_text("{not valid json", encoding="utf-8")
            with mock.patch.object(rulepack_cli.sys, "stderr", io.StringIO()) as err:
                code = rulepack_cli.rulepack_main(["validate"])
        self.assertEqual(code, 1)
        self.assertIn("broken", err.getvalue())


if __name__ == "__main__":
    _ = unittest.main()
