from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project_code_intelligence import profile_context
from project_code_intelligence.code_profiles.base import GenericProfile
from project_code_intelligence.inventory import (
    classify_file,
    discover_files,
    inspect_inventory_file,
    language_for,
    should_parse_text,
)
from project_code_intelligence.models import Snapshot


def snapshot_for(repo: str) -> Snapshot:
    return Snapshot(
        collection="test",
        repo=repo,
        repo_role="project",
        branch="main",
        commit_sha="commit",
        tree_sha="tree",
        dirty=False,
        metadata={},
    )


class InventoryContractTests(unittest.TestCase):
    def test_language_for_common_project_files(self) -> None:
        cases = {
            "src/main.c": "c",
            "include/demo.hpp": "c",
            "arch/start.S": "asm",
            "Kconfig": "kconfig",
            "Config.in": "kconfig",
            "Makefile": "make",
            "scripts/build.sh": "shell",
            "etc/init.d/demo": "shell",
            "pkg/module.tsx": "typescript",
            "docs/README.md": "doc",
            "service/config": "config",
        }

        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(language_for(path), expected)

    def test_classify_file_preserves_public_metadata_contract(self) -> None:
        doc = classify_file("docs/README.md", "doc")
        test_file = classify_file("tests/test_api.py", "python")
        vendor_file = classify_file("vendor/lib/demo.c", "c")
        manifest = classify_file("pyproject.toml", "toml")

        self.assertEqual(doc["content_class"], "doc")
        self.assertTrue(doc["is_doc"])
        self.assertEqual(test_file["content_class"], "test")
        self.assertTrue(test_file["is_test"])
        self.assertEqual(vendor_file["content_class"], "vendor")
        self.assertTrue(vendor_file["is_vendor"])
        self.assertEqual(manifest["content_class"], "build")
        self.assertTrue(manifest["is_build"])
        self.assertTrue(manifest["is_config"])

    def test_inspect_inventory_file_resource_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary_suffix = root / "image.png"
            nul_file = root / "blob.dat"
            large_file = root / "large.txt"
            missing = root / "missing.txt"
            _ = binary_suffix.write_bytes(b"png bytes")
            _ = nul_file.write_bytes(b"abc\x00def")
            _ = large_file.write_text("x" * 20, encoding="utf-8")

            reason, data, size_bytes, read_ok = inspect_inventory_file(binary_suffix, max_file_bytes=100)
            self.assertEqual(reason, "binary_suffix")
            self.assertEqual(data, b"png bytes")
            self.assertEqual(size_bytes, len(b"png bytes"))
            self.assertTrue(read_ok)

            reason, _data, _size_bytes, read_ok = inspect_inventory_file(nul_file, max_file_bytes=100)
            self.assertEqual(reason, "binary_nul")
            self.assertTrue(read_ok)

            reason, data, size_bytes, read_ok = inspect_inventory_file(large_file, max_file_bytes=5)
            self.assertEqual(reason, "file_too_large")
            self.assertEqual(data, b"")
            self.assertEqual(size_bytes, 20)
            self.assertFalse(read_ok)

            reason, data, size_bytes, read_ok = inspect_inventory_file(missing, max_file_bytes=100)
            self.assertEqual(reason, "read_error")
            self.assertEqual(data, b"")
            self.assertEqual(size_bytes, 0)
            self.assertFalse(read_ok)

    def test_should_parse_text_keeps_generic_compatibility_paths(self) -> None:
        self.assertTrue(should_parse_text("README.txt", "text", None))
        self.assertTrue(should_parse_text("package/demo/files/etc/config/demo", "config", None))
        self.assertTrue(should_parse_text("target/base-files/etc/inittab", "config", None))
        self.assertFalse(should_parse_text("image.png", "text", "binary_suffix"))

    def test_discover_files_builds_intel_file_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _ = (root / "src").mkdir()
            _ = (root / "src" / "main.py").write_text("def main():\n    return 0\n", encoding="utf-8")
            _ = (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            _ = (root / "image.png").write_bytes(b"png bytes")
            previous_profile = profile_context.active_profile
            try:
                profile_context.set_active_profile(GenericProfile())
                with patch(
                    "project_code_intelligence.inventory.git_ls_files",
                    return_value=[
                        ("a" * 40, "src/main.py"),
                        ("b" * 40, "README.md"),
                        ("c" * 40, "image.png"),
                        ("d" * 40, "missing.py"),
                    ],
                ):
                    files = discover_files(root, snapshot_for("."), max_file_bytes=1024)
            finally:
                profile_context.set_active_profile(previous_profile)

        by_path = {item.repo_rel_path: item for item in files}
        self.assertEqual(sorted(by_path), ["README.md", "image.png", "src/main.py"])
        self.assertEqual(by_path["src/main.py"].source_path, "src/main.py")
        self.assertEqual(by_path["src/main.py"].language, "python")
        self.assertEqual(by_path["src/main.py"].file_role, "source")
        self.assertEqual(by_path["src/main.py"].content_class, "source")
        self.assertEqual(by_path["src/main.py"].metadata["path_parts"], ["src", "main.py"])
        self.assertEqual(by_path["README.md"].content_class, "doc")
        self.assertEqual(by_path["image.png"].skipped_reason, "binary_suffix")


if __name__ == "__main__":
    _ = unittest.main()
