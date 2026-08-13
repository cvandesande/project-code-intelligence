from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project_code_intelligence import profile_context
from project_code_intelligence.code_profiles.base import GenericProfile
from project_code_intelligence.inventory import (
    DiscoveryReuse,
    classify_file,
    discover_files,
    git_dirty_paths,
    inspect_inventory_file,
    language_for,
    language_for_read_file,
    should_parse_text,
)
from project_code_intelligence.models import PreviousFileState, Snapshot


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
            "Sources/App.swift": "swift",
            "web/index.html": "html",
            "web/styles.css": "css",
            "web/styles.scss": "scss",
            "web/App.vue": "vue",
            "web/App.svelte": "svelte",
            "api/schema.graphql": "graphql",
            "api/query.gql": "graphql",
            "BUILD": "bazel",
            "BUILD.bazel": "bazel",
            "WORKSPACE": "bazel",
            "WORKSPACE.bazel": "bazel",
            "MODULE.bazel": "bazel",
            ".bazelrc": "bazel",
            "tools/rules.bzl": "starlark",
            "tools/rules.star": "starlark",
            "build.gradle": "groovy",
            "Jenkinsfile": "groovy",
            "scripts/deploy.ps1": "powershell",
            "modules/Demo.scala": "scala",
            "build.sbt": "scala",
            "lib/demo.ex": "elixir",
            "lib/demo.exs": "elixir",
            "src/demo.erl": "erlang",
            "include/demo.hrl": "erlang",
            "src/main.zig": "zig",
            "src/App.m": "objective_c",
            "src/App.mm": "objective_cpp",
            "pkg/tool.pl": "perl",
            "web/index.php": "php",
            "lib/demo.rb": "ruby",
            "schema/demo.proto": "protobuf",
            "db/schema.sql": "sql",
            "config/settings.xml": "xml",
            "Dockerfile": "dockerfile",
            "Containerfile": "dockerfile",
            "docker/app.dockerfile": "dockerfile",
            "build/Dockerfile.ubi8": "dockerfile",
            "build/Dockerfile.ubi9": "dockerfile",
            "build/Dockerfile.ubi10": "dockerfile",
            "build/Dockerfile-debug": "dockerfile",
            "build/Containerfile.fedora": "dockerfile",
            "infra/main.tf": "terraform",
            "infra/vars.tfvars": "terraform",
            "infra/build.pkr.hcl": "packer",
            "CMakeLists.txt": "cmake",
            "cmake/toolchain.cmake": "cmake",
            "meson.build": "meson",
            "meson_options.txt": "meson",
            "target/demo.dtsi": "dts",
            "target/demo.dtso": "dts",
            "target/config-6.12": "config",
            "target/image/linker.lds": "linker_script",
            "target/image/boot.bootscript": "boot_script",
            "include/scan.awk": "awk",
            "scripts/config/lexer.l": "lex",
            "scripts/config/parser.y": "yacc",
            "docs/README.md": "doc",
            "service/config": "config",
        }

        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(language_for(path), expected)

    def test_language_for_read_file_detects_dockerfile_syntax_directive(self) -> None:
        # BuildKit Dockerfiles can be named anything when paired with `# syntax=docker/dockerfile:...`.
        # Sniff the directive so e.g. `infra/recipe.txt` containing that pragma is classified properly.
        self.assertEqual(
            language_for_read_file(
                "infra/recipe.txt",
                b"# syntax=docker/dockerfile:1.6\nFROM alpine\n",
                read_ok=True,
            ),
            "dockerfile",
        )
        # Case-insensitive: `# SYNTAX=docker/dockerfile:...` is valid per the directive parser.
        self.assertEqual(
            language_for_read_file(
                "infra/recipe.txt",
                b"# SYNTAX=docker/dockerfile:1.6\nFROM alpine\n",
                read_ok=True,
            ),
            "dockerfile",
        )

    def test_language_for_read_file_uses_shebang_for_unsuffixed_scripts(self) -> None:
        self.assertEqual(language_for_read_file("scripts/env", b"#!/usr/bin/env bash\n", read_ok=True), "shell")
        self.assertEqual(language_for_read_file("scripts/checkpatch", b"#!/usr/bin/env perl\n", read_ok=True), "perl")
        self.assertEqual(
            language_for_read_file("scripts/bootstrap", b"#!/usr/bin/env pwsh\n", read_ok=True), "powershell"
        )
        self.assertEqual(language_for_read_file("usr/sbin/provision", b"#!/usr/bin/env ucode\n", read_ok=True), "ucode")
        self.assertEqual(
            language_for_read_file("include/AppDelegate.h", b"#import <Foundation/Foundation.h>\n", read_ok=True),
            "objective_c",
        )

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

    def test_classify_file_marks_generated_go_codegen_paths(self) -> None:
        paths = (
            "pkg/client/applyconfiguration/example/v1/virtualserver.go",
            "pkg/client/clientset/versioned/clientset.go",
            "pkg/client/informers/externalversions/factory.go",
            "pkg/client/listers/example/v1/virtualserver.go",
            "pkg/apis/example/v1/zz_generated.deepcopy.go",
        )

        for path in paths:
            with self.subTest(path=path):
                classified = classify_file(path, "go")
                self.assertTrue(classified["is_generated"])
                self.assertEqual(classified["content_class"], "generated")
                self.assertEqual(classified["file_role"], "generated")

    def test_classify_file_marks_standard_generated_header(self) -> None:
        classified = classify_file(
            "pkg/apis/example/v1/types.go",
            "go",
            "// Code generated by controller-gen. DO NOT EDIT.\npackage v1\n",
        )

        self.assertTrue(classified["is_generated"])
        self.assertEqual(classified["content_class"], "generated")

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

            # NUL after the first 4 KB of ASCII preamble — the prior detector
            # only scanned 4096 bytes and would miss this case.
            deep_nul_file = root / "deep_nul.txt"
            _ = deep_nul_file.write_bytes(b"a" * 8192 + b"\x00rest")
            reason, _data, _size_bytes, read_ok = inspect_inventory_file(deep_nul_file, max_file_bytes=1_000_000)
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
            _ = (root / "scripts").mkdir()
            _ = (root / "src" / "main.py").write_text("def main():\n    return 0\n", encoding="utf-8")
            _ = (root / "scripts" / "env").write_text("#!/usr/bin/env bash\nrun_demo() { true; }\n", encoding="utf-8")
            _ = (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            _ = (root / "image.png").write_bytes(b"png bytes")
            previous_profile = profile_context.active_profile
            try:
                profile_context.set_active_profile(GenericProfile())
                with patch(
                    "project_code_intelligence.inventory.git_ls_files",
                    return_value=[
                        ("a" * 40, "src/main.py"),
                        ("e" * 40, "scripts/env"),
                        ("b" * 40, "README.md"),
                        ("c" * 40, "image.png"),
                        ("d" * 40, "missing.py"),
                    ],
                ):
                    files = discover_files(root, snapshot_for("."), max_file_bytes=1024)
            finally:
                profile_context.set_active_profile(previous_profile)

        by_path = {item.repo_rel_path: item for item in files}
        self.assertEqual(sorted(by_path), ["README.md", "image.png", "scripts/env", "src/main.py"])
        self.assertEqual(by_path["src/main.py"].source_path, "src/main.py")
        self.assertEqual(by_path["src/main.py"].language, "python")
        self.assertEqual(by_path["src/main.py"].file_role, "source")
        self.assertEqual(by_path["src/main.py"].content_class, "source")
        self.assertEqual(by_path["src/main.py"].metadata["path_parts"], ["src", "main.py"])
        self.assertEqual(by_path["src/main.py"].metadata["python_module"], "main")
        self.assertEqual(by_path["src/main.py"].metadata["python_functions"], ["main"])
        self.assertEqual(by_path["scripts/env"].language, "shell")
        self.assertEqual(by_path["scripts/env"].metadata["shell_functions"], ["run_demo"])
        self.assertEqual(by_path["README.md"].content_class, "doc")
        self.assertEqual(by_path["image.png"].skipped_reason, "binary_suffix")

    def test_discover_files_scan_root_overrides_repo_root_but_keeps_identity(self) -> None:
        # Worktree reindex case: content is scanned at ``scan_root`` (the worktree
        # checkout), while source paths stay relative to the stamped repo identity.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            other_root = root / "elsewhere"
            _ = other_root.mkdir()
            _ = (other_root / "main.py").write_text("def main():\n    return 0\n", encoding="utf-8")
            previous_profile = profile_context.active_profile
            try:
                profile_context.set_active_profile(GenericProfile())
                with patch(
                    "project_code_intelligence.inventory.git_ls_files",
                    return_value=[("a" * 40, "main.py")],
                ) as ls_files_mock:
                    files = discover_files(
                        root, snapshot_for("worktree-repo"), max_file_bytes=1024, scan_root=other_root
                    )
            finally:
                profile_context.set_active_profile(previous_profile)

        ls_files_mock.assert_called_once_with(other_root)
        self.assertEqual([item.source_path for item in files], ["worktree-repo/main.py"])

    def test_discover_files_reuses_previous_clean_blob_without_reading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src" / "main.py"
            source.parent.mkdir()
            _ = source.write_text("def changed_on_disk():\n    return 1\n", encoding="utf-8")
            previous = PreviousFileState(
                source_path="src/main.py",
                git_blob_sha="a" * 40,
                file_sha256="previous-file-sha",
                size_bytes=25,
                language="python",
                file_role="source",
                content_class="source",
                is_generated=False,
                is_vendor=False,
                is_test=False,
                is_source=True,
                is_build=False,
                is_config=False,
                is_doc=False,
                skipped_reason=None,
                metadata={"python_functions": ["previous"]},
            )

            with (
                patch("project_code_intelligence.inventory.git_ls_files", return_value=[("a" * 40, "src/main.py")]),
                patch("project_code_intelligence.inventory.inspect_inventory_file") as mocked_inspect,
            ):
                files = discover_files(
                    root,
                    snapshot_for("."),
                    max_file_bytes=1024,
                    reuse=DiscoveryReuse(
                        previous_files={"src/main.py": previous},
                        reuse_unchanged_blobs=True,
                    ),
                )

        mocked_inspect.assert_not_called()
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].file_sha256, "previous-file-sha")
        self.assertEqual(files[0].metadata["python_functions"], ["previous"])

    def test_discover_files_reuses_clean_blob_in_dirty_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_source = root / "src" / "clean.py"
            dirty_source = root / "src" / "dirty.py"
            clean_source.parent.mkdir()
            _ = clean_source.write_text("def clean_on_disk():\n    return 1\n", encoding="utf-8")
            _ = dirty_source.write_text("def dirty_on_disk():\n    return 1\n", encoding="utf-8")
            previous_clean = PreviousFileState(
                source_path="src/clean.py",
                git_blob_sha="a" * 40,
                file_sha256="previous-clean-sha",
                size_bytes=25,
                language="python",
                file_role="source",
                content_class="source",
                is_generated=False,
                is_vendor=False,
                is_test=False,
                is_source=True,
                is_build=False,
                is_config=False,
                is_doc=False,
                skipped_reason=None,
                metadata={"python_functions": ["clean_previous"]},
            )
            previous_dirty = PreviousFileState(
                source_path="src/dirty.py",
                git_blob_sha="b" * 40,
                file_sha256="previous-dirty-sha",
                size_bytes=25,
                language="python",
                file_role="source",
                content_class="source",
                is_generated=False,
                is_vendor=False,
                is_test=False,
                is_source=True,
                is_build=False,
                is_config=False,
                is_doc=False,
                skipped_reason=None,
                metadata={"python_functions": ["dirty_previous"]},
            )

            with patch(
                "project_code_intelligence.inventory.git_ls_files",
                return_value=[("a" * 40, "src/clean.py"), ("b" * 40, "src/dirty.py")],
            ):
                files = discover_files(
                    root,
                    snapshot_for("."),
                    max_file_bytes=1024,
                    reuse=DiscoveryReuse(
                        previous_files={"src/clean.py": previous_clean, "src/dirty.py": previous_dirty},
                        reuse_unchanged_blobs=True,
                        dirty_paths=frozenset({"src/dirty.py"}),
                    ),
                )

        by_path = {item.repo_rel_path: item for item in files}
        self.assertEqual(by_path["src/clean.py"].file_sha256, "previous-clean-sha")
        self.assertEqual(by_path["src/clean.py"].metadata["python_functions"], ["clean_previous"])
        self.assertNotEqual(by_path["src/dirty.py"].file_sha256, "previous-dirty-sha")
        self.assertEqual(by_path["src/dirty.py"].metadata["python_functions"], ["dirty_on_disk"])

    def test_discover_files_skips_language_metadata_dispatch_for_plain_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _ = (root / "README.txt").write_text("plain text\n", encoding="utf-8")
            with (
                patch("project_code_intelligence.inventory.git_ls_files", return_value=[("a" * 40, "README.txt")]),
                patch("project_code_intelligence.inventory.language_metadata_for_file") as metadata_for_file,
            ):
                files = discover_files(root, snapshot_for("."), max_file_bytes=1024)

        metadata_for_file.assert_not_called()
        self.assertEqual(files[0].metadata["path_parts"], ["README.txt"])

    def test_git_dirty_paths_parses_porcelain_paths_and_renames(self) -> None:
        with patch(
            "project_code_intelligence.inventory.run_git",
            return_value=" M src/app.py\nR  old/name.py -> new/name.py\n D deleted.py\n",
        ):
            self.assertEqual(
                git_dirty_paths(Path("/repo")),
                {"src/app.py", "old/name.py", "new/name.py", "deleted.py"},
            )


if __name__ == "__main__":
    _ = unittest.main()
