from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project_code_intelligence import cli


class CliWrapperTests(unittest.TestCase):
    def test_pci_index_enables_embeddings_by_default(self) -> None:
        forwarded: list[str] = []

        def fake_ingest_main(args: list[str]) -> int:
            forwarded.extend(args)
            return 0

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
        ):
            status = cli.index_main([".", "--dry-run"])

        self.assertEqual(status, 0)
        self.assertIn("--embed", forwarded)
        self.assertIn("--embedding-endpoint", forwarded)
        self.assertIn("--root", forwarded)
        self.assertIn("--repos", forwarded)
        self.assertEqual(forwarded[forwarded.index("--repos") + 1], ".")
        self.assertIn("--dry-run", forwarded)

    def test_pci_index_requires_repo_path_for_indexing(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main") as ingest_main,
            patch("sys.stderr", io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            _ = cli.index_main(["--dry-run"])

        self.assertEqual(raised.exception.code, 2)
        ingest_main.assert_not_called()

    def test_repo_paths_map_to_common_root_and_repo_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo_a = root / "repo-a"
            repo_b = root / "repo-b"
            repo_a.mkdir()
            repo_b.mkdir()

            args = cli.repo_paths_to_ingest_args([str(repo_a), str(repo_b)])

        self.assertEqual(args, ["--root", str(root.resolve()), "--repos", "repo-a,repo-b"])

    def test_pci_index_no_embed_is_explicit_text_only_mode(self) -> None:
        forwarded: list[str] = []

        def fake_ingest_main(args: list[str]) -> int:
            forwarded.extend(args)
            return 0

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
        ):
            status = cli.index_main([".", "--no-embed", "--dry-run"])
            self.assertEqual(os.environ["PROJECT_CODE_INTELLIGENCE_EMBED"], "0")

        self.assertEqual(status, 0)
        self.assertNotIn("--embed", forwarded)
        self.assertNotIn("--embedding-endpoint", forwarded)
        self.assertIn("--dry-run", forwarded)

    def test_pci_index_exposes_and_forwards_reset_flags(self) -> None:
        forwarded: list[str] = []

        def fake_ingest_main(args: list[str]) -> int:
            forwarded.extend(args)
            return 0

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
        ):
            status = cli.index_main([
                "--reset-code-intel",
                "--i-know-this-deletes-code-intel-db",
            ])

        self.assertEqual(status, 0)
        self.assertIn("--reset-code-intel", cli.index_parser().format_help())
        self.assertIn("--reset-code-intel", forwarded)
        self.assertIn("--reset-only", forwarded)
        self.assertIn("--i-know-this-deletes-code-intel-db", forwarded)
        self.assertNotIn("--embed", forwarded)
        self.assertNotIn("--embedding-endpoint", forwarded)
