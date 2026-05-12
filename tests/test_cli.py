from __future__ import annotations

import os
import unittest
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
            status = cli.index_main(["--dry-run"])

        self.assertEqual(status, 0)
        self.assertIn("--embed", forwarded)
        self.assertIn("--embedding-endpoint", forwarded)
        self.assertIn("--dry-run", forwarded)

    def test_pci_index_no_embed_is_explicit_text_only_mode(self) -> None:
        forwarded: list[str] = []

        def fake_ingest_main(args: list[str]) -> int:
            forwarded.extend(args)
            return 0

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("project_code_intelligence.cli.ingest_code_intel.cli_main", side_effect=fake_ingest_main),
        ):
            status = cli.index_main(["--no-embed", "--dry-run"])
            self.assertEqual(os.environ["PROJECT_CODE_INTELLIGENCE_EMBED"], "0")

        self.assertEqual(status, 0)
        self.assertNotIn("--embed", forwarded)
        self.assertNotIn("--embedding-endpoint", forwarded)
        self.assertIn("--dry-run", forwarded)
