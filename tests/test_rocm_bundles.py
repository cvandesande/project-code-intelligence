from __future__ import annotations

import unittest

from project_code_intelligence.rocm_bundles import (
    RocmBundleError,
    bundle_for_gfx_target,
    gfx_target_from_pci_ids,
    gfx_targets_from_text,
    llama_rocm_asset_name,
    llama_rocm_download_url,
    normalize_github_repo,
    normalize_rocm_bundle,
)


class RocmBundleTests(unittest.TestCase):
    def test_detects_unique_gfx_targets_from_rocm_output(self) -> None:
        output = """
        Name:                    gfx1100
        Marketing Name:          AMD Radeon
        Name:                    gfx1100
        Name:                    gfx1036
        """

        self.assertEqual(gfx_targets_from_text(output), ["gfx1100", "gfx1036"])

    def test_detects_known_gfx_targets_from_pci_or_marketing_output(self) -> None:
        self.assertEqual(gfx_targets_from_text("IDs (DID, GUID) 0x1586, 15365"), ["gfx1151"])
        self.assertEqual(
            gfx_targets_from_text("Display controller: AMD Strix Halo [Radeon 8060S Graphics]"),
            ["gfx1151"],
        )

    def test_detects_known_gfx_targets_from_pci_ids(self) -> None:
        self.assertEqual(gfx_target_from_pci_ids("0x1002", "0x1586"), "gfx1151")
        self.assertIsNone(gfx_target_from_pci_ids("0x8086", "0x1586"))

    def test_maps_gfx_targets_to_lemonade_bundle_families(self) -> None:
        self.assertEqual(bundle_for_gfx_target("gfx1030"), "gfx103X")
        self.assertEqual(bundle_for_gfx_target("gfx1100"), "gfx110X")
        self.assertEqual(bundle_for_gfx_target("gfx1102"), "gfx110X")
        self.assertEqual(bundle_for_gfx_target("gfx1150"), "gfx1150")
        self.assertEqual(bundle_for_gfx_target("gfx1151"), "gfx1151")
        self.assertEqual(bundle_for_gfx_target("gfx1201"), "gfx120X")

    def test_accepts_bundle_or_exact_gfx_override(self) -> None:
        self.assertEqual(normalize_rocm_bundle("gfx110X"), "gfx110X")
        self.assertEqual(normalize_rocm_bundle("gfx1100"), "gfx110X")

    def test_rejects_unknown_gfx_target(self) -> None:
        with self.assertRaises(RocmBundleError):
            _ = bundle_for_gfx_target("gfx9999")

    def test_builds_release_asset_name_and_url(self) -> None:
        self.assertEqual(
            llama_rocm_asset_name("b1264", "gfx110X"),
            "llama-b1264-ubuntu-rocm-gfx110X-x64.zip",
        )
        self.assertEqual(
            llama_rocm_download_url("lemonade-sdk/llamacpp-rocm", "b1264", "gfx110X"),
            "https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/"
            "b1264/llama-b1264-ubuntu-rocm-gfx110X-x64.zip",
        )

    def test_rejects_unsafe_repository_and_release_values(self) -> None:
        with self.assertRaises(RocmBundleError):
            _ = normalize_github_repo("owner/name/extra")
        with self.assertRaises(RocmBundleError):
            _ = llama_rocm_download_url("lemonade-sdk/llamacpp-rocm", "../bad", "gfx110X")


if __name__ == "__main__":
    _ = unittest.main()
