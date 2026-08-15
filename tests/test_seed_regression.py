from __future__ import annotations

import unittest

from scripts import seed_regression


class SeedRegressionTests(unittest.TestCase):
    def test_extracts_candidate_ids_from_both_tiers(self) -> None:
        payload = {
            "collection/repo": {
                "redundancy": {
                    "near_certain": [{"id": "redundancy-near"}],
                    "candidates_unranked": [{"id": "redundancy-candidate"}],
                }
            }
        }
        self.assertEqual(
            seed_regression.audit_candidate_ids(payload),
            {"redundancy-near", "redundancy-candidate"},
        )

    def test_open_ledger_entries_are_fixed_expectations(self) -> None:
        payload = {
            "candidates": {
                "redundancy-fixed": {"status": "open"},
                "redundancy-dismissed": {"status": "dismissed"},
            }
        }
        self.assertEqual(seed_regression.fixed_candidate_ids(payload), {"redundancy-fixed"})

    def test_reports_missing_active_and_resurfaced_fixed_seeds(self) -> None:
        failures = seed_regression.regression_failures(
            {"redundancy-fixed"},
            {"redundancy-fixed"},
        )
        self.assertEqual(
            failures,
            [
                "active seed missing: redundancy-78d93855a704",
                "fixed seed resurfaced: redundancy-fixed",
            ],
        )


if __name__ == "__main__":
    _ = unittest.main()
