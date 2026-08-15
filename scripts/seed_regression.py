"""Check the source-verified redundancy seeds against the current audit."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))

from project_code_intelligence import process  # noqa: E402

TRIAGE_PATH = REPO_ROOT / ".pci" / "audit-triage.json"

# Seed 14 remains active. The other seeds were fixed and are represented by
# the still-open entries in the historical triage ledger.
ACTIVE_SEED_IDS = frozenset({"redundancy-78d93855a704"})


def audit_candidate_ids(payload: object) -> set[str]:
    if not isinstance(payload, dict):
        raise TypeError("audit output must be a JSON object")
    candidate_ids: set[str] = set()
    for result in cast("dict[object, object]", payload).values():
        if not isinstance(result, dict):
            continue
        redundancy = cast("dict[object, object]", result).get("redundancy")
        if not isinstance(redundancy, dict):
            continue
        typed_redundancy = cast("dict[object, object]", redundancy)
        for key in ("near_certain", "candidates_unranked"):
            groups = typed_redundancy.get(key)
            if not isinstance(groups, list):
                continue
            for group in cast("list[object]", groups):
                if not isinstance(group, dict):
                    continue
                candidate_id = cast("dict[object, object]", group).get("id")
                if isinstance(candidate_id, str):
                    candidate_ids.add(candidate_id)
    return candidate_ids


def fixed_candidate_ids(payload: object) -> set[str]:
    if not isinstance(payload, dict):
        raise TypeError("triage ledger must be a JSON object")
    candidates = cast("dict[object, object]", payload).get("candidates")
    if not isinstance(candidates, dict):
        raise TypeError("triage ledger has no candidates object")
    fixed: set[str] = set()
    for candidate_id, entry in cast("dict[object, object]", candidates).items():
        if not isinstance(candidate_id, str) or not isinstance(entry, dict):
            continue
        if cast("dict[object, object]", entry).get("status") == "open":
            fixed.add(candidate_id)
    return fixed


def regression_failures(current: set[str], fixed: set[str]) -> list[str]:
    failures = [f"active seed missing: {candidate_id}" for candidate_id in sorted(ACTIVE_SEED_IDS - current)]
    failures.extend(f"fixed seed resurfaced: {candidate_id}" for candidate_id in sorted(fixed & current))
    return failures


def run_audit() -> object:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = process.run(
        [sys.executable, "-m", "project_code_intelligence.pci", "audit", "--json", "--limit", "999"],
        process.RunOptions(cwd=REPO_ROOT, env=env, capture_output=True, check=False),
    )
    if proc.returncode != 0:
        if proc.stderr:
            _ = sys.stderr.write(proc.stderr)
        raise SystemExit(f"pci audit exited with {proc.returncode}")
    return cast("object", json.loads(proc.stdout))


def main() -> int:
    current = audit_candidate_ids(run_audit())
    triage = cast("object", json.loads(TRIAGE_PATH.read_text(encoding="utf-8")))
    failures = regression_failures(current, fixed_candidate_ids(triage))
    if failures:
        for failure in failures:
            _ = sys.stderr.write(f"seed regression: {failure}\n")
        return 1
    _ = sys.stdout.write(f"seed regression passed ({len(ACTIVE_SEED_IDS)} active, {len(current)} candidates)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
