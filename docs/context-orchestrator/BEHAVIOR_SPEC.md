# Context Orchestrator: Behavior Specification

## Purpose

Define executable, testable behavior scenarios for context generation,
recommendation logic, and hook/plugin adapter behavior.

## Scenario 1: Bootstrap On Session Start

Given:

- repository is indexed,
- no fresh context artifact exists,
- new assistant session starts.

When:

- client adapter invokes policy evaluation.

Then:

- `generate_mode=bootstrap`,
- `.pci/context/latest.md` and `.pci/context/latest.json` are written,
- payload token budget <= configured limit,
- no fresh-session recommendation emitted.

## Scenario 2: No Regeneration When Fresh

Given:

- context artifact exists,
- HEAD unchanged,
- generation hash unchanged.

When:

- policy evaluation runs.

Then:

- `generate_mode=none`,
- no artifact rewrite,
- no recommendation emitted.

## Scenario 3: Delta After Commit Drift

Given:

- post-commit changed HEAD,
- session continues.

When:

- next policy evaluation runs.

Then:

- `generate_mode=delta`,
- `changed_files` populated,
- context marked fresh after write.

## Scenario 4: Fresh-Session Recommendation

Given:

- turn count >= threshold,
- and/or duration >= threshold,
- cooldown window expired.

When:

- policy evaluation runs.

Then:

- `recommend_fresh_session=true`,
- reason codes included,
- recommendation text includes concise rationale.

## Scenario 5: Recommendation Cooldown

Given:

- recommendation emitted recently,
- cooldown not expired.

When:

- policy evaluation runs again.

Then:

- no duplicate recommendation emitted.

## Scenario 6: Missing Index Fallback

Given:

- index unavailable or stale.

When:

- `pci-context` runs.

Then:

- command exits non-fatally,
- minimal payload generated with warnings,
- adapters continue workflow without blocking.

## Scenario 7: Deterministic Truncation

Given:

- ranked payload exceeds token budget.

When:

- output renderer runs.

Then:

- truncation preserves deterministic ordering,
- highest-ranked entries retained first,
- output remains valid schema.

## Scenario 8: Local-Only Artifact Hygiene

Given:

- context artifacts written.

When:

- repo hygiene check runs.

Then:

- `.pci/context/` is recommended for ignore,
- no credentials present in artifacts.

## Golden Test Fixtures

Maintain fixtures under `tests/fixtures/context_orchestrator/`:

1. `bootstrap_latest.json`
2. `delta_latest.json`
3. `handoff_latest.json`
4. `state_after_recommendation.json`

## Acceptance Gate

All scenarios above must pass before enabling default integrations for a client.
