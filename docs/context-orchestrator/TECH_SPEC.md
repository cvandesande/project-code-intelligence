# Context Orchestrator: Technical Specification

## Status

Draft v1.

## Goal

Deliver a plugin/hook-first context orchestration layer that improves assistant
first-turn orientation and reduces long-session cost drift without requiring
users to replace provider-owned CLIs.

## Scope (Phase 1)

Targets:

1. Claude Code hooks integration.
2. OpenCode plugin/hooks integration.

Out of scope for Phase 1:

1. Codex-specific integration.
2. VS Code/Copilot-specific integration.

## Architecture

The system has four modules:

1. **Map Generator** (`pci-context`): builds compact bootstrap/delta context.
2. **Policy Engine**: computes freshness and long-session recommendation state.
3. **Client Adapter**: per-client hook/plugin glue for Claude/OpenCode.
4. **Artifact Store**: local files under `.pci/context/` for transport.

## Artifact Contracts

### Paths

- `.pci/context/latest.md`
- `.pci/context/latest.json`
- `.pci/context/state.json`

These files are local-only operational artifacts and must not be committed.

### `latest.json` schema (v1)

```json
{
  "version": "v1",
  "generated_at": "2026-01-01T00:00:00Z",
  "repo_root": "/abs/path",
  "mode": "bootstrap|delta|handoff",
  "head_commit": "<sha>",
  "snapshot_id": 123,
  "token_budget": 1000,
  "summary": ["..."],
  "key_files": [{"path": "src/x.py", "why": "..."}],
  "key_symbols": [{"symbol": "Foo.bar", "path": "src/x.py"}],
  "changed_files": [{"path": "src/x.py", "reason": "..."}],
  "next_queries": ["search_code_intel_text query=..."],
  "stale": false,
  "warnings": []
}
```

### `state.json` schema (v1)

```json
{
  "version": "v1",
  "repo_root": "/abs/path",
  "last_head_commit": "<sha>",
  "last_generated_at": "2026-01-01T00:00:00Z",
  "last_mode": "bootstrap",
  "last_hash": "<sha256>",
  "turn_count": 0,
  "session_started_at": "2026-01-01T00:00:00Z",
  "last_recommendation_at": null,
  "context_dirty": false
}
```

## CLI Contract

### New command

`pci-context`

### Flags (Phase 1)

- `pci-context .`
- `pci-context . --tokens <int>`
- `pci-context . --topic "..."`
- `pci-context . --delta`
- `pci-context . --format markdown|json`
- `pci-context . --write-artifacts`

## Policy Engine Contract

### Inputs

- Session turn count.
- Session duration.
- Optional token usage estimates.
- Repo drift signal (HEAD changed since last generation).
- Manual task-complete hint from client adapter.

### Outputs

- `generate_mode`: `none|bootstrap|delta|handoff`
- `recommend_fresh_session`: `true|false`
- `reason_codes`: array (`turns_high`, `duration_high`, `drift_high`, etc.)

### Default thresholds (tunable)

- turn count >= 40
- session duration >= 75 minutes
- recommendation cooldown: 20 minutes

## Client Adapter Requirements

### Claude Adapter

- Use documented hook events to:
  - run bootstrap generation at session start,
  - run delta generation at task-complete milestones,
  - emit recommendation messages on threshold crossing.

### OpenCode Adapter

- Use plugin event subscriptions to:
  - run bootstrap generation on session creation,
  - refresh delta/handoff on session compaction/update milestones,
  - emit recommendation messages with cooldown enforcement.

## Failure Handling

1. Missing/stale index: generate minimal context with actionable warning.
2. Command failure: do not block user workflow; log and continue.
3. Oversized context: truncate deterministically by ranking priority.

## Security/Privacy

1. Artifacts remain local.
2. Never write credentials to context artifacts.
3. Recommend ignore entries for `.pci/context/`.

## Deliverables

1. `pci-context` command + tests.
2. Shared policy engine module + tests.
3. Claude adapter reference integration.
4. OpenCode adapter reference integration.
5. Compatibility matrix update.
