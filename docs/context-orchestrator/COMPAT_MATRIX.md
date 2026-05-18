# Context Orchestrator: Client Compatibility Matrix

## Purpose

Track integration maturity per client and avoid assumptions about hook/plugin
parity.

| Client | Integration Mode | Hook/Plugin Support | Phase | Status | Notes |
|---|---|---|---|---|---|
| Claude Code | Native hooks | Yes | 1 | Planned | Primary target for first adapter. |
| OpenCode | Plugin + events | Yes | 1 | Planned | Primary target for first adapter. |
| Codex CLI | TBD (native if available, else sidecar) | Unknown/validate per release | 2 | Backlog | Do not assume lifecycle hook parity until validated. |
| VS Code Copilot | IDE/CLI hooks + extension path | Partial/mixed | 2 | Backlog | Treat as separate integration effort. |
| Zed | Extension/tasks/slash command path | Partial/mixed | 2 | Backlog | Likely extension-driven insertion flow. |

## Decision Rule

For each client:

1. Prefer native hook/plugin integration.
2. Fall back to sidecar artifact mode when native lifecycle support is missing.
3. Keep manual `pci-context` commands as universal fallback.
