# Context Orchestrator: Implementation Roadmap

## Milestone 1: Core Context Generator

Deliver:

- `pci-context` command,
- markdown/json renderers,
- artifact writer,
- deterministic budget truncation.

Exit criteria:

- Behavior scenarios 1, 2, 6, 7, 8 pass.

## Milestone 2: Shared Policy Engine

Deliver:

- threshold/cooldown logic,
- recommendation reason codes,
- state persistence.

Exit criteria:

- Behavior scenarios 4 and 5 pass.

## Milestone 3: Claude Adapter

Deliver:

- hook integration wiring,
- session start + drift refresh + recommendation emission.

Exit criteria:

- Behavior scenarios 1-5 pass in Claude integration tests.

## Milestone 4: OpenCode Adapter

Deliver:

- plugin event wiring,
- session lifecycle integration and recommendation emission.

Exit criteria:

- Behavior scenarios 1-5 pass in OpenCode integration tests.

## Milestone 5: Telemetry + Evaluation

Deliver:

- local metrics capture,
- before/after comparison report format.

Exit criteria:

- measurable trial data for adoption decision.

## Deferred Milestones

- Codex integration exploration.
- VS Code/Copilot integration.
- Zed integration.
