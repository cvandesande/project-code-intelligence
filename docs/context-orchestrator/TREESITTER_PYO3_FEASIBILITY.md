# Tree-sitter + PyO3 Feasibility Guide

## Purpose

Describe a practical path for integrating Tree-sitter into
`project-code-intelligence` via Rust + PyO3, and evaluate whether keeping the
main project in Python is still justified given goals of cost savings and agent
execution efficiency.

## Executive Summary

- **Yes, this integration is achievable** with a phased adoption path.
- **No, a full rewrite is not currently justified by default** for the stated
  goals.
- The most effective near-term plan is **Python-first orchestration + optional
  Rust acceleration** for parse/index hot paths.

## Decision Framework

Your top goals are:

1. lower agent cost,
2. higher agent efficiency,
3. reliable adoption across clients.

Those goals do **not** automatically require a language rewrite. They require:

1. better context orchestration,
2. better retrieval quality,
3. lower latency where it matters.

A full rewrite increases delivery risk and migration cost before proving it
improves those metrics.

## Where Tree-sitter + PyO3 Helps Most

Tree-sitter can improve:

1. symbol extraction quality,
2. structural context maps,
3. deterministic syntax-aware chunking.

PyO3 enables exposing Rust parsing capabilities as a Python module so existing
CLI/MCP orchestration can stay intact.

## Where It Does Not Directly Help

Tree-sitter/PyO3 does not directly solve:

1. long-chat context bloat,
2. hook/plugin adoption behavior,
3. stale-session recommendation UX,
4. prompt/context policy quality.

Those are orchestration concerns and remain largely language-agnostic.

## Why Keeping Python Still Makes Sense (for now)

Keeping the project Python-centric is justified if you value:

1. fast iteration on CLI/MCP/tool schemas,
2. broad contributor accessibility,
3. existing integration surface stability,
4. current embedding/hardware orchestration flow (including Apple MLX pathways).

For your specific concern: Apple MLX support is currently part of the Python
runtime and packaging flow in this repository, which is a practical reason to
avoid a rushed full-language migration.

## When a Full Rewrite Becomes Justified

A rewrite should be considered only if measured evidence shows all of:

1. parsing/indexing is the dominant bottleneck,
2. Rust acceleration via extension modules is insufficient,
3. operational complexity of mixed Python+Rust exceeds rewrite cost,
4. migration can preserve MCP/API compatibility with acceptable effort.

## Recommended Architecture: Hybrid Model

Adopt a hybrid architecture:

1. **Python control plane**
   - CLI entrypoints
   - MCP server transport/tool contracts
   - context-orchestrator policy logic
   - hardware/runtime orchestration
2. **Rust data plane (optional modules)**
   - Tree-sitter parse/extract
   - high-volume structural indexing helpers

This yields most performance upside with lower migration risk.

## Integration Plan

### Phase A: Prototype module

Build a small PyO3 extension (for one or two languages) exposing:

- `parse_symbols(path, text) -> symbols`
- `extract_structure(path, text) -> records`

Measure against current parser outputs for quality and speed.

### Phase B: Feature parity track

Add adapters so parser selection is configurable:

- `parser_backend=python|rust_treesitter`

Run side-by-side in CI fixtures to compare output drift.

### Phase C: Optional default for selected languages

Promote Rust backend for languages where it clearly improves quality/latency.
Keep Python fallback for portability and debuggability.

### Phase D: Re-evaluate rewrite decision

Only after telemetry from real sessions indicates substantial gains and reduced
cost-per-task should a full rewrite be reconsidered.

## Build/Packaging Considerations

1. Use maturin/PyO3 for wheel builds.
2. Provide prebuilt wheels for major platforms to reduce user friction.
3. Keep pure-Python fallback path when extension install is unavailable.
4. Gate Rust backend behind a feature flag during rollout.

## Risk Assessment

### Risks of immediate rewrite

1. delivery slowdown,
2. compatibility breakage,
3. contributor/toolchain friction,
4. delayed impact on your actual top goals.

### Risks of hybrid approach

1. dual-language complexity,
2. potential output divergence,
3. CI/build matrix expansion.

### Mitigations

1. strict behavior fixtures,
2. backend selection toggles,
3. phased rollout by language,
4. compatibility tests on MCP/public API surfaces.

## Metrics To Decide Next Steps

Track before/after across representative tasks:

1. time to first useful map,
2. parser throughput and latency,
3. context artifact quality (symbol precision/coverage),
4. MCP follow-up calls per task,
5. total session token usage,
6. task completion quality and cycle time.

If hybrid gives clear wins on these metrics, keep hybrid. If not, defer or
re-scope Rust investment.

## Recommendation

For cost savings and agent efficiency goals, the best current strategy is:

1. keep the project primarily in Python,
2. implement context-orchestrator integrations first,
3. add Tree-sitter via PyO3 as an optimization module,
4. postpone full rewrite decision until benchmark and production telemetry exist.

This is the highest-probability path to near-term value with manageable risk.
