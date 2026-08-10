# Feasibility: structural compression and architectural simplification (issue #3)

Status: research / feasibility writeup. No code or schema changes.

This document records the feasibility of GitHub issue #3 ("structural code
compression and architectural simplification analysis") against the current
PCI code. It states a recommended first target and gives file-level
integration points. Confidence levels are stated inline.

## Summary / verdict

The issue's core premise is correct: the record / edge / embedding /
snapshot / MCP substrate already exists and fits the *orchestration and
analysis* layers well (Phases 1, 3, 5, and the MCP tools). Confidence: high.

Cost and risk are not spread evenly across the six phases. They are
concentrated in three places:

1. The Rust semantic extractor (Phase 2) — the biggest net-new build, and
   the one part that sits outside this repo's Python / Postgres comfort zone.
   Confidence: medium.
2. Motif detection (Phase 4) — approximate subgraph matching. Nothing like
   it exists today. Confidence: medium.
3. Scaling — there is no ANN vector index, and graph traversal is
   single-hop only. Confidence: high.

Recommended first target (chosen): an **analysis-first prototype on the
existing heuristic graph plus embeddings**, with no Rust. This proves value
cheaply before the expensive, higher-risk extractor work.

## Current substrate map

What already exists, with references.

- Code records / chunks: `IntelRecord` (`src/project_code_intelligence/models.py:308-337`);
  table `project_code_intel_records` (`src/project_code_intelligence/schema.sql:51-106`).
  Rich metadata: `record_type`, `symbol`, `symbol_kind`, `line_start/line_end`,
  provenance fields, `embedding_text` vs `display_content`, and a stored
  weighted `search_document` tsvector.
- Edges / relationships: `IntelEdge` (`models.py:340-350`); table
  `project_code_intel_edges` (`schema.sql:108-125`). `edge_type` is free text,
  so new fact types need no migration. `source_record_id` / `target_record_id`
  are stable string IDs, not foreign keys; `target_record_id IS NULL` marks an
  unresolved candidate.
- Provenance / confidence: closed `confidence_kind` vocabulary
  (`src/project_code_intelligence/mcp/taxonomies.py:44-49`) plus per-edge and
  per-record `metadata jsonb`. Records carry a numeric `confidence`
  (`schema.sql:83`); edges do **not**.
- Edge types produced today: `call_candidate` (heuristic; e.g.
  `parsers/core.py:207-217`, `parsers/cfamily.py:229-236`,
  `parsers/javascript.py:438-449`) and `include` (high-confidence C includes,
  `parsers/cfamily.py:170-179`). Candidate resolution:
  `storage/core.py:561-594` (in-memory) and `storage/core.py:615-734` (SQL
  name-binding against `symbol_definition` records).
- Embeddings / pgvector: dimensionless `embedding vector` column
  (`schema.sql:93`); dimensions and model pinned at runtime by an "embedding
  contract" in snapshot metadata (`embedding/store.py:47-98`); cosine distance
  via `<=>` in the semantic-search handler (`mcp/tools.py:296-386`, distance at
  line 307). There is **no ivfflat/hnsw index** — similarity is a sequential
  scan.
- Graph traversal: single-hop only. `related_code_intel` joins edges to
  records twice (`mcp/tools.py:439-459`); direction and base filters in
  `mcp/related.py:50-243`. There is no recursive / multi-hop CTE.
- External-tool ingestion precedent: SARIF. A separate tool emits JSON, PCI
  ingests it to `IntelRecord` without parsing source text
  (`sarif/ingest.py:206-226`; wired into the driver at
  `ingest_code_intel.py:1506-1563`). This is the template for any future
  external extractor. Note: there is **no generic node/edge JSON loader**
  today — SARIF is finding-shaped.
- MCP tool recipe: four files — model (`mcp/tool_inputs.py`), schema
  (`mcp/tool_catalog.py`), handler `def tool_x(args) -> Json` returning `ok()`
  (`mcp/tools.py`), and a contract test (`tests/test_mcp_contracts.py`).
  Dispatch is registry-driven; `transport.py` needs no change. Write tools are
  hidden when writes are disabled (`mcp/tools.py:859-870`).
- CLI: argparse. `main()` already dispatches one subcommand
  (`cli.py:582-586`); `pci-context` (`context.py`) is the read-only DB-reader
  precedent. Console scripts live in `pyproject.toml`.

## Phase-by-phase feasibility

### Phase 1 — richer graph model. Fit: strong. Confidence: high.

New edge / fact types (`implements_trait`, `uses_type`, `returns_type`, …)
need no migration because `edge_type` is free text. Provenance goes into
`metadata jsonb`. A precise extractor can set `target_record_id` directly and
skip heuristic resolution.

Gaps to decide on:
- Edges have no numeric `confidence` column (records do).
- No dedicated provenance column; provenance lives in `metadata`. Adding a
  column is a schema / compatibility event (see AGENTS.md).
- Graph-neighborhood queries are single-hop; multi-hop traversal for analysis
  is net-new SQL.

### Phase 2 — Rust extractor. Fit: additive, highest risk. Confidence: medium.

The SARIF path proves the ingestion pattern. But there is no generic JSON
node/edge loader yet, so a new ingest module is required. The real risk is
external: rust-analyzer exposes no stable public library API; integration is
via LSP or the unstable `ra_ap_*` crates, or rustc internals — substantial
standalone Rust work against a moving target. This is the least generic /
publishable part and should stay optional and out of shared ingest, matching
the issue's own non-goals and AGENTS.md.

### Phase 3 — similarity analysis. Fit: good, with a scaling caveat. Confidence: medium-high.

Embeddings and cosine distance exist; textual similarity via the stored
tsvector exists. Caveat: no ANN index means "compare all functions" is O(n^2)
sequential scans — fine at small scale, a wall at large scale. Call-subgraph
and type / control-flow similarity need a new per-function neighborhood
builder; control flow is not captured at all today, so control-flow and
type-flow signals are extractor-dependent (Phase 2).

### Phase 4 — motif detection. Fit: net-new algorithmic work. Confidence: medium.

Postgres is not a graph engine, so approximate subgraph matching plus
identifier-to-role normalization would run Python-side. This is the
algorithmic core and the least de-risked part.

### Phase 5 — compression / MDL scoring. Fit: feasible as advisory. Confidence: medium.

The four-file MCP recipe is clean and read-only-friendly. "Abstraction cost"
estimation is inherently fuzzy; keep the output advisory, as the issue states.

### Phase 6 — visualization. Fit: out of first scope; model supports it. Confidence: high.

## Risks and limits

- No ANN vector index: semantic similarity is a sequential cosine scan
  (`mcp/tools.py:296-386`). Pairwise similarity does not scale without an
  index.
- Single-hop traversal only (`mcp/tools.py:439-459`): analysis passes needing
  multi-hop neighborhoods require new SQL.
- Edges carry only categorical `confidence_kind`, no numeric `confidence`.
- Provenance lives in `metadata jsonb`; a dedicated column is a
  schema / compatibility event.
- rust-analyzer has no stable library API — the one real external unknown.

## Recommended sequencing (with de-risking gates)

- Gate A — analysis-first prototype on the existing heuristic `call_candidate`
  graph plus embeddings. No Rust. Prove that graph-structural analysis
  produces useful, non-obvious results.
- Gate B — build a generic, language-agnostic `pci-ingest-graph` JSON facts
  loader. More generic and publishable than a Rust-coupled path.
- Gate C — add the Rust extractor as the first client of that loader, only if
  Gate A shows value.

Rationale (confidence: high): the issue orders the Rust extractor before
similarity / motif analysis. Inverting that order tests the cheap, portable
hypothesis first. If the analysis adds no value on the heuristic graph, the
extractor investment is not justified.

## Analysis-first prototype design (the chosen first target)

### Inputs available today (no Rust)

- Resolved `call_candidate` edges (those with `target_record_id` set).
- `symbol_definition` records (functions / methods / classes) and their line
  ranges and `symbol_kind`.
- `records.embedding` (cosine via `<=>`).
- The weighted `search_document` tsvector for textual similarity.

### Per-function neighborhood representation

For each `symbol_definition` record, build a bounded neighborhood: the set and
the ordered sequence of resolved callees (target symbol plus `symbol_kind`),
to a small fixed radius. This is the "call-subgraph" for that function.

### Similarity signals usable now

- Semantic embedding similarity (record embedding cosine).
- Call-subgraph similarity (Jaccard over callee sets; sequence edit distance
  over callee order).
- Textual similarity (tsvector / trigram over `display_content`).
- Name-shape similarity (normalized symbol names).

Explicitly NOT available without Phase 2, and marked as extractor-dependent:
type-flow similarity, control-flow similarity, trait / implementation
similarity.

### Motif detection sketch

1. Normalize concrete identifiers and types into structural roles (for
   example `UserRequest -> Request<T>`, `create_user -> create_<T>`).
2. Hash normalized neighborhoods with a Weisfeiler-Lehman-style label hash to
   produce a structural fingerprint per function.
3. Cluster functions by fingerprint proximity.
4. Rank groups by group size times structural agreement, penalized by an
   estimated abstraction cost (MDL-flavored, advisory only).

### Delivery surface for the prototype

Read-only `pci-analyze compression` CLI. It reads the DB the same way
`pci-context` does, so no write path is touched. Insertion points:

- Subcommand dispatch: extend `cli.py:582-586` (the existing `main()`
  dispatcher) with an `analyze` branch, mirroring the `mcp-smoke` pattern.
- Console script: add `pci-analyze` to `[project.scripts]` in `pyproject.toml`.

MCP tools (`find_structurally_similar_code`, `find_repeated_code_motifs`,
`explain_structural_similarity`, `find_compression_candidates`,
`get_code_subgraph`) come later via the four-file recipe, once the analysis
produces stable data. CLI first is the cheapest way to reach Gate A.

### Gate A success criterion

The prototype produces non-obvious, source-verifiable candidate groups when
run against this repo's own index. If the only results are trivially textual
duplicates that ordinary clone tools already find, Gate A fails and the
extractor work is not justified.

## Open decisions

- Add a numeric `confidence` column to edges, or keep only `confidence_kind`?
- Keep provenance in `metadata jsonb`, or add a dedicated column (schema /
  compatibility event)?
- Add an ANN vector index (ivfflat / hnsw) before Phase 3 scales?
- Add a multi-hop CTE neighborhood query, or build neighborhoods Python-side
  from single-hop reads?

## Non-goals / guardrails (from the issue)

- Do not optimize for LOC alone.
- Similar graphs do not imply code should be merged.
- Do not replace source verification with similarity scores.
- Do not require compiler-native extraction for every language.
- The Rust extractor is not responsible for PCI persistence, embeddings, MCP,
  or orchestration.
- Preserve provenance / confidence so heuristic and compiler-resolved facts
  cannot be confused.
