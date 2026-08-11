# Verified duplications (seed list, 2026-08-11)

Seventeen duplications in this repository, labeled from source during the
2026-08-11 measurement cycle (nine under blind protocol). These are verdicts,
not candidates: each was read in source and judged worth collapsing. The list
serves two purposes:

1. a one-time seeded cleanup list for `/pci-audit`;
2. a regression set for `find_redundancy` changes — a change that stops
   surfacing these groups is a regression, whatever its scores say.

Line numbers are as of commit `81370a4`; re-locate by symbol. Strike items
through as they are fixed; do not delete them (the regression role needs the
history).

| # | duplication | where |
|---|-------------|-------|
| 1 | `verbose_record`/`verbose_file` — byte-identical bodies | `mcp/formatting.py` |
| 2 | `parse_embedding_response` reimplements the parse `parse_embedding_items` owns | `embedding/bench.py` vs `embedding/endpoint.py` |
| 3 | `hooks/install._as_object` == `console_ui.as_object` exactly | `hooks/install.py`, `console_ui.py` |
| 4 | `hooks/install._as_list` ~= `console_ui.as_list` (failure value differs) | `hooks/install.py`, `console_ui.py` |
| 5 | LIKE-escape chain x3 (`source_path_*_pattern`) | `mcp/filters.py` |
| 6 | `_row_str`/`_row_text` — differ only on empty-string handling | `mcp/formatting.py` |
| 7 | `table_exists`/`table_regclass_exists`/`code_intel_tables_exist` — same `to_regclass` query x3 | `doctor/common.py`, `mcp/db.py` |
| 8 | `_format_seconds` / `format_duration` — two duration formatters, one a superset | `progress.py`, `runtime.py` |
| 9 | `_postgres_admin_check_settings` / `postgres_admin_target_fallback_settings` — same credential check | `doctor/database.py`, `ingest_code_intel.py` |
| 10 | 4 semantic penalty fns — identical modulo returned constant | `mcp/semantic.py:240-267` |
| 11 | `endpoint_is_remote` duplicated verbatim | `doctor/embeddings.py`, `embedding/framework.py` |
| 12 | HMAC password-derivation core x2 — crypto deserves one home | `db.py:254-268` |
| 13 | repo-root inference x2 — drift here would be a bug | `cli.py:162-177` |
| 14 | DSN split/rebuild frame x4 | `config.py:340-368` |
| 15 | `bounded_brace_body` / `bounded_brace_body_from_open` — incl. shared `-38` truncation magic | `parsers/core.py`, `parsers/javascript.py` |
| 16 | repo-path inject idiom x3 (`_inject_*`) | `mcp/formatting.py` |
| 17 | `js_symbol_records` reimplements `make_symbol_chunk` — record_id/title formats must stay in sync | `parsers/javascript.py`, `parsers/core.py` |

Fixed since labeling:

- (2026-08-11) `evidence._select_snapshots` == `analyze.select_snapshots`
  (not in the 17; caught in the same cycle, folded into `analyze`).
- (2026-08-11) `_coerce_str`/`_coerce_int` x3 (`analyze.py`, `evidence.py`,
  `context.py`) — `analyze.py` copy promoted to public `coerce_str`/`coerce_int`;
  `evidence.py` now imports it. `context.py` copy remains (no shared import
  path worth adding for one caller).

## Measurement context

Full record: `docs/feasibility-structural-compression.md` and git log
(`21f4c82`, `052a303`, `81370a4`). Precision of `find_redundancy` groups:
~42% real (n=40 labeled, 26 blind). Only an exact-text pair (max pairwise
text similarity >= 0.99) is near-certain; the blind set's highest-scoring
NON-duplicate pair scored 0.9876. Coherence does not separate real from junk
below that gate; `recommendation` runs at base rate. Labeled artifacts:
`~/pci-measurement-harness/blind-validation-2026-08-11/`.
