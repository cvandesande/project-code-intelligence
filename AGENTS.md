# Agent Instructions

This repo has a project-code-intelligence MCP server. For non-trivial code
discovery, use it before broad rg/find or speculative file reads. Use rg/direct
reads for known small files and final verification.

Before you remove or rename a definition (function, class, method), or change
one's signature, call the `blast_radius` MCP tool for that symbol first. All
three break callers the same way, and the tool answers all three with one
query. Treat its callers, test coverage, and entry-point/orphan flags as
evidence (not a verdict), and verify live callers in source before finalizing
the change. The evidence hook injects the same bundle on an edit-delete as a
backstop, but pulling it up front is cheaper than undoing a bad removal.

To find duplication in the code already indexed, `find_redundancy` reports
groups of functions that repeat one call-shape, ranked by whether collapsing
them is worth it. For a whole-tree sweep (staleness, duplicate
names, redundancy candidates, static findings), run `pci-analyze audit` from
the repo root; measured precisions are printed in the report, and
`docs/verified-duplications-2026-08-11.md` carries the source-verified seed
list.

This repository builds `project-code-intelligence`: an MCP server that gives
coding agents structured access to indexed Git repositories — semantic
search, lexical search, a candidate-edge graph, and static-analysis findings
— so agents can find context without speculative file reads. Indexed data
lives in Postgres/pgvector.

## Project Intent

- Keep the default package generic. Project-specific behaviour belongs in
  code profiles, not in shared ingest or MCP code.
- Do not publish or register private project profiles.
- Preserve the stdio MCP deployment path. Docker Compose runs local
  pgvector and optional embedding backends; it is not a deployment vehicle
  for the MCP server itself.
- Give a confidence level when recommending an approach, and push back
  when a suggestion would make the project less generic, less publishable,
  or unnecessarily complex.

## Architecture Map

- `mcp/` — stdio MCP server. `transport.py` handles JSON-RPC,
  `tool_catalog.py` declares schemas, `tool_inputs.py` validates them,
  `tools.py` implements handlers, `filters.py` builds SQL clauses,
  `protocol.py` is wire helpers.
- `parsers/` — language-agnostic chunkers plus project-specific record
  parsers (Kconfig, Makefile, DTS, shell, …); `registry.py` maps
  language → parser.
- `language_profiles/` — per-language metadata extractors (`go_functions`,
  `doc_links`, …). Additive to chunked content. Not the same as code
  profiles.
- `code_profiles/` — repository-level profile hooks. `example.py` is the
  public starting point; `registry.py` handles registration.
- `sarif/` — SARIF ingest, normalization, and rendering for the
  `*_static_*` MCP tools.
- `embedding/` — embedding backends (Apple MLX, llama.cpp CUDA/ROCm,
  fastembed, lemonade NPU) and the pre-embedding pipeline.
- `storage/` — Postgres schema, copy-based inserts, static-finding storage.
- `doctor/` — hardware detection and local-service orchestration; see
  Development Checks for invocation.

## Conventions

- Environment-first config, parsed through
  `project_code_intelligence.config`.
- MCP tools are filter-oriented; let callers compose. Keep tool and
  property descriptions terse — they load via `tools/list` on every client
  session and consume agent context on every turn. Prefer per-property
  `description` fields over prose in the tool description, and factor
  recurring text into shared constants (see `_SOURCE_PATH_DESC` etc. in
  `tool_catalog.py`).
- Compatibility surfaces with authoritative docs and tests:
  - `docs/PUBLIC_API.md` — MCP wire contract.
  - `tests/test_mcp_contracts.py` — MCP behaviour tests; add one here when
    MCP responses or input validation change.
  - `schema.sql` — Postgres schema; treat migrations as compatibility
    events.
- Pattern matching (`match`): use where it does structural work — nested
  dict destructuring (e.g. `case {"k": {"inner": str() as x}}:`),
  closed-union or AST type dispatch, exception `isinstance` alternation,
  type-coercion chains where bool-before-int ordering matters. Skip
  single-shot `if isinstance(x, T) and <cond>: <use>` guards: basedpyright
  strict requires `case _: pass`, so the rewrite is strictly longer with
  no clarity gain. Always use class patterns with `()` (`bool()`, `str()`,
  `dict()`) — bare names bind, they don't compare to outer scope. For
  constants in patterns, use dotted names or `if` guards.
- Don't weaken any of the checks `make check` runs. Fix the underlying
  issue, or narrowly justify a documented exception.

## Development Checks

`make check` runs the full local gate: ruff format-check, shell
format-check, ruff lint, shellcheck, unit tests, basedpyright, Bandit,
pip-audit, and compose-check (which validates `docker-compose.yml` and
keeps the bundled copy in `src/project_code_intelligence/` in sync with
the repo root). Use `make format` for Ruff auto-format.

When changing ingest, database, or MCP behaviour, also run the
integration smoke against a running Compose database:

```sh
docker compose up -d pgvector
make integration-smoke
```

For local services: `pci-doctor` reports hardware and what's running;
`pci-doctor --start` brings up the DB and the best embedding backend for
the host; `pci-doctor --stop` tears everything down.

## Privacy And Publication

Do not commit database dumps, restore artifacts, SARIF output, embedding
caches, model files, vector indexes, local MCP configs, or generated data
from private repositories. These can carry source snippets, internal
paths, symbols, findings, metadata, and embeddings derived from source
text.
