# Agent Instructions

This repository builds `project-code-intelligence`: a generic code indexer plus
stdio MCP server backed by Postgres/pgvector.

## Project Intent

- Keep the default package generic. Project-specific behavior belongs in code
  profiles, not in shared ingest or MCP code.
- Keep the public example profile in
  `src/project_code_intelligence/code_profiles/example.py`.
- Do not publish or register private project profiles.
- Preserve the stdio MCP deployment path. Docker Compose is for local
  dependencies such as pgvector and optional embedding servers.

## Development Checks

Run the local quality gate after Python code changes:

```sh
make check
```

When changing ingest, database, or MCP behavior, also run the integration smoke
against a running Compose database:

```sh
docker compose up -d pgvector
make integration-smoke
```

Use `make format` for Ruff formatting.

## Architecture Preferences

- Prefer small modules with clear responsibilities over large catch-all files.
- Keep configuration environment-first and parse it through
  `project_code_intelligence.config`.
- Keep MCP tools generic and filter-oriented. Avoid hard-coding assumptions from
  one downstream repository.
- Treat schema, ingest output, MCP responses, and privacy-sensitive behavior as
  compatibility surfaces.
- Add focused tests for parser behavior, SARIF normalization, config parsing,
  database write behavior, and MCP tool responses when those areas change.
- Do not weaken Ruff, basedpyright, Bandit, shell, coverage, or test rules just
  to make a warning disappear. Prefer fixing the underlying design or code. Only
  relax a rule when the exception is architecturally or logically justified for
  this project, and keep such exceptions narrow and documented.

## Privacy And Publication

Do not commit database dumps, restore artifacts, SARIF output, embedding caches,
model files, vector indexes, local MCP configs, or generated data from private
repositories. These can contain source snippets, internal paths, symbols,
findings, metadata, and embeddings derived from source text.

Give a confidence level when recommending an approach, and push back when a
suggestion would make the project less generic, less publishable, or
unnecessarily complex.
