# Contributing

This project is intended to stay small enough for a developer to understand from
the source tree, while still having enough checks to be safe to publish and
extend.

## Local Setup

Use any supported Python 3.10+ environment:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

If you use `uv`, the equivalent is:

```sh
uv sync
```

Dev tools (ruff, basedpyright, coverage, bandit) are in `[dependency-groups]`
and included by `uv sync` automatically.

If you need to test the installed CLI commands from other repositories while
editing this checkout, install the tool in editable mode:

```sh
uv tool install --editable /path/to/project-code-intelligence
```

Editable tool installs are for development. Reinstall or uninstall the tool when
you want to test the packaged, non-editable install path.

Start the local database when you want to run the integration smoke:

```sh
docker compose up -d --wait --wait-timeout 60 pgvector
```

## Checks

Run the local quality gate before publishing changes:

```sh
make check
```

That runs Ruff formatting checks, Ruff linting, unit tests, basedpyright, and
Bandit. Apply Ruff formatting with:

```sh
make format
```

Run the end-to-end database and MCP smoke with:

```sh
make integration-smoke
```

The integration smoke creates a temporary Git repository, indexes it, reruns in
incremental mode, calls the MCP status tool, and verifies text search through
the MCP server.

## Architecture Guardrails

- Keep default behavior generic. Project-specific behavior belongs in an
  explicit code profile.
- Do not add private profiles to the public registry. Use
  `src/project_code_intelligence/code_profiles/example.py` as the public example.
- Keep configuration environment-first so nightly jobs, local shells, and MCP
  clients can use the same code paths.
- Prefer small modules with clear responsibilities over cross-cutting helper
  layers.
- Add checks when a change affects ingest behavior, database writes, MCP tool
  responses, or privacy-sensitive output.

## Privacy

Do not commit database dumps, restore artifacts, SARIF output, embedding caches,
model files, or vector indexes produced from private repositories. These can
contain source snippets, internal paths, symbols, findings, metadata, and
embeddings derived from source text.
