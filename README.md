# Project Code Intelligence

Hardware-Accelerated Codebase Mapping

`project-code-intelligence` indexes a Git repository into Postgres/pgvector and
serves the result through a small stdio MCP server.

The goal is simple: give coding agents and developers a searchable map of a
codebase without baking one project's assumptions into the tool.

It can store:

- repository snapshots and file inventory
- functions, classes, symbols, docs, config, and other code records
- candidate relationships between records
- SARIF/static-analysis findings and code-flow steps
- semantic embeddings for similarity search

The package is generic by default. Project-specific behavior belongs in code
profiles, with [`example.py`](src/project_code_intelligence/code_profiles/example.py)
as the public example.

## Quick Start

Use the checkout scripts directly, or install the package into your active
Python environment.

```sh
cd /path/to/project-code-intelligence
uv sync --extra dev
export PATH="$PWD:$PATH"
pci-doctor --skip-db --embedding skip
```

The first `pci-doctor` run prints startup commands that fit the current
machine. Run one of the commands from its `Available startup commands` section,
then verify the chosen services:

```sh
pci-doctor --embedding required
```

Text-only indexing is available as a fallback for bootstrap, debugging, or
privacy-sensitive environments. In that case, choose the Postgres-only command
and verify with `pci-doctor --embedding skip`.

Then index a Git repository:

```sh
cd /path/to/repo-to-index
pci-index --dry-run
pci-index
pci-mcp-smoke
```

For that fallback text-only mode, run `pci-index --no-embed`.

In a brand-new local repository, make an initial commit before scanning so the
indexer has a Git `HEAD` snapshot.

## Installation

For development:

```sh
uv sync --extra dev
```

For use from another repository:

```sh
uv pip install -e /path/to/project-code-intelligence
```

Without `uv`:

```sh
python -m pip install -e /path/to/project-code-intelligence
```

The installed console scripts are:

- `pci-index`
- `pci-doctor`
- `pci-mcp`
- `pci-mcp-smoke`
- `pci-embedding-bench`
- `pci-embedding-server`

## MCP Setup

Point Codex, Claude Desktop, or another MCP client at `pci-mcp`:

```json
{
  "mcpServers": {
    "project-code-intelligence": {
      "command": "/path/to/project-code-intelligence/pci-mcp"
    }
  }
}
```

The default database settings match the local Docker Compose database. Set
`PGVECTOR_*` only when using a different Postgres/pgvector instance.

For agent-heavy workflows, copy
[`docs/examples/AGENTS.md`](docs/examples/AGENTS.md) into the repository being
indexed so coding assistants know when to use the MCP index.

## Embeddings

Embeddings are the expected path for normal use. They are what make the MCP
index useful for semantic search instead of only exact text lookup.

Common paths are CPU FastEmbed, AMD Ryzen AI NPU, AMD GPU, NVIDIA GPU, and
remote OpenAI-compatible providers. `pci-doctor` prints the exact startup
commands that are available on the current machine.

Run `pci-doctor` to see which paths are available on the current machine:

```sh
pci-doctor --embedding required
```

`pci-index` itself does not download models. The Docker Compose embedding
profiles may download models into Docker volumes or ignored local paths.

Remote embedding endpoints receive source-derived text. For private code, use a
local endpoint or a provider you trust, and set
`PROJECT_CODE_INTELLIGENCE_ALLOW_REMOTE_EMBEDDING=1` only intentionally.

## Docker Compose Profiles

Profiles are runtime choices, not project modes:

| Profile | Use when |
| --- | --- |
| none | Postgres/pgvector only, for text search or an external embedding provider. |
| `cpu` | Portable local semantic-search demo with FastEmbed. |
| `npu` | Experimental AMD Ryzen AI/XDNA NPU embeddings. |
| `amdgpu` | Experimental AMD ROCm llama.cpp embeddings. |
| `nvidia` | Experimental NVIDIA CUDA llama.cpp embeddings. |

List the profiles with:

```sh
docker compose config --profiles
```

Most users should start with `cpu`, then let `pci-doctor` suggest hardware
specific commands if local acceleration is available.

## Docker Lifecycle

Use `up -d` to start the profile suggested by `pci-doctor`. Use `stop` when you
want to pause containers but keep them around:

```sh
docker compose stop
```

Use `down` for normal cleanup. This removes containers and the Compose network
while keeping the local database and downloaded model caches:

```sh
docker compose down
```

Use `down -v` only when you intentionally want a fresh database and fresh
Docker-managed model caches:

```sh
docker compose down -v
```

That deletes the named volumes for Postgres, FastEmbed, Lemonade, and ROCm
runtime caches. It does not delete the bind-mounted `./models` directory used by
the GPU profiles.

On Apple Silicon, Docker Compose is still useful for Postgres/pgvector. Local
Apple GPU embeddings should run on the macOS host, not inside Docker.

## What the MCP Server Provides

The server exposes tools for:

- checking indexed snapshot and embedding status
- text and semantic search over indexed records
- fetching individual records
- following candidate relationships
- searching SARIF/static-analysis findings
- fetching CodeQL/SARIF code-flow steps

The MCP server runs over stdio. Docker Compose is used for local dependencies,
not for wrapping the MCP process.

## Project Profiles

The generic profile covers common source, docs, build files, config files, and
SARIF input. A project can add its own profile for domain-specific file roles,
metadata, records, or security context.

Private profiles do not need to be registered in this package. Put them on
`PYTHONPATH` and select them with a fully qualified profile path:

```sh
PROJECT_CODE_INTELLIGENCE_PROFILE=my_project.code_profile:MyProjectProfile pci-index
```

Profiles are ordinary Python code, so load them only from trusted local modules.

## Development

Run the local quality gate:

```sh
make check
```

Run the integration smoke against a running Compose database:

```sh
docker compose up -d pgvector
make integration-smoke
```

Useful docs:

- [CONTRIBUTING.md](CONTRIBUTING.md): contributor workflow and guardrails
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md): local CPU/NPU/GPU benchmark notes
- [.env.example](.env.example): available environment variables
- [AGENTS.md](AGENTS.md): instructions for assistants working on this repo

## Privacy

Do not publish database dumps, restore artifacts, SARIF output, embedding
caches, model files, vector indexes, local MCP configs, or generated data from
private repositories. These can contain source snippets, internal paths,
symbols, findings, metadata, and embeddings derived from source text.

## License

MIT. See [LICENSE](LICENSE).
