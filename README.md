# Project Code Intelligence

## Hardware-Accelerated Codebase Mapping

`project-code-intelligence` indexes a Git repository into Postgres/pgvector and
serves the result through a small stdio MCP server.

The goal is higher-quality agent results: reuse a local code index instead of
re-reading the same repository over and over, reducing token and embedding cost
while making codebase navigation faster.

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

Install the CLI tools once, then use them from any repository:

```sh
uv tool install --editable /path/to/project-code-intelligence
pci-doctor --skip-db --embedding skip
```

On macOS Apple Silicon this automatically installs Core ML dependencies.
`pci-doctor` detects the Apple Neural Engine and available acceleration paths.

The first `pci-doctor` run prints startup commands that fit the current
machine. For a fully local setup, start `pgvector` plus one embedding service.
If you already have an external Postgres/pgvector database, skip `pgvector` and
start only the embedding service you want to use. Then verify the chosen
services:

```sh
pci-doctor --embedding required
```

Text-only indexing is available as a fallback for bootstrap, debugging, or
privacy-sensitive environments. In that case, choose the Postgres-only command
and verify with `pci-doctor --embedding skip`.

Then index a Git repository. Use `.` when you mean the current directory:

```sh
cd /path/to/repo-to-index
pci-index . --dry-run
pci-index .
pci-mcp-smoke
```

You can also index one or more repositories without changing directories:

```sh
pci-index /path/to/repo-to-index
pci-index /path/to/repo-a /path/to/repo-b
```

For advanced ingest options, put them after `--`:

```sh
pci-index /path/to/repo-to-index -- --limit-files 100
```

If indexing is interrupted, rerun the same command. `pci-index .` reuses the
same snapshot when the Git tree is unchanged, keeps compatible existing
embeddings, and fills in records that are still missing embeddings. In normal
incremental mode it only reparses changed files.

For that fallback text-only mode, run `pci-index . --no-embed`.

To wipe and rebuild the code-intelligence tables in the configured database:

```sh
pci-index --reset-code-intel
```

This drops and recreates this project's `project_code_intel_*` tables. It does
not drop the database or unrelated tables. The command prints the resolved
database target, asks for confirmation before deleting anything, and exits
without scanning. Run `pci-index .` afterwards to rebuild the index. For
non-interactive automation, add `--i-know-this-deletes-code-intel-db`.

In a brand-new local repository, make an initial commit before scanning so the
indexer has a Git `HEAD` snapshot.

## Installation

Install the CLI tools system-wide with `uv`:

```sh
uv tool install --editable /path/to/project-code-intelligence
```

This places `pci-doctor`, `pci-index`, `pci-mcp`, and the other console scripts
on your PATH (usually `~/.local/bin`). On macOS Apple Silicon, Core ML
dependencies (coremltools, transformers, numpy, huggingface_hub) are installed
automatically via platform-conditional dependencies. No extra install step is
needed.

Make sure the tool path is on `PATH`:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

After that, run commands from any repository:

```sh
cd /path/to/repo-to-index
pci-index .
```

Without `uv`, create and activate a virtualenv first:

```sh
cd /path/to/project-code-intelligence
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

### Development setup

For contributing, use `uv sync` which creates a `.venv` in the project
directory with dev tools (ruff, basedpyright, coverage, bandit) included
automatically:

```sh
cd /path/to/project-code-intelligence
uv sync
```

The shell wrapper scripts in the repository root (`./pci-doctor`, `./pci-index`,
etc.) auto-detect `.venv/bin/python`, so `make` commands and direct
`./pci-doctor` invocations work without activating the virtualenv.

Do not mix `uv tool install` and `uv sync` for the same package. If you
previously ran `uv tool install`, remove it first with
`uv tool uninstall project-code-intelligence` before switching to `uv sync` for
development.

The installed console scripts are:

- `pci-index`
- `pci-doctor`
- `pci-mcp`
- `pci-mcp-smoke`
- `pci-coreml-server`
- `pci-embedding-bench`
- `pci-fastembed-server`
- `pci-llama-embed`

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

The default database settings match the local Docker Compose database. For a
different Postgres/pgvector instance, prefer one database URL:

```sh
export PROJECT_CODE_INTELLIGENCE_DATABASE_URL='postgresql://user:password@host:5432/database?sslmode=prefer'
```

Percent-encode special characters in the username or password.

The older split `PGVECTOR_*` variables remain supported, mostly for Docker
Compose and compatibility.

The MCP server is read-only by default and applies per-request database safety
limits. Expensive queries are bounded by `PROJECT_CODE_INTELLIGENCE_MCP_STATEMENT_TIMEOUT_MS`,
lock waits by `PROJECT_CODE_INTELLIGENCE_MCP_LOCK_TIMEOUT_MS`, and oversized
requests by `PROJECT_CODE_INTELLIGENCE_MCP_MAX_REQUEST_BYTES`. `get_code_intel_record`
returns concise metadata by default; pass `include_content: true` when an agent
needs the indexed text, capped by `PROJECT_CODE_INTELLIGENCE_MCP_MAX_RECORD_CONTENT_CHARS`.

For agent-heavy workflows, copy
[`docs/examples/AGENTS.md`](docs/examples/AGENTS.md) into the repository being
indexed so coding assistants know when to use the MCP index.

## Embeddings

Embeddings are the expected path for normal use. They are what make the MCP
index useful for semantic search instead of only exact text lookup.

Common paths are CPU FastEmbed, Apple Core ML (Neural Engine + GPU + CPU), AMD
Ryzen AI NPU, AMD GPU, NVIDIA GPU, and remote OpenAI-compatible providers.
`pci-doctor` prints the exact service commands that are available on the current
machine.

Local CPU, NPU, and GPU embedding services all publish the same host endpoint by
default: `http://127.0.0.1:18081/v1/embeddings`. Run only one local embedding
service at a time. Runtime-specific models have profile defaults; set model
environment variables only when overriding those defaults.

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

Profiles are runtime choices, not project modes. The local database is isolated
from the embedding services so users with an external Postgres/pgvector database
can start embeddings without also starting a local database.

| Profile or service | Use when |
| --- | --- |
| `pgvector` (`db`) | Local Postgres/pgvector database. Skip this when using an external database. |
| `cpu` (`fastembed`) | Portable local semantic-search demo with FastEmbed. |
| `npu` (`lemonade-npu`) | Experimental AMD Ryzen AI/XDNA NPU embeddings. |
| `amdgpu` (`llama-rocm`) | Experimental AMD ROCm llama.cpp embeddings. |
| `nvidia` (`llama-cuda`) | Experimental NVIDIA CUDA llama.cpp embeddings. |

List the profiles with:

```sh
docker compose config --profiles
```

For a local database, start:

```sh
docker compose up -d pgvector
```

For embeddings only, start the specific service:

```sh
docker compose --profile cpu up -d --build fastembed
docker compose --profile npu up -d lemonade-npu
docker compose --profile amdgpu up -d --build llama-rocm
docker compose --profile nvidia up -d --build llama-cuda
```

Most users should start with `cpu`, then let `pci-doctor` suggest hardware
specific commands if local acceleration is available.

## Docker Lifecycle

Use the exact service commands suggested by `pci-doctor`. Start `pgvector` only
when you want the local database; omit it when
`PROJECT_CODE_INTELLIGENCE_DATABASE_URL` points at an external database. Use
`stop` when you want to pause containers but keep them around:

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
Apple embeddings run on the macOS host, not inside Docker.

With coremltools installed (automatic on macOS arm64), `pci-doctor` detects the
Apple Neural Engine and recommends `pci-coreml-server`, which uses Core ML to
distribute inference across the ANE, GPU, and CPU. Without coremltools, it falls
back to host-native llama.cpp with Metal GPU acceleration.

To inspect per-operation device assignments:

```sh
pci-coreml-server --diagnose
```

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
PROJECT_CODE_INTELLIGENCE_PROFILE=my_project.code_profile:MyProjectProfile pci-index .
```

Profiles are ordinary Python code, so load them only from trusted local modules.

## Development

Run the local quality gate:

```sh
make check
```

Run the integration smoke. This starts the local Compose `pgvector` service if
needed:

```sh
make integration-smoke
```

Useful docs:

- [CONTRIBUTING.md](CONTRIBUTING.md): contributor workflow and guardrails
- [docs/PUBLIC_API.md](docs/PUBLIC_API.md): supported CLI, MCP, config, and Python import surfaces
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
