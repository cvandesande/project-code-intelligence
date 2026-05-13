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

Install the CLI tools once, then let `pci-doctor` inspect the machine:

```sh
uv tool install /path/to/project-code-intelligence
pci-doctor
```

`pci-doctor` checks Postgres/pgvector, embeddings, and available local
acceleration. It prints the startup commands that fit the current machine. For
a fully local setup, start the suggested `pgvector` command and one embedding
service. If you already have an external Postgres/pgvector database, set
`PROJECT_CODE_INTELLIGENCE_DATABASE_URL` and start only the embedding service
you want to use.

Run `pci-doctor` again after starting services. When it reports `Status: ok
ready`, index a Git repository:

```sh
cd /path/to/repo-to-index
pci-index . --dry-run
pci-index .
pci-mcp-smoke
```

After indexing, configure your assistant to run `pci-mcp`. See MCP Setup below.

## Indexing

Use `.` when you mean the current directory. You can also index one or more
repositories without changing directories:

```sh
pci-index /path/to/repo-to-index
pci-index /path/to/repo-a /path/to/repo-b
```

For a workspace with related repositories, use one collection for the workspace
and stable repo names under that collection:

```sh
cd /path/to/workspace
PROJECT_CODE_INTELLIGENCE_COLLECTION=workspace-name pci-index openwrt ask-cmm fci
```

MCP repo filters then use those repo names, such as `openwrt`, not absolute
filesystem paths. Run `code_intel_status` without a repo filter to see the
available collection and repo keys.

For advanced ingest options, put them after `--`:

```sh
pci-index /path/to/repo-to-index -- --limit-files 100
```

If indexing is interrupted, rerun the same command. `pci-index .` reuses the
same snapshot when the Git tree is unchanged, keeps compatible existing
embeddings, and fills in records that are still missing embeddings. In normal
incremental mode it only reparses changed files.

Text-only indexing is available as a fallback for bootstrap, debugging, or
privacy-sensitive environments:

```sh
pci-index . --no-embed
```

For that mode, choose the Postgres-only command from `pci-doctor` and verify
that the database is reachable. `pci-doctor` may still warn about the missing
embedding endpoint, which is expected for a deliberate text-only run.

To wipe and rebuild the code-intelligence tables in the configured database:

```sh
pci-index --reset
```

This drops and recreates this project's `project_code_intel_*` tables. It does
not drop the database or unrelated tables. The command prints the resolved
database target, asks for confirmation before deleting anything, and exits
without scanning. Run `pci-index .` afterwards to rebuild the index. For
non-interactive automation, add `--i-know-this-deletes-code-intel-db`.

In a brand-new local repository, make an initial commit before scanning so the
indexer has a Git `HEAD` snapshot.

## Supported Hardware

`pci-doctor` is the source of truth for the current machine. It detects usable
local runtimes and prints the exact startup command for each available path.

| Path | Runtime | Notes |
| --- | --- | --- |
| CPU | FastEmbed | Portable default for local testing and machines without accelerator support. |
| Apple Silicon | Core ML or llama.cpp Metal | Core ML can use ANE, GPU, and CPU; Docker is still useful for Postgres. |
| AMD Ryzen AI NPU | Lemonade FLM | Experimental; requires supported XDNA hardware, driver, and firmware. |
| AMD GPU | llama.cpp ROCm | Experimental; uses the `amdgpu` Compose profile. |
| NVIDIA GPU | llama.cpp CUDA | Experimental; requires the NVIDIA driver and NVIDIA Container Toolkit. |
| Remote provider | OpenAI-compatible embeddings endpoint | Useful when local embeddings are not desired; source-derived text leaves the machine. |

## Installation

Install the CLI tools for your user with `uv`:

```sh
uv tool install /path/to/project-code-intelligence
```

This places `pci-doctor`, `pci-index`, `pci-mcp`, and the other console scripts
on your PATH (usually `~/.local/bin`). Platform-specific optional dependencies
are selected by the package metadata where supported.

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

Point Codex, Claude Code, OpenCode, or another MCP client at the installed
`pci-mcp` command. Use `command -v pci-mcp` to find the absolute path if your
client does not inherit your shell `PATH`.

For setup examples, database configuration, and collection/repo filter guidance,
see [docs/MCP_SETUP.md](docs/MCP_SETUP.md).

## Embeddings

Embeddings are the expected path for normal use. They are what make the MCP
index useful for semantic search instead of only exact text lookup.

Local CPU, NPU, and GPU embedding services all publish the same host endpoint by
default: `http://127.0.0.1:18081/v1/embeddings`. Run only one local embedding
service at a time. Runtime-specific models have profile defaults.

Run `pci-doctor` to see which embedding paths and models are available on the
current machine:

```sh
pci-doctor
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

When unsure, use the commands from `pci-doctor`. The `cpu` profile is the
portable local fallback.

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
Apple embeddings run on the macOS host, not inside Docker. See Supported
Hardware for the available acceleration paths.

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
- [docs/MCP_SETUP.md](docs/MCP_SETUP.md): MCP setup examples for Codex, Claude Code, and OpenCode
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
