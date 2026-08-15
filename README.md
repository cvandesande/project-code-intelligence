# Project Code Intelligence

**Repository intelligence and change-safety evidence for coding agents.**

Project Code Intelligence (PCI) indexes Git repositories and gives coding
agents structured evidence for research, refactoring, maintenance, and security
work. Agents can search by concept or identifier, inspect candidate
relationships, estimate a change's blast radius, find repeated implementation
shapes, and query static-analysis findings before they edit code.

PCI is not an autonomous reviewer and its graph is not a compiler-grade call
graph. It narrows discovery and supplies evidence; the agent still verifies
important conclusions in source.

## What It Helps With

### Research and orientation

- Search exact identifiers, filenames, configuration keys, and known strings.
- Search by behavior when the relevant names are unknown.
- Inspect repository languages, file roles, snapshots, parser coverage, and
  index freshness.
- Fetch bounded records with paths, line ranges, metadata, and source snippets.
- Work across several repositories through named collections and repo filters.

### Safer changes

- Find candidate callers, callees, references, tests, and module-level wiring.
- Check blast-radius evidence before removing, renaming, or changing a symbol.
- Surface entry-point, orphan, and test-coverage signals.
- Inject nearby evidence into supported coding agents when definitions are
  added or removed.

### Maintenance and redundancy

- Find groups of functions that repeat a call-shape motif.
- Rank redundancy candidates by similarity, estimated abstraction cost, and
  likely net value.
- Run a repository audit for stale indexes, duplicate names, redundancy
  candidates, and static findings.

### Security and static analysis

- Ingest SARIF reports alongside source records.
- Search normalized findings by tool, rule, level, baseline state, or path.
- Fetch diagnostics, code flows, and run metadata without making an agent parse
  raw SARIF artifacts.

## How It Works

`pci index` parses repository files into bounded records, extracts metadata and
candidate relationships, and stores snapshots in Postgres/pgvector. Semantic
embeddings are optional: lexical search and most structural evidence remain
available with `--no-embed`.

`pci mcp` exposes the index through a local stdio MCP server. Coding agents use
its filter-oriented tools to discover likely-relevant code, then read and
verify the live source before acting.

The default local architecture is:

- **Postgres/pgvector:** Docker or Podman Compose.
- **Linux embedding services:** Podman Quadlet units managed by user systemd.
- **Apple Silicon embeddings:** a native MLX service.
- **Agent integration:** a stdio MCP server, with optional edit-evidence hooks.

Remote Postgres and OpenAI-compatible embedding endpoints are supported when
that tradeoff is intentional.

## Quick Start

Install the CLI from a checkout:

```sh
uv tool install /path/to/project-code-intelligence
export PATH="$HOME/.local/bin:$PATH"
```

Start the bundled database and the best available local embedding backend:

```sh
pci doctor --start
pci doctor
```

Index a Git repository:

```sh
cd /path/to/repo
pci index .
```

Install MCP configuration for your coding agent:

```sh
pci mcp install --target codex
```

Supported targets include `claude`, `codex`, `opencode`, `pi`, `vscode`,
`copilot`, `cline`, and `zed`. See [docs/MCP_SETUP.md](docs/MCP_SETUP.md) for
client-specific setup, project scoping, and credential handling.

For lexical search without embeddings:

```sh
pci doctor --start-db
pci index --no-embed .
```

## Core Agent Tools

| Tool | Purpose |
| --- | --- |
| `code_intel_status` | Index freshness, scope, record counts, and query capabilities. |
| `list_code_intel_files` | File inventory filtered by language, role, path, or generated/test status. |
| `search_code_intel_text` | Exact indexed search for symbols, filenames, keys, and known strings. |
| `search_code_intel_semantic` | Concept search when identifiers are unknown. |
| `get_code_intel_record` | Fetch complete indexed records and metadata. |
| `related_code_intel` | Candidate caller, callee, reference, and related-symbol evidence. |
| `blast_radius` | Callers, tests, wiring, entry-point signals, and semantic neighbors for a proposed change. |
| `find_redundancy` | Repeated call-shape groups ranked by likely refactoring value. |
| `search_static_findings` | Filter normalized SARIF findings. |
| `get_static_finding` | Fetch diagnostics, code flows, and static-analysis run details. |

Run `pci audit` for a whole-tree evidence report.

## Evidence, Not Verdicts

PCI deliberately distinguishes stronger indexed facts from approximate and
heuristic evidence.

- Candidate relationship edges are not type-checked call-graph facts.
- Blast radius cannot prove that a change is safe.
- Redundancy scores cannot decide whether two functions should share an
  abstraction.
- Static findings retain the limitations of their originating analyzer.
- An index can be stale after uncommitted or newly committed changes.
- Semantic retrieval can miss relevant code or return plausible neighbors.

Verify important callers and findings in live source. Use direct file reads for
known paths and small bounded questions; PCI is most useful when the location,
name, or impact is not yet known.

## Installation

### Python CLI

Install for the current user with `uv`:

```sh
uv tool install /path/to/project-code-intelligence
```

For an editable development install:

```sh
cd /path/to/project-code-intelligence
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

### Nix

On NixOS or another Linux host with flakes enabled:

```sh
nix build
nix run . -- doctor --skip-db --embedding skip
nix develop
```

Install persistently into the user profile:

```sh
nix profile install .#project-code-intelligence
```

The Nix closure contains the CLI, MCP server, Python dependencies, and bundled
Compose and Quadlet assets. Heavy Linux embedding runtimes remain in Podman
containers rather than becoming host-native Nix dependencies.

### Local database

Start only the bundled Postgres/pgvector database:

```sh
pci doctor --start-db
```

The installed Compose file is materialized into a user cache. To use a custom
copy instead:

```sh
export PCI_COMPOSE_FILE=/path/to/docker-compose.yml
pci doctor --start-db
```

### Local embedding service

The CLI includes the service templates; there is no separate PCI embedding
package. On Linux, install Podman and ensure `systemctl --user` works, then run:

```sh
pci doctor
pci doctor --start-embedding
```

PCI detects available hardware, materializes only the selected backend under
`~/.config/containers/systemd/`, reloads user systemd, and starts it. Stale PCI
units for other embedding backends are stopped and removed. Images and default
models download on first use.

Choose a backend explicitly when desired:

```sh
# AMD GPU
pci doctor --start-embedding --embedding-backend rocm

# NVIDIA GPU
pci doctor --start-embedding --embedding-backend cuda

# Portable CPU fallback
pci doctor --start-embedding --embedding-backend fastembed

# AMD Ryzen AI NPU (experimental)
pci doctor --start-embedding --embedding-backend lemonade
```

Available selectors are `auto`, `fastembed`, `lemonade`, `rocm`, `cuda`, and
`apple`. PCI rejects a requested backend when its required hardware or runtime
is unavailable. `apple` runs natively rather than through Quadlet.

To start the database and an explicit backend together:

```sh
pci doctor --start --embedding-backend rocm
```

Containerized backends publish an OpenAI-compatible endpoint at
`http://127.0.0.1:18081/v1/embeddings` by default. Run one local backend at a
time because they share this endpoint.

| Hardware | Backend | Runtime notes |
| --- | --- | --- |
| CPU | FastEmbed | Portable fallback; Podman Quadlet. |
| Apple Silicon | MLX | Native process using the Apple GPU. |
| AMD Ryzen AI NPU | Lemonade FLM | Experimental; requires supported XDNA hardware, driver, and firmware. |
| AMD GPU | llama.cpp ROCm | Podman Quadlet with `/dev/kfd` and `/dev/dri`. |
| NVIDIA GPU | llama.cpp CUDA | Requires the NVIDIA driver, Container Toolkit, and Podman CDI support. |

Stop embedding services without touching the database:

```sh
pci doctor --stop-embedding
```

Remove generated local services, caches, and the bundled database volume:

```sh
pci doctor --clean
```

`--clean` is destructive and prompts before removing data.

### Remote embeddings

Start only the database, then configure a trusted OpenAI-compatible provider:

```sh
pci doctor --start-db
export PCI_ALLOW_REMOTE_EMBEDDING=1
export PCI_EMBEDDING_ENDPOINT=https://api.openai.com/v1/embeddings
export PCI_EMBEDDING_ENDPOINT_MODEL=text-embedding-3-small
export OPENAI_API_KEY=...
pci doctor
```

Remote endpoints receive source-derived text. Enable them only when that is
acceptable for the repositories being indexed.

## Indexing Repositories

Index one repository:

```sh
pci index /path/to/repo
```

Index several repositories as a workspace:

```sh
cd /path/to/workspace
pci index service-api web-ui shared-lib
```

PCI infers collection and repository names from the paths. MCP clients filter
by these logical names rather than absolute filesystem paths. Indexing is
incremental: unchanged files are reused when compatible snapshots exist.

SARIF reports found under indexed repository paths are ingested automatically.
Reset and rebuild one repository with:

```sh
pci index --reset /path/to/repo
```

Use `pci status` to inspect indexing runs and `pci index --help` for parser,
embedding, collection, and database options.

## MCP and Agent Hooks

Create project-scoped read-only database credentials and print an MCP config:

```sh
pci index --init-db --mcp-config codex .
```

Install or remove MCP configuration without reindexing:

```sh
pci mcp install --target codex
pci mcp install --target codex --uninstall
```

Generated client configuration contains no database password. Credentials are
stored under the user's PCI configuration directory with mode `0600`. Pi uses a
project-local `.pi/extensions/` MCP bridge. Cline requires `--config-path`
because its settings file is user-scoped.

Optional hooks can remind an agent to use the index and inject evidence near
edits that add or remove definitions:

```sh
pci hook install --target codex
```

Hook support and installation details vary by client. Hooks are an aid, not an
enforcement or correctness mechanism. See [docs/MCP_SETUP.md](docs/MCP_SETUP.md)
and [docs/SYSTEM_PROMPT.md](docs/SYSTEM_PROMPT.md).

## Privacy and Security

The local default keeps source-derived records and embeddings on the machine.
That does not make every artifact safe to publish.

Do not commit or distribute database dumps, restore artifacts, SARIF output,
embedding caches, model files, vector indexes, generated data from private
repositories, or local MCP credential files. They can contain source snippets,
paths, symbols, findings, metadata, and embeddings derived from source.

Collections organize repositories but are not a security boundary. Use separate
databases or database users when repositories require stronger isolation.
Project-scoped MCP credentials restrict normal access, but they do not replace
host and database security.

## When PCI Is a Good Fit

PCI tends to help when:

- the repository or workspace is large or unfamiliar;
- identifiers are unknown at the start of a task;
- generated files make broad reads expensive or noisy;
- a refactor needs caller, test, and wiring evidence;
- maintenance work needs repeated-pattern discovery;
- static findings need to be correlated with source;
- several repositories must be searched through one interface.

It may add little value for a small familiar repository, a known file and line
range, or a question answered by one bounded `rg` or file read. The goal is not
to replace standard developer tools; it is to improve the uncertain discovery
and change-planning steps around them.

Token and cost reduction can be a useful side effect, but PCI does not promise
it. Measure whether the index improves real sessions using
[docs/EVALUATING_VALUE.md](docs/EVALUATING_VALUE.md) and the
[session retrospective prompt](docs/SESSION_RETROSPECTIVE_PROMPT.md).

## Development

The MCP server uses stdio; Docker Compose is for the local database, not for
hosting the MCP process. Run the full development gate with:

```sh
nix develop
make check
```

For ingest, database, or MCP behavior changes, also run:

```sh
docker compose up -d pgvector
make integration-smoke
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) for project
conventions and publication safeguards.

## Documentation

- [docs/MCP_SETUP.md](docs/MCP_SETUP.md) — MCP clients, scopes, credentials, and security model
- [docs/PUBLIC_API.md](docs/PUBLIC_API.md) — CLI, environment, MCP, and Python compatibility surfaces
- [docs/EVALUATING_VALUE.md](docs/EVALUATING_VALUE.md) — evaluating PCI on real coding sessions
- [docs/SYSTEM_PROMPT.md](docs/SYSTEM_PROMPT.md) — agent instructions for using PCI
- [docs/SYSTEM_PROMPT_RATIONALE.md](docs/SYSTEM_PROMPT_RATIONALE.md) — prompt-design rationale
- [docs/SESSION_RETROSPECTIVE_PROMPT.md](docs/SESSION_RETROSPECTIVE_PROMPT.md) — end-of-session evaluation prompt
- [.env.example](.env.example) — environment configuration reference
- [CONTRIBUTING.md](CONTRIBUTING.md) — development workflow

## License

MIT. See [LICENSE](LICENSE).
