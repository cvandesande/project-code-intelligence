# Project Code Intelligence

Coding agents are only as good as the context they get. Reading isn't the problem. Knowing what to read is.

## The actual problem

When an agent works through a codebase without tooling, it reads files to find what it needs. The waste isn't in the reading itself — it's in the speculative loading that comes before. A file gets opened because it *might* contain what's needed. Sometimes it does, often it doesn't. On a large codebase, that speculation burns tokens fast.

Plain code RAG helps by embedding chunks and retrieving similar passages, but it ignores what makes source code different from prose. Code has structure: files have roles, functions have callers, static analysis has findings. None of that survives chunking into text.

`project-code-intelligence` stores that structure directly. The index combines semantic search with a structured map of the codebase, so an agent can retrieve relevant context, fetch exact records, follow candidate relationships, and surface static-analysis findings without reading files speculatively.

## Where this pays off

**Large and generated files.** Every serious codebase has files that exist to be consumed by machines: protobuf-generated code, auto-generated clients, ORM models. These can be hundreds of kilobytes. Without the index, an agent reads a huge slice or runs multiple greps and still loads more than it needs. With the index, it queries for the symbol and gets a 20-line snippet.

**Not knowing what you're looking for.** Grep requires knowing the word. So does any tool that navigates by symbol name. Semantic search doesn't. "Find code that handles connection retry backoff" or "where does TLS configuration get assembled" are questions grep can't answer without already knowing the answer. The index makes those queries cheap.

**Getting oriented.** Understanding the shape of an unfamiliar codebase (which languages, what's generated versus hand-written, where the entry points are) normally costs exploratory reading. `code_intel_status` and `list_code_intel_files` with filters make it a single query.

**Finding callers.** `related_code_intel` returns callers with file paths, line numbers, and snippets across the whole codebase in one round-trip. The alternative is grep plus reading each match in context.

**Private or sensitive codebases.** Most embedding setups send source text to a remote API. Every function signature, comment, and identifier you index leaves the machine. `project-code-intelligence` runs embeddings locally by default, on CPU, Apple Silicon via MLX, AMD, or NVIDIA hardware. The code stays on the machine, and there's no per-token API cost.

## What it is

An MCP server backed by Postgres/pgvector. `pci-index` scans one or more Git repositories and stores source files, code records (functions, classes, symbols, config entries), static-analysis findings from SARIF, and candidate relationships between records. `pci-mcp` exposes that index to Claude Code, Codex, OpenCode, or any other MCP client. `pci-doctor` inspects the local machine and prints the exact startup commands for the current hardware. Embeddings run locally by default on CPU, Apple Silicon via MLX, AMD, or NVIDIA, so indexed source text stays on the machine.

The package is generic by default. Project-specific behavior belongs in code profiles, with [`example.py`](src/project_code_intelligence/code_profiles/example.py) as the public starting point.

## What it won't do well

Call graph edges are heuristic — inferred from symbol co-occurrence, not proven by a type checker or linker. They're useful for navigating to candidates, not for asserting definitive caller/callee relationships. Treat them as "probably calls" and verify in source when correctness matters.

Text search falls back through multiple strategies when the primary index finds nothing. Results are still useful, but relevance ranking degrades. If text search returns noise, try semantic search instead. They use different mechanisms and one often succeeds where the other doesn't.

Embeddings are what make semantic search useful. Text-only indexing (`--no-embed`) works as a fallback but limits the MCP server to lexical search. For most use cases, running a local embedding service is worth the setup cost.

## Quick Start

Install the CLI tools once, then let `pci-doctor` inspect the machine:

```sh
uv tool install /path/to/project-code-intelligence
pci-doctor
```

`pci-doctor` checks Postgres/pgvector, embeddings, and available local acceleration. It prints the startup commands that fit the current machine. Start the suggested services, then run `pci-doctor` again. When it reports `Status: ok ready`, index a repository:

```sh
cd /path/to/repo
pci-index . --dry-run
pci-index .
pci-mcp-smoke .
```

Then point your MCP client at `pci-mcp`. See [docs/MCP_SETUP.md](docs/MCP_SETUP.md) for client-specific configuration and collection options.

---

## Indexing

Use `.` when you mean the current directory. You can also index one or more
repositories without changing directories:

```sh
pci-index /path/to/repo-to-index
pci-index /path/to/repo-a /path/to/repo-b
```

`pci-index` infers a collection name for you. A single repo path uses that repo
directory name. Multiple repo paths use their common parent directory name. For
a workspace with related repositories, run from the workspace and pass the repo
directories:

```sh
cd /path/to/workspace
pci-index service-api web-ui shared-lib
```

MCP repo filters then use those repo names, such as `service-api`, not
absolute filesystem paths. Run `code_intel_status` without a repo filter to
see the available collection and repo keys. Use `--collection` only when you
want a name different from the inferred workspace name:

```sh
pci-index --collection workspace-name service-api web-ui shared-lib
```

For advanced ingest options, put them after `--`:

```sh
pci-index /path/to/repo-to-index -- --limit-files 100
```

SARIF files are discovered automatically under the selected repo paths. Obvious
test fixtures such as `*-expected.sarif` under test directories are ignored
unless passed explicitly with `--sarif`. The indexer records SARIF findings even
when reports live in ignored output directories, and prints soft notes when
freshness cannot be verified, for example when a report file is older than the
indexed commit. Use `--sarif` for reports outside the selected repos, or
`--no-profile-sarif` to disable automatic SARIF discovery.

If indexing is interrupted, rerun the same command. `pci-index .` reuses the
same snapshot when the Git tree is unchanged, keeps compatible existing
embeddings, and fills in records that are still missing embeddings. In normal
incremental mode it only reparses changed files.

When upgrading this tool, reindex if release notes or local changes mention a
parser, chunker, schema, or profile version bump. Those changes affect the
records stored in Postgres, so old snapshots may not contain newly indexed
metadata such as Makefile package pins.

Text-only indexing is available as a fallback for bootstrap, debugging, or
privacy-sensitive environments:

```sh
pci-index . --no-embed
```

For that mode, choose the Postgres-only command from `pci-doctor` and verify
that the database is reachable. `pci-doctor` may still warn about the missing
embedding endpoint, which is expected for a deliberate text-only run.

To delete indexed data for one repo and rebuild it:

```sh
pci-index --reset .
```

This deletes snapshots, records, edges, embeddings, and findings for the
selected collection and repo key only. Other repos and the schema are untouched.
The command prints the resolved database target, asks for confirmation before
deleting anything, and exits without scanning. Run `pci-index .` afterwards to
rebuild the index.

To delete all indexed data in the configured database while keeping the schema:

```sh
pci-index --reset-all
```

For non-interactive automation, add `--i-know-this-deletes-code-intel-db`.

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
SARIF input under the selected repo paths. A project can add its own profile for
domain-specific file roles, metadata, records, security context, or extra SARIF
locations.

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

Remote embedding endpoints receive source-derived text. Use local embeddings for
private code unless you have explicitly accepted the risk of sending that text to
a remote provider.

Collections help organize multiple repos in one database and prevent accidental
cross-repo MCP results, but they are not a security boundary. Use separate
databases or database users when repos need stronger isolation. See
[docs/MCP_SETUP.md](docs/MCP_SETUP.md#security-model).

## License

MIT. See [LICENSE](LICENSE).
