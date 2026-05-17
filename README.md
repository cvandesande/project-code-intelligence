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

**Private or sensitive codebases.** Most embedding setups send source text to a remote API. Every function signature, comment, and identifier you index leaves the machine. `project-code-intelligence` runs embeddings locally by default, using Apple Silicon (MLX), AMD (ROCm), or NVIDIA (CUDA) hardware when available and falling back to CPU otherwise. The code stays on the machine, and there's no per-token API cost.

## Where it doesn't help

Known-target retrieval — "give me lines 60–80 of `internal/foo/bar.go`" — is cheaper with plain `grep` + a bounded read than with the MCP. The index pays off when the agent doesn't already know the answer (semantic queries, caller graphs, project orientation, doc navigation). For pinpoint lookups against a file the agent has already identified, raw tools win on token cost.

## What it is

An MCP server backed by Postgres/pgvector. `pci-index` scans one or more Git repositories and stores source files, code records (functions, classes, symbols, config entries), static-analysis findings from SARIF, and candidate relationships between records. `pci-mcp` exposes that index to Claude Code, Codex, OpenCode, or any other MCP client. `pci-doctor` inspects the local machine and prints the exact startup commands for the current hardware. Embeddings run locally by default, using Apple Silicon (MLX), AMD (ROCm), or NVIDIA (CUDA) hardware when available and falling back to CPU otherwise, so indexed source text stays on the machine.

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

Then point your MCP client at `pci-mcp`. See [docs/MCP_SETUP.md](docs/MCP_SETUP.md) for client-specific configuration.

---

## Supported Hardware

`pci-doctor` is the source of truth for the current machine. It detects usable local runtimes and prints the exact startup command for each available path.

| Path | Runtime | Notes |
| --- | --- | --- |
| CPU | FastEmbed | Portable default for local testing and machines without accelerator support. |
| Apple Silicon | MLX | Native MLX embedding server (`pci-apple-embed-server`) using the GPU; Docker is still useful for Postgres. |
| AMD Ryzen AI NPU | Lemonade FLM | Experimental; requires supported XDNA hardware, driver, and firmware. |
| AMD GPU | llama.cpp ROCm | Uses the `amdgpu` Compose profile. |
| NVIDIA GPU | llama.cpp CUDA | Requires the NVIDIA driver and NVIDIA Container Toolkit. |
| Remote provider | OpenAI-compatible embeddings endpoint | Useful when local embeddings are not desired; source-derived text leaves the machine. |

## Installation

Install the CLI tools for your user with `uv`:

```sh
uv tool install /path/to/project-code-intelligence
```

This places the console scripts on your PATH (usually `~/.local/bin`). Make sure that path is on `PATH`:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

Without `uv`, use a virtualenv:

```sh
cd /path/to/project-code-intelligence
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

The full list of installed commands lives in [docs/PUBLIC_API.md](docs/PUBLIC_API.md).

## MCP Setup

Point Codex, Claude Code, or OpenCode at the installed `pci-mcp` command.
`pci-index --init-db --mcp-config codex .` prints a project-scoped read-only
Codex snippet for the current repo plus the required environment exports with
read-only credentials; `claude` and `opencode` are also supported.
See [docs/MCP_SETUP.md](docs/MCP_SETUP.md) for setup examples, database
configuration, and collection/repo filter guidance.

## Indexing

`pci-index .` indexes the current Git repository. You can also pass multiple repo paths to index them as a workspace:

```sh
cd /path/to/workspace
pci-index service-api web-ui shared-lib
```

`pci-index` infers a collection name from the paths. MCP tool filters then use repo names like `service-api`, not absolute filesystem paths. See [docs/MCP_SETUP.md](docs/MCP_SETUP.md) for the collection model.

Rerunning the same command is incremental: unchanged files are skipped and the Git snapshot is reused when the working tree hasn't changed. SARIF reports under the indexed repo paths are picked up automatically.

To reset one repo's indexed data and rebuild it:

```sh
pci-index --reset .
```

For advanced flags (text-only mode, custom SARIF locations, ingest tuning), see `pci-index --help`.

## Embeddings

Embeddings power semantic search. Local CPU, NPU, and GPU embedding services all publish the same default endpoint at `http://127.0.0.1:18081/v1/embeddings`. Run only one local embedding service at a time. `pci-doctor` prints the right startup command for the current hardware.

Default models download on first run.

Remote embedding endpoints receive source-derived text. For private code, use a local endpoint or a provider you trust, and set `PROJECT_CODE_INTELLIGENCE_ALLOW_REMOTE_EMBEDDING=1` only intentionally.

## Privacy

Do not publish database dumps, restore artifacts, SARIF output, embedding caches, model files, vector indexes, local MCP configs, or generated data from private repositories. These can contain source snippets, internal paths, symbols, findings, metadata, and embeddings derived from source text.

Remote embedding endpoints receive source-derived text. Use local embeddings for private code unless you have explicitly accepted the risk of sending that text to a remote provider.

Collections help organize multiple repos in one database and prevent accidental cross-repo MCP results, but they are not a security boundary. Use separate databases or database users when repos need stronger isolation. See [docs/MCP_SETUP.md](docs/MCP_SETUP.md#security-model).

## Documentation

- [docs/MCP_SETUP.md](docs/MCP_SETUP.md) — MCP client configuration, collections, repo filters, security model
- [docs/PUBLIC_API.md](docs/PUBLIC_API.md) — installed CLI commands, environment variables, MCP tool surface, Python imports
- [.env.example](.env.example) — available environment variables
- [AGENTS.md](AGENTS.md) — instructions for assistants working on this repo
- [CONTRIBUTING.md](CONTRIBUTING.md) — contributor workflow, local checks, and guardrails

## License

MIT. See [LICENSE](LICENSE).
