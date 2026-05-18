# Project Code Intelligence

Coding agents are only as good as the context they get. Reading isn't the hard
part; knowing what to read is.

When an agent works through a codebase without tooling, it reads files to find
what it needs. The waste is everything it loads that turns out not to matter. A
file gets opened because it might contain the answer. Sometimes it does, often
it doesn't. On a large codebase, that speculative loading burns tokens fast.

`project-code-intelligence` inverts that workflow. Instead of reading to find,
agents query to decide what's worth reading. The index stores source files, code
records, SARIF static-analysis findings, semantic embeddings, and candidate
relationships in Postgres/pgvector, then exposes them through an MCP server.

The default setup is local: Postgres/pgvector and embeddings can run on your
machine, so source-derived text does not need to leave it. Remote databases and
OpenAI-compatible embedding endpoints are supported when you choose that
tradeoff.

## Where It Pays Off

**Large and generated files.** Protobuf output, generated clients, ORM models,
and other machine-written files can be hundreds of kilobytes. Usually you need
one method, not the whole file. Query the index for the symbol, then fetch the
specific record or chunk.

**Unknown terms.** Grep requires knowing the word. Semantic search can answer
questions like "where does TLS configuration get assembled" before you know the
identifier names.

**Project orientation.** `code_intel_status` and `list_code_intel_files` show
languages, generated/test/source roles, snapshots, parser health, and indexed
record counts without exploratory file reads.

**Related code.** `related_code_intel` returns candidate related/caller/callee
records with paths, line ranges, and snippets across the indexed codebase.

**Static-analysis context.** SARIF findings are indexed alongside code records,
so agents can search findings and fetch code-flow details without opening report
artifacts directly.

## How To Use Results

Search snippets are not the answer. They are evidence for deciding whether a
record is worth reading. "Is this probably the function I need?" can often be
answered from a snippet. "Does this handle the edge case correctly?" still
requires fetching the record or reading the source.

The index does not replace reading code. It replaces the part where you are not
sure what to read yet.

## Limits

- Known-target reads, such as "show lines 60-80 of `internal/foo/bar.go`", are
  cheaper with raw shell tools and a bounded file read.
- Related-code edges are heuristic candidates, not type-checked call graph
  facts. Verify important relationships in source.
- Text search uses several fallback strategies. If exact search returns noise,
  try semantic search; it uses a different mechanism.
- Text-only indexing (`--no-embed`) works for lexical search, but semantic
  search requires an embedding endpoint.

## Quick Start

Install the CLI tools and start the local services that fit the machine:

```sh
uv tool install /path/to/project-code-intelligence
pci-doctor --start
pci-doctor
```

When `pci-doctor` reports `Status: ok ready`, index a repository:

```sh
cd /path/to/repo
pci-index . --dry-run
pci-index .
pci-mcp-smoke .
```

Then point your MCP client at `pci-mcp`. See [docs/MCP_SETUP.md](docs/MCP_SETUP.md)
for client-specific configuration.

---

## Supported Hardware

`pci-doctor` is the source of truth for the current machine. It detects usable
local runtimes and prints the exact startup command for each available path.

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

This places the console scripts on your PATH, usually `~/.local/bin`:

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

Point your MCP client at the installed `pci-mcp` command. To print
project-scoped read-only config and required environment exports:

```sh
pci-index --init-db --mcp-config codex .
```

`--mcp-config` also supports `claude`, `opencode`, `vscode`, `copilot`, `cline`,
and `zed`. See [docs/MCP_SETUP.md](docs/MCP_SETUP.md) for setup examples,
database configuration, and collection/repo filter guidance.

## Indexing

`pci-index .` indexes the current Git repository. Pass multiple repo paths to
index a workspace:

```sh
cd /path/to/workspace
pci-index service-api web-ui shared-lib
```

`pci-index` infers a collection name from the paths. MCP tool filters then use
repo names like `service-api`, not absolute filesystem paths. See
[docs/MCP_SETUP.md](docs/MCP_SETUP.md) for the collection model.

Rerunning the same command is incremental: unchanged files are skipped and the
Git snapshot is reused when the working tree hasn't changed. SARIF reports under
the indexed repo paths are picked up automatically.

To reset one repo's indexed data and rebuild it:

```sh
pci-index --reset .
```

For advanced flags, see `pci-index --help`.

## Embeddings And Privacy

Embeddings power semantic search. Local CPU, NPU, and GPU embedding services all
publish the same default endpoint at `http://127.0.0.1:18081/v1/embeddings`.
Run only one local embedding service at a time. `pci-doctor --start` picks the
best available local path, and default models download on first run.

Remote embedding endpoints receive source-derived text. Set
`PCI_ALLOW_REMOTE_EMBEDDING=1` only when that is
intentional.

Do not publish database dumps, restore artifacts, SARIF output, embedding
caches, model files, vector indexes, local MCP configs, or generated data from
private repositories. These can contain source snippets, internal paths, symbols,
findings, metadata, and embeddings derived from source text.

Collections help organize multiple repos in one database, but they are not a
security boundary. Use separate databases or database users when repos need
stronger isolation. See [docs/MCP_SETUP.md](docs/MCP_SETUP.md#security-model).

## Documentation

- [docs/MCP_SETUP.md](docs/MCP_SETUP.md) — MCP client configuration, collections, repo filters, security model
- [docs/PUBLIC_API.md](docs/PUBLIC_API.md) — installed CLI commands, environment variables, MCP tool surface, Python imports
- [.env.example](.env.example) — available environment variables
- [docs/examples/AGENTS.md](docs/examples/AGENTS.md) — example guidance for coding agents using the MCP server
- [AGENTS.md](AGENTS.md) — instructions for assistants working on this repo
- [CONTRIBUTING.md](CONTRIBUTING.md) — contributor workflow, local checks, and guardrails

## License

MIT. See [LICENSE](LICENSE).
