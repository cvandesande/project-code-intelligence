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
It does not replace reading code. It replaces the part where you are not sure
what to read yet.

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

## How Agents Use This

An agent connected to `project-code-intelligence` handles discovery questions
differently. "Where is `Foo` used?", "what calls `Bar`?", "find code that
handles X" become one-hop queries that return file paths, line ranges, and
snippets — not the answer, but enough to decide which files are worth reading.

The agent still reads files. The change is in the *find* step: on a large or
unfamiliar codebase, that step is where speculative-read tokens add up. The
index cuts them down.

How much it actually saves depends on the codebase and the work. The included
[session-retrospective prompt](docs/SESSION_RETROSPECTIVE_PROMPT.md)
helps you tell, and [docs/EVALUATING_VALUE.md](docs/EVALUATING_VALUE.md) walks
through how to interpret the answer. The system prompt is shaped so an agent
doesn't over-apply MCP on questions where `Read` is cheaper.

## Limits

- Known-target reads, such as "show lines 60-80 of `internal/foo/bar.go`", are
  cheaper with raw shell tools and a bounded file read.
- Related-code edges are heuristic candidates, not type-checked call graph
  facts. Verify important relationships in source.
- Text search uses several fallback strategies. If exact search returns noise,
  try semantic search; it uses a different mechanism.
- Text-only indexing (`--no-embed`) works for lexical search, but semantic
  search requires an embedding endpoint.

For small or familiar codebases the overhead may not pay back at all. See
[docs/EVALUATING_VALUE.md](docs/EVALUATING_VALUE.md) for a session
retrospective workflow you can run against your own usage to decide
whether the install is earning its keep.

## Quick Start

Install the CLI tools, then choose the startup path that fits the machine.
For a local all-in-one setup with a bundled pgvector database and a local
embedding service:

```sh
uv tool install /path/to/project-code-intelligence
pci doctor --start
pci doctor
```

On low-power machines, or when embeddings should run elsewhere, start only the
bundled database and point PCI at an OpenAI-compatible embedding endpoint:

```sh
uv tool install /path/to/project-code-intelligence
pci doctor --start-db
export PCI_ALLOW_REMOTE_EMBEDDING=1
export PCI_EMBEDDING_ENDPOINT=https://api.openai.com/v1/embeddings
export PCI_EMBEDDING_ENDPOINT_MODEL=text-embedding-3-small
export OPENAI_API_KEY=...
pci doctor
```

When `pci doctor` reports the database and embedding endpoint are ready, index a
repository:

```sh
cd /path/to/repo
pci index .
```

For lexical search only, skip embeddings explicitly:

```sh
pci index --no-embed .
```

Then point your MCP client at `pci mcp`. See [docs/MCP_SETUP.md](docs/MCP_SETUP.md)
for client-specific configuration.

---

## Supported Hardware

`pci doctor` is the source of truth for the current machine. It detects usable
local runtimes and prints the exact startup command for each available path.

| Path | Runtime | Notes |
| --- | --- | --- |
| CPU | FastEmbed | Portable default for local testing and machines without accelerator support; runs as a Podman Quadlet unit. |
| Apple Silicon | MLX | Native MLX embedding server (`pci embed apple`) using the GPU; no container involved. |
| AMD Ryzen AI NPU | Lemonade FLM | Experimental; requires supported XDNA hardware, driver, and firmware; runs as a Podman Quadlet unit. |
| AMD GPU | llama.cpp ROCm | Runs as a Podman Quadlet unit (`pci doctor --start-embedding`). |
| NVIDIA GPU | llama.cpp CUDA | Requires the NVIDIA driver and NVIDIA Container Toolkit; runs as a Podman Quadlet unit. |
| Remote provider | OpenAI-compatible embeddings endpoint | Useful when local embeddings are not desired; source-derived text leaves the machine. |

The bundled Postgres/pgvector database still runs under Docker (or Podman)
Compose. The local embedding backends above run as Podman Quadlet units,
managed through `systemctl --user`, not Compose.

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

On NixOS, or on another Linux host with Nix flakes enabled, build or run the
core CLI package directly from a checkout:

```sh
nix build
nix run . -- doctor --skip-db --embedding skip
nix develop
```

To install the commands persistently into your user profile:

```sh
nix profile install .#project-code-intelligence
```

To uninstall that profile entry later:

```sh
nix profile list
nix profile remove <index>
```

The Nix package intentionally keeps the Nix closure focused on the
CLI/MCP/indexing commands, Python runtime dependencies, and bundled Compose
assets. Heavy embedding runtimes are not added as host-native Nix dependencies.
Use `pci doctor --start-db` for the bundled database, then either index
text-only with `pci index --no-embed .` or configure a trusted
OpenAI-compatible remote embedding endpoint explicitly:

```sh
export PCI_ALLOW_REMOTE_EMBEDDING=1
export PCI_EMBEDDING_ENDPOINT=https://api.openai.com/v1/embeddings
export PCI_EMBEDDING_ENDPOINT_MODEL=text-embedding-3-small
export OPENAI_API_KEY=...
```

The bundled Compose file is materialized from installed package data when the
project is installed through `uv tool` or Nix. To customize Compose behavior,
copy the repo's `docker-compose.yml` and point PCI at your copy instead of
editing installed package files:

```sh
export PCI_COMPOSE_FILE=/path/to/docker-compose.yml
pci doctor --start-db
```

`pci doctor --clean` stops local services, removes the bundled database volume,
removes generated Compose cache files and Quadlet unit files, and removes the
generated `pci index` user config. `make tool-uninstall` runs that cleanup
first, then uninstalls the `uv tool` command shims.

The full list of installed commands lives in [docs/PUBLIC_API.md](docs/PUBLIC_API.md).

## MCP Setup

To create private project-scoped read-only credentials and print a
credential-free client config:

```sh
pci index --init-db --mcp-config codex .
```

`--mcp-config` also supports `claude`, `opencode`, `vscode`, `copilot`, `cline`,
and `zed`. The generated config launches `pci mcp --scope /path/to/project`;
database credentials are stored under the user's PCI config directory with
mode `0600`, never in the repository or harness config. Install or remove the
printed configuration with:

```sh
pci mcp install --target codex
pci mcp install --target codex --uninstall
```

The install command supports every target above plus `pi`. Pi installs a
project-local `.pi/extensions/` MCP bridge; add its evidence hooks with
`pci hook install --target pi`. Cline additionally requires `--config-path`
because its settings file is user-scoped. See
[docs/MCP_SETUP.md](docs/MCP_SETUP.md) for database and scope guidance.

A ready-to-paste system prompt for the connected agent lives at
[docs/SYSTEM_PROMPT.md](docs/SYSTEM_PROMPT.md). The design
notes in [docs/SYSTEM_PROMPT_RATIONALE.md](docs/SYSTEM_PROMPT_RATIONALE.md)
cover the prompt-engineering choices behind it (e.g., asking the agent for
negative-only self-reports instead of token-savings estimates it can't compute
honestly) — useful reading if you're tuning agent prompts for any MCP server,
not just this one.

To install PCI's edit evidence and session banner hooks for Codex in the
current repository:

```sh
pci hook install --target codex
```

Codex requires reviewing newly installed command hooks with `/hooks` before
they run. Add `--user` for `~/.codex/hooks.json`, or `--uninstall` to remove
only PCI's handlers while preserving other hooks.

Install MCP server configuration without reindexing (`--target` also supports
`claude`, `opencode`, `pi`, `vscode`, `copilot`, `cline`, and `zed`):

```sh
pci mcp install --target codex
```

The command preserves unrelated Codex TOML and can be reversed with
`pci mcp install --target codex --uninstall`.

## Indexing

`pci index .` indexes the current Git repository. Pass multiple repo paths to
index a workspace:

```sh
cd /path/to/workspace
pci index service-api web-ui shared-lib
```

`pci index` infers a collection name from the paths. MCP tool filters then use
repo names like `service-api`, not absolute filesystem paths. See
[docs/MCP_SETUP.md](docs/MCP_SETUP.md) for the collection model.

Rerunning the same command is incremental: unchanged files are skipped and the
Git snapshot is reused when the working tree hasn't changed. SARIF reports under
the indexed repo paths are picked up automatically.

To reset one repo's indexed data and rebuild it:

```sh
pci index --reset .
```

For advanced flags, see `pci index --help`.

## Embeddings And Privacy

Embeddings power semantic search. Local CPU, NPU, and GPU embedding services all
publish the same default endpoint at `http://127.0.0.1:18081/v1/embeddings`.
Run only one local embedding service at a time. `pci doctor --start` starts the
bundled database and picks the best available local embedding path.
`pci doctor --start-db` starts only the bundled pgvector database, which is the
better default when embeddings come from a remote provider or should be skipped.
Default local models download on first run.

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
- [docs/EVALUATING_VALUE.md](docs/EVALUATING_VALUE.md) — how to tell whether the install is earning its keep on your codebase
- [.env.example](.env.example) — available environment variables
- [docs/SYSTEM_PROMPT.md](docs/SYSTEM_PROMPT.md) — ready-to-paste system prompt for an agent connected to the MCP server (rationale in [SYSTEM_PROMPT_RATIONALE.md](docs/SYSTEM_PROMPT_RATIONALE.md))
- [docs/SESSION_RETROSPECTIVE_PROMPT.md](docs/SESSION_RETROSPECTIVE_PROMPT.md) — pasteable end-of-session retrospective for evaluating MCP usage
- [AGENTS.md](AGENTS.md) — instructions for assistants working on this repo
- [CONTRIBUTING.md](CONTRIBUTING.md) — contributor workflow, local checks, and guardrails

## License

MIT. See [LICENSE](LICENSE).
