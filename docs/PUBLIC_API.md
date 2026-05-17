# Public API

This project is still pre-1.0, so compatibility is best-effort rather than a
formal semver promise. The interfaces below are the intended public surfaces for
users, scripts, MCP clients, and profile authors.

Everything else under `project_code_intelligence` should be treated as internal
implementation detail unless it is listed here.

## Command-Line Interface

Installed console scripts are public:

| Command | Purpose |
| --- | --- |
| `pci-index` | Main indexing command. Requires one or more repository paths, such as `pci-index .`. Can emit MCP client snippets and required environment exports with `--mcp-config {env,codex,claude,opencode,vscode,copilot,cline,zed}`. |
| `pci-ingest-code` | Lower-level ingest command used by `pci-index`; useful for advanced scripting. |
| `pci-doctor` | Detects database, embedding endpoint, CPU, GPU, and NPU readiness. |
| `pci-mcp` | stdio MCP server entry point. |
| `pci-mcp-smoke` | Basic MCP status and tool smoke check. Requires one or more repo paths, such as `pci-mcp-smoke .`. |
| `pci-fastembed-server` | Small OpenAI-compatible FastEmbed server for local CPU embeddings. |
| `pci-llama-embed` | llama.cpp embedding CLI helper. |
| `pci-embedding-bench` | Embedding endpoint benchmark helper. |

The repository checkout also contains `pci-embedding-server`, a shell helper for
starting a local `llama-server`. It is useful for local experiments, but it is
not installed as a Python console script.

CLI flags shown by `--help` are public enough to use in scripts. For this
pre-1.0 project, flags may still be refined, but changes should be documented
and kept compatible where practical.

## Configuration

Environment variables are the public automation API. Prefer these over importing
internal modules from scripts.

Stable configuration groups:

| Group | Public variables |
| --- | --- |
| Database | `PROJECT_CODE_INTELLIGENCE_DATABASE_URL`, `PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH`, `PROJECT_CODE_INTELLIGENCE_DATABASE_USER`, `PROJECT_CODE_INTELLIGENCE_DATABASE_PASSWORD`, `PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL`, `PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER`, `PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD`, `PROJECT_CODE_INTELLIGENCE_POSTGRES_ADMIN_USER`, `PROJECT_CODE_INTELLIGENCE_POSTGRES_ADMIN_PASSWORD`, `PROJECT_CODE_INTELLIGENCE_DATABASE_ADMIN_USER`, `PROJECT_CODE_INTELLIGENCE_DATABASE_ADMIN_PASSWORD`, `PGVECTOR_*`, `PROJECT_CODE_INTELLIGENCE_DB_*` |
| Ingest | `PROJECT_CODE_INTELLIGENCE_COLLECTION`, `PROJECT_CODE_INTELLIGENCE_PROFILE`, `PROJECT_CODE_INTELLIGENCE_REPOS`, `PROJECT_CODE_INTELLIGENCE_MODE` |
| Embeddings | `PROJECT_CODE_INTELLIGENCE_EMBEDDING_*`, `PROJECT_CODE_INTELLIGENCE_ALLOW_REMOTE_EMBEDDING`, `PROJECT_CODE_INTELLIGENCE_PREEMBED` |
| MCP safety | `PROJECT_CODE_INTELLIGENCE_MCP_*` |
| Docker Compose | variables documented in `.env.example` |

`PROJECT_CODE_INTELLIGENCE_DATABASE_URL` is preferred for host tools.
Credentials may be embedded in the URL or supplied with
`PROJECT_CODE_INTELLIGENCE_DATABASE_USER` and
`PROJECT_CODE_INTELLIGENCE_DATABASE_PASSWORD`. If the URL contains a database
path, that database is used exactly. If the URL omits the database path, the
connection endpoint and credentials come from the URL while the database name is
inferred from the repository or workspace path. The split `PGVECTOR_*`
variables remain supported for Docker Compose and compatibility; set
`PGVECTOR_DB` only when a fixed database should override inference. `pci-index`
owns inferred project database initialization: `pci-index .` initializes before
indexing, and `pci-index --init-db .` initializes the database/schema and exits.
`pci-doctor --init-postgres` stays out of project database creation; it only
uses `PROJECT_CODE_INTELLIGENCE_POSTGRES_ADMIN_USER` and
`PROJECT_CODE_INTELLIGENCE_POSTGRES_ADMIN_PASSWORD` to create/update the
cluster-level Postgres role that `pci-index` can use through the database admin
variables below, and prepares `template1` with pgvector. When an inferred
database is missing, `pci-index` may use
`PROJECT_CODE_INTELLIGENCE_DATABASE_ADMIN_USER` and
`PROJECT_CODE_INTELLIGENCE_DATABASE_ADMIN_PASSWORD` to create the inferred
database and project-scoped RW/RO roles, then uses the scoped RW role for
schema and ingest work. When `pci-index` can derive the scoped RO password, it
prints an `Export for pci-mcp (RO)` block after `pci-index --init-db` and
ordinary index runs; `--mcp-config codex`, `--mcp-config claude`,
`--mcp-config opencode`, and `--mcp-config vscode`/`copilot` print
credential-free project config snippets followed by the read-only environment
values those snippets need. `--mcp-config zed` prints a project-scoped
`.zed/settings.json` snippet with read-only database values embedded, because
Zed does not document environment-variable interpolation for MCP `env` values.
`--mcp-config cline` prints a user-scoped GUI snippet with read-only database
values embedded, matching Cline's local MCP settings shape. `pci-mcp` prefers
`PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL`,
`PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER`, and
`PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD` when set, and otherwise falls
back to the generic database variables. Set
`PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH` for MCP clients or custom
launchers that run outside the indexed repo/workspace but still need the same
inferred database name. Scoped role passwords are stable for the same admin
password and inferred database name. When database admin variables are set for an
inferred database, generated scoped roles override URL-embedded credentials;
set separate runtime user/password variables only when explicit credentials
should win. The pci-index admin role must be able to create databases, create
roles, and use pgvector inherited from `template1`. The one-time
`pci-doctor --init-postgres` bootstrap usually needs Postgres admin credentials
such as a local `postgres` superuser because pgvector is not trusted. By
default, `pci-doctor --init-postgres` writes only the generated non-superuser
`pci_index_admin` connection values for `pci-index` to
`${XDG_CONFIG_HOME:-~/.config}/project-code-intelligence/pci-index.env` with
`0600` permissions; `pci-index` loads that file when the corresponding
environment variables are unset and reports the loaded path in normal text
mode. Use `--no-write-config` to skip the file write. Without database admin
variables, `pci-index` uses the normal configured credentials and does not
generate separate RW/RO roles.

`PROJECT_CODE_INTELLIGENCE_COLLECTION` remains a supported override, but normal
CLI/MCP use should not need it. Prefer `pci-index --collection NAME` for index
runs; an inherited collection environment variable is ignored by `pci-index`
unless `PROJECT_CODE_INTELLIGENCE_ALLOW_COLLECTION_OVERRIDE=1` is also set.
`pci-index` infers the collection from the repo path or workspace path, and
`pci-mcp` infers it from the process working directory when the variable is
unset. In the same default path, `pci-index` and `pci-mcp` infer a project
database name from that repository or workspace scope.

## MCP Tools

The MCP protocol surface is public. The server runs over stdio through
`pci-mcp`.

Public tool names:

| Tool | Purpose |
| --- | --- |
| `code_intel_status` | First call for non-trivial code discovery; inspect schema, current snapshot freshness, repo/file/record/edge counts, and compact `queryability` counts. Compact scoped output puts `collection`/`repo` at top level instead of repeating them in every row and omits duplicate `head_commit` values; stale, dirty, unverifiable, or missing repo scopes include `warnings`, and missing repo scopes include `found:false`. |
| `list_code_intel_files` | List indexed source files filtered by language, role, content class, or skip status. Useful for discovering the shape of the codebase. |
| `list_code_intel_parser_failures` | List files that failed to parse during ingestion, so agents can report which parts of the codebase are missing from the index. |
| `search_code_intel_text` | Exact indexed search for identifiers, symbols, filenames, config keys, known strings, or filtered record listing. Default `query_mode=auto` uses exact term matching for identifier-like single tokens, otherwise PostgreSQL full-text search first, then exact multi-term fallbacks when needed. `mode=enumerate` lists records by filters in deterministic path/line/record order and cannot be combined with a non-empty `query`; empty optional strings are treated as omitted. Fallback, regex-looking tokenized queries, and empty-scope responses include `warnings`; broad text search excludes `security_pattern` records unless `record_type` is set or the query clearly asks for security findings. |
| `search_code_intel_semantic` | Concept search using the configured embedding endpoint when exact identifiers are unknown. Use text search for symbols. Ranking is vector distance with a generic lexical boost for query terms that also occur in symbols, titles, paths, IDs, summaries, or content, plus source-role preference and non-source/generated penalties unless the caller narrows the role/path or asks for tests/docs. Broad results are diversified by `parent_record_id` by default; pass `diversify=false` to preserve raw rank order. Empty-scope and non-embedded filter responses include `warnings`; broad semantic search excludes `security_pattern` records unless `record_type` is set or the query clearly asks for security findings. |
| `get_code_intel_record` | Fetch one record by stable `record_id`. Scoped to the active snapshot by default. Content and metadata are omitted unless `include_content` or `include_metadata` is true; pass `verbose=true` to retain full diagnostic fields. |
| `get_code_intel_records` | Fetch many records by stable `record_ids`; found records preserve input order and missing IDs are reported separately. Scoped to the active snapshot by default. Content and metadata are omitted unless `include_content` or `include_metadata` is true; pass `verbose=true` to retain full diagnostic fields. |
| `related_code_intel` | Follow heuristic caller/callee and related-symbol candidates by record ID or symbol. Unresolved heuristic targets are hidden by default; pass `include_unresolved=true` to inspect them. Compact results include symbols, record IDs, paths, line ranges, `direction`, `target_resolved`, `target_kind`, and `confidence_kind`; heuristic candidates include `warnings`, and missing `record_id` lookups include `found:false`. Verify important relationships in source. |
| `search_static_findings` | Search SARIF/static-analysis findings. `source_path` and `source_path_prefix` accept repo-relative paths using the same normalization as code/file tools. Empty responses include `static_runs_found` when the server can distinguish “no matching SARIF/static run” from “a run exists with no matching findings.” |
| `get_static_finding` | Fetch one SARIF/static-analysis finding with compact rule and location details. Pass `include_code_flows`, `include_raw`, or `include_run_metadata` for larger diagnostic payloads. |
| `get_static_code_flow` | Fetch ordered code-flow steps for one finding. |

Tool schemas are defined in `project_code_intelligence.mcp.tool_catalog`.
Clients should use `tools/list` instead of assuming schemas from documentation.

## Python Imports

The Python API is intentionally small. These imports are public:

```python
from project_code_intelligence.code_profiles import CodeIntelProfile, GenericProfile, load_profile
from project_code_intelligence.code_profiles.base import ProfileRecord, SecurityPattern
from project_code_intelligence.models import IntelEdge, IntelFile, IntelRecord, JsonObject
```

Profile authors should subclass `CodeIntelProfile` and select their profile with
`PROJECT_CODE_INTELLIGENCE_PROFILE=package.module:ProfileClass`.

The following modules are compatibility facades. They are importable, but they
mostly exist so tests, profile code, and future callers have stable boundaries:

```python
import project_code_intelligence.doctor
import project_code_intelligence.embeddings
import project_code_intelligence.parsers
import project_code_intelligence.sarif
import project_code_intelligence.storage
```

The subpackages below are internal. They may change as the implementation is
split further:

```text
project_code_intelligence.embedding.*
project_code_intelligence.mcp.*
project_code_intelligence.parsers.*
project_code_intelligence.sarif.*
project_code_intelligence.storage.*
project_code_intelligence.doctor.*
```

An exception is `project_code_intelligence.mcp.tool_catalog`: MCP tool names and
schemas are public through `tools/list`, and this module is the local source of
those schemas.

## Database Schema

The database tables are an operational compatibility surface, not a general
Python API.

Public expectations:

- Tables are namespaced as `project_code_intel_*`.
- `pci-index --reset <repo>` drops the inferred PCI-managed database for that
  repository/workspace scope. Explicit database URLs are not dropped.
- When no database name is configured, host tools infer a PostgreSQL database
  name from the repository or workspace path. `pci-index` may create or reset
  that inferred database; `pci-mcp` only connects to it read-only.
- Existing embeddings are checked for model and dimension compatibility before
  resume.
- Collections are an application-level scope used by CLI and MCP behavior; they
  are not a database security boundary.
- Private database dumps, vector indexes, and generated SARIF output should not
  be published.

Direct SQL against these tables is acceptable for local inspection, but scripts
should prefer MCP tools where possible. Schema changes should be treated as
compatibility-sensitive and covered by integration smoke tests.
