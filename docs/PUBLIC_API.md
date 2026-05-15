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
| `pci-index` | Main indexing command. Requires one or more repository paths, such as `pci-index .`. |
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
| Database | `PROJECT_CODE_INTELLIGENCE_DATABASE_URL`, `PROJECT_CODE_INTELLIGENCE_DATABASE_USER`, `PROJECT_CODE_INTELLIGENCE_DATABASE_PASSWORD`, `PGVECTOR_*`, `PROJECT_CODE_INTELLIGENCE_DB_*` |
| Ingest | `PROJECT_CODE_INTELLIGENCE_COLLECTION`, `PROJECT_CODE_INTELLIGENCE_PROFILE`, `PROJECT_CODE_INTELLIGENCE_REPOS`, `PROJECT_CODE_INTELLIGENCE_MODE` |
| Embeddings | `PROJECT_CODE_INTELLIGENCE_EMBEDDING_*`, `PROJECT_CODE_INTELLIGENCE_ALLOW_REMOTE_EMBEDDING`, `PROJECT_CODE_INTELLIGENCE_PREEMBED` |
| MCP safety | `PROJECT_CODE_INTELLIGENCE_MCP_*` |
| Docker Compose | variables documented in `.env.example` |

`PROJECT_CODE_INTELLIGENCE_DATABASE_URL` is preferred for host tools.
Credentials may be embedded in the URL or supplied with
`PROJECT_CODE_INTELLIGENCE_DATABASE_USER` and
`PROJECT_CODE_INTELLIGENCE_DATABASE_PASSWORD`. The split `PGVECTOR_*` variables
remain supported for Docker Compose and compatibility.

`PROJECT_CODE_INTELLIGENCE_COLLECTION` remains a supported override, but normal
CLI/MCP use should not need it. `pci-index` infers the collection from the repo
path or workspace path, and `pci-mcp` infers it from the process working
directory when the variable is unset.

## MCP Tools

The MCP protocol surface is public. The server runs over stdio through
`pci-mcp`.

Public tool names:

| Tool | Purpose |
| --- | --- |
| `code_intel_status` | Inspect schema, snapshots, files, records, edges, and embedding state. |
| `list_code_intel_files` | List indexed source files filtered by language, role, content class, or skip status. Useful for discovering the shape of the codebase. |
| `list_code_intel_parser_failures` | List files that failed to parse during ingestion, so agents can report which parts of the codebase are missing from the index. |
| `search_code_intel_text` | Text search or filtered listing of indexed records. Default `query_mode=auto` uses PostgreSQL full-text search first, then exact multi-term fallbacks when needed. |
| `search_code_intel_semantic` | Semantic search using the configured embedding endpoint. |
| `get_code_intel_record` | Fetch one record by stable `record_id` (string), scoped to the active snapshot by default. Content is omitted unless `include_content` is true; pass `verbose=true` to retain heavy metadata fields. |
| `related_code_intel` | Follow candidate relationships by record ID or symbol. |
| `search_static_findings` | Search SARIF/static-analysis findings. |
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
- `pci-index --reset <repo>` deletes data for the selected collection and repo
  key while leaving other repos and the schema untouched.
- `pci-index --reset-all` deletes all indexed data in the configured database
  while leaving the schema in place.
- Existing embeddings are checked for model and dimension compatibility before
  resume.
- Collections are an application-level scope used by CLI and MCP behavior; they
  are not a database security boundary.
- Private database dumps, vector indexes, and generated SARIF output should not
  be published.

Direct SQL against these tables is acceptable for local inspection, but scripts
should prefer MCP tools where possible. Schema changes should be treated as
compatibility-sensitive and covered by integration smoke tests.
