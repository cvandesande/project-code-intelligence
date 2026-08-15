# Public API

This project is still pre-1.0, so compatibility is best-effort rather than a
formal semver promise. The interfaces below are the intended public surfaces for
users, scripts, MCP clients, and profile authors.

Everything else under `project_code_intelligence` should be treated as internal
implementation detail unless it is listed here.

## Command-Line Interface

A single console script, `pci`, is installed. Its subcommands are public:

| Command | Purpose |
| --- | --- |
| `pci index` | Main indexing command. Requires one or more repository paths, such as `pci index .`. Can emit MCP client snippets and required environment exports with `--mcp-config {env,codex,claude,opencode,vscode,copilot,cline,zed}`. Low-level ingest flags are reachable via `pci index ... -- <flags>`. |
| `pci check <sarif files...>` | SARIF regression ratchet. Ingests SARIF (any producer) and fails (exit 1) only on findings that are new or escalated in level since the current branch's frozen baseline; exit 0 otherwise. `pci check --baseline <sarif files...>` (re)freezes the baseline for the current branch. Baseline identity is (collection, repo, branch), same as snapshots. |
| `pci audit --gate` | Adds a gate section to `pci audit`: diffs each snapshot's already-ingested static findings against its `pci check` branch baseline (same identity, same diff logic) and exits 1 if any repo has a new or worsened finding. |
| `pci audit --full-triage --init-triage` | Creates or extends `.pci/audit-triage.json` with language-agnostic, collection/repository-scoped redundancy candidate IDs. Use `--candidate ID --status open` or `--candidate ID --status dismissed --reason TEXT` to record a disposition. During a full triage sweep, a unique overlap of at least two members preserves the disposition when a group expands or contracts; ambiguous splits/merges remain new candidates. Bounded reports neither reconcile against omitted candidates nor infer that saved candidates are fixed. Saved candidates absent from a later `--full-triage` sweep are reported as fixed. Writes are locked and atomic; the shared state records timestamps and the generating PCI version. Candidate discovery still depends on PCI's parser and call-edge coverage for each language. `--triage-file PATH` selects another state file. |
| `pci rulepack list` | Lists `.pci/rulepacks/<name>/` directories found under the current directory, with rule counts per tier (1=mechanical, 2=metric-gateable, 3=LLM-judge). |
| `pci rulepack validate` | Validates discovered rulepacks: unknown tier, duplicate rule IDs, a Tier-3 rule with no matching rubric entry, and a dangling Tier-1/2 producer-config path. Exits 1 on any error; prints file/field/reason for each. Rulepacks are files in the target repo, not indexed state -- this command touches no database. |
| `pci doctor` | Detects database, embedding endpoint, CPU, GPU, and NPU readiness. Also starts/stops bundled local services with `--start`, `--start-db`, `--start-embedding`, `--stop`, and `--clean`. |
| `pci mcp` | stdio MCP server entry point. `pci mcp install --target CLIENT` installs configuration for Codex, Claude, OpenCode, Pi, VS Code/Copilot, Cline, or Zed; add `--uninstall` to remove only PCI's server entry. |
| `pci smoke` | Basic MCP status and tool smoke check. Requires one or more repo paths, such as `pci smoke .`. |
| `pci embed fastembed` | Small OpenAI-compatible FastEmbed server for local CPU embeddings. |
| `pci embed apple` | OpenAI-compatible embedding server backed by Apple MLX (Apple Silicon only). Writes a PID file so `pci doctor --stop` can terminate it. |
| `pci embed llama` | llama.cpp embedding CLI helper. |
| `pci embed bench` | Embedding endpoint benchmark helper. |

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
| Database | `PCI_DATABASE_URL`, `PCI_DATABASE_SCOPE_PATH`, `PCI_DATABASE_USER`, `PCI_DATABASE_PASSWORD`, `PCI_MCP_DATABASE_URL`, `PCI_MCP_DATABASE_USER`, `PCI_MCP_DATABASE_PASSWORD`, `PCI_POSTGRES_ADMIN_USER`, `PCI_POSTGRES_ADMIN_PASSWORD`, `PCI_DATABASE_ADMIN_USER`, `PCI_DATABASE_ADMIN_PASSWORD`, `PCI_PG_*`, `PCI_MCP_PG_*`, `PCI_DB_*` |
| Ingest | `PCI_COLLECTION`, `PCI_PROFILE`, `PCI_REPOS`, `PCI_MODE` |
| Embeddings | `PCI_EMBEDDING_*`, `PCI_ALLOW_REMOTE_EMBEDDING`, `PCI_PREEMBED` |
| MCP safety | `PCI_MCP_*` |
| Docker Compose | variables documented in `.env.example` |

`PCI_DATABASE_URL` is preferred for host tools.
Credentials may be embedded in the URL or supplied with
`PCI_DATABASE_USER` and
`PCI_DATABASE_PASSWORD`. If the URL contains a database
path, that database is used exactly. If the URL omits the database path, the
connection endpoint and credentials come from the URL while the database name is
inferred from the repository or workspace path. The split `PCI_PG_*`
variables remain supported for Docker Compose and compatibility; set
`PCI_PG_DB` only when a fixed database should override inference. `pci index`
owns inferred project database initialization: `pci index .` initializes before
indexing, and `pci index --init-db .` initializes the database/schema and exits.
`pci doctor --init-postgres` stays out of project database creation; it only
uses `PCI_POSTGRES_ADMIN_USER` and
`PCI_POSTGRES_ADMIN_PASSWORD` to create/update the
cluster-level Postgres role that `pci index` can use through the database admin
variables below, and prepares `template1` with pgvector. When an inferred
database is missing, `pci index` may use
`PCI_DATABASE_ADMIN_USER` and
`PCI_DATABASE_ADMIN_PASSWORD` to create the inferred
database and project-scoped RW/RO roles, then uses the scoped RW role for
schema and ingest work. When `pci index` can derive the scoped RO password, it
prints an `Export for pci mcp (RO)` block after `pci index --init-db` and
ordinary index runs; every `--mcp-config` target writes the derived read-only
values to a project-keyed file under the private user config directory (mode
`0600`) and prints a credential-free config that launches
`pci mcp --scope PATH`. `pci mcp install --target TARGET` installs or updates
that config for Codex, Claude Code, OpenCode, Pi, VS Code/Copilot, Cline, or Zed;
add `--uninstall` to remove only PCI's server entry. `pci mcp` prefers
`PCI_MCP_DATABASE_URL`,
`PCI_MCP_DATABASE_USER`, and
`PCI_MCP_DATABASE_PASSWORD` when set, and otherwise falls
back to the generic database variables. `PCI_MCP_PG_USER` and `PCI_MCP_PG_PASS`
are compatibility aliases that override the split `PCI_PG_USER` /
`PCI_PG_PASS` values for the MCP connection; prefer the
`PCI_MCP_DATABASE_*` names in new setups. Set
`PCI_DATABASE_SCOPE_PATH` for MCP clients or custom
launchers that run outside the indexed repo/workspace but still need the same
inferred database name. Scoped role passwords are stable for the same admin
password and inferred database name. When database admin variables are set for an
inferred database, generated scoped roles override URL-embedded credentials;
set separate runtime user/password variables only when explicit credentials
should win. The pci-index admin role must be able to create databases, create
roles, and use pgvector inherited from `template1`. The one-time
`pci doctor --init-postgres` bootstrap usually needs Postgres admin credentials
such as a local `postgres` superuser because pgvector is not trusted. By
default, `pci doctor --init-postgres` writes only the generated non-superuser
`pci_index_admin` connection values for `pci index` to
`${XDG_CONFIG_HOME:-~/.config}/project-code-intelligence/pci-index.env` with
`0600` permissions; `pci index` loads that file when the corresponding
environment variables are unset and reports the loaded path in normal text
mode. Use `--no-write-config` to skip the file write. When database admin
variables are unset, `pci index` promotes the configured writer credentials to
the effective admin and still creates per-project RW/RO roles (with
deterministic HMAC-derived passwords keyed on the writer password). The
bundled local pgvector container ships with a writer that has CREATEROLE, so
this works out of the box; for a remote Postgres whose writer lacks CREATEROLE
the role-creation SQL fails with guidance to run `pci doctor --init-postgres`.

For the bundled local database, `pci doctor --start-db` starts only the
pgvector container. `pci doctor --start` starts the database plus the best local
embedding service it can run on the host. Use `--start-db` when embeddings are
remote or intentionally skipped.

Installed packages materialize bundled Compose assets into the user cache. Set
`PCI_COMPOSE_FILE` to use a customized Compose file without editing the
installed package or Nix store path. `pci doctor --clean` removes generated
Compose cache files in addition to stopping local services and removing the
bundled database volume.

`PCI_COLLECTION` remains a supported override, but normal
CLI/MCP use should not need it. Prefer `pci index --collection NAME` for index
runs; an inherited collection environment variable is ignored by `pci index`
unless `PCI_ALLOW_COLLECTION_OVERRIDE=1` is also set.
`pci index` infers the collection from the repo path or workspace path, and
`pci mcp` infers it from the process working directory when the variable is
unset. In the same default path, `pci index` and `pci mcp` infer a project
database name from that repository or workspace scope.

## MCP Tools

The MCP protocol surface is public. The server runs over stdio through
`pci mcp`.

Public tool names:

Optional `null` values and empty optional strings are treated as omitted at the
transport boundary. Boolean `false` is preserved because it is an active filter.

Every record, edge, and file shape carries `repo_path` (and edges carry
`source_repo_path` / `target_repo_path`): the path relative to the repo root,
suitable for passing directly to a file-reading tool when the consumer's cwd is
the repo root. `source_path` keeps the stored workspace-relative form (with the
repo prefix for multi-repo collections); the two coincide when the row's repo is
"." or absent.

| Tool | Purpose |
| --- | --- |
| `code_intel_status` | First call for non-trivial code discovery; inspect schema, current snapshot freshness, repo/file/record/edge counts, and compact `queryability` counts (record types, edge types, languages, file roles, content classes; pass `include_queryability=true` for the full lists plus `configured_embed_record_type_count`). Compact `queryability` surfaces `empty_embed_record_type_count` only when non-zero, since a zero value carries no action. Compact scoped output puts `collection`/`repo` at top level instead of repeating them in every row and omits duplicate `head_commit` values; stale, dirty, unverifiable, missing repo, or unknown `snapshot_id` scopes include `warnings`, and missing repo scopes include `found:false`. Pass `include_runtime=true` for server executable/module identity, installed VCS commit when package metadata provides it, and redacted DB identity. Pass `include_active_runs=true` (not implied by `verbose`) for only currently running `active_runs` (phase, progress metrics, per-repo incremental/full mode with full-fallback reason, `heartbeat_age_seconds`); when a run is in flight the response carries an `index_run_active` warning. Databases created before the ledger table return `active_runs: []`. A snapshot whose commit matches local HEAD but whose stamped `branch` differs from the checkout's current branch reports `head_status: "stale"` with `head_status_reason: "branch_mismatch"` -- same commit, different branch, so the index may not reflect the code on the branch you are on. |
| `list_code_intel_files` | List indexed source files filtered by language, role, content class, or skip status. Includes record-backed paths when searchable records exist without a file inventory row. Boolean filters are active when set to `false`; omit unset booleans. Useful for discovering the shape of the codebase. |
| `search_code_intel_text` | Exact indexed search for identifiers, symbols, filenames, config keys, known strings, or filtered record listing. Default `query_mode=auto` uses exact term matching for identifier-like single tokens, otherwise PostgreSQL full-text search first, then exact multi-term fallbacks when needed. `mode=enumerate` lists records by filters in deterministic path/line/record order and cannot be combined with a non-empty `query`; omitting `mode` with an empty query falls through to enumeration and emits a `mode_inferred_enumerate` warning. Empty optional strings are treated as omitted. Fallback, regex-looking tokenized queries, and empty-scope responses include `warnings` (`empty_repo_scope`, `empty_snapshot_scope`, and `empty_<dim>_scope` for `language`/`file_role`/`record_type`/`content_class`); broad text search excludes `security_pattern` records unless `record_type` is set or the query clearly asks for security findings. |
| `search_code_intel_semantic` | Concept search using the configured embedding endpoint when exact identifiers are unknown. Use text search for symbols. Ranking is vector distance with a generic lexical boost for query terms that also occur in symbols, titles, paths, IDs, summaries, or content, plus source-role preference and non-source/generated penalties unless the caller narrows the role/path or asks for tests/docs. Each result carries `similarity` (cosine similarity in `[0, 1]`, higher = closer match, parallel to text search's `rank`) so consumers can self-judge confidence without a follow-up call; `verbose=true` also surfaces the raw `distance` (`1 - similarity`). Broad results are diversified by `parent_record_id` by default; pass `diversify=false` to preserve raw rank order. Empty-scope and non-embedded filter responses include `warnings`; broad semantic search excludes `security_pattern` records unless `record_type` is set or the query clearly asks for security findings. |
| `get_code_intel_record` | Fetch one record by stable `record_id`, or many by stable `record_ids` (pass exactly one); batch results preserve input order and missing IDs are reported separately. Scoped to the active snapshot by default. Content and metadata are omitted unless `include_content` or `include_metadata` is true; pass `verbose=true` to retain full diagnostic fields. |
| `related_code_intel` | Follow heuristic caller/callee and related-symbol candidates by exactly one of `record_id` or `symbol` (supplying both errors). Default `direction=any` runs incoming and outgoing in parallel and interleaves the results so neither side starves the other within the limit; pass `direction=incoming` or `direction=outgoing` to focus on one side. Unresolved heuristic targets are hidden by default; pass `include_unresolved=true` to inspect them. Compact results include symbols, record IDs, paths, line ranges, `direction`, `target_resolved`, `target_kind`, and `confidence_kind`; heuristic candidates include `warnings`, and missing `record_id` lookups include `found:false`. Verify important relationships in source. |
| `blast_radius` | Impact of removing or refactoring a definition, resolved by `symbol` or `source_path`+`line`. Returns per-symbol bundles with resolved `callers` (each flagged `is_test` / `at_module_level`), `inbound_count`, `name_reference_count` (text-level backstop for dynamic dispatch), `covered_by_tests`, `is_entrypoint`, `wired_at_module_level`, `looks_orphaned`, semantic `neighbors` (default 3, `neighbors=0` to skip), and a `staleness` block. Evidence, not a verdict — the index is keyed on the git-index blob and usually predates the working tree, so `looks_orphaned` / `found:false` are prompts to verify live callers in source, not proof of deadness. No match returns `found:false` with a `symbol_not_found` warning. |
| `find_redundancy` | Groups of functions repeating one call-shape motif, ranked within actionable/non-actionable tiers by `coherence` (`max(semantic_similarity, text_similarity)`; provisionally calibrated on a 10-group labeled sample, this repo), then by `net_value` (redundancy removed minus abstraction cost) as tiebreak. Each group carries `members`, `common_shape`, `graph_similarity`, `semantic_similarity`, `text_similarity` (average pairwise body-text similarity), `max_text_similarity` (max pairwise body-text similarity; the average dilutes a byte-identical pair inside a larger group — measured on this repo, only an exact-text pair at >= 0.99 is a near-certain duplicate), `coherence`, `net_value`, `typed_variants` (bodies are near-identical and members' return types differ beyond optionality — e.g. `optional_int`/`optional_bool`; when true, `recommendation` is forced to `leave-as-is` since collapsing would lose type-checker precision), a `recommendation` (`worth-collapsing`, `parameterize-carefully`, `leave-as-is`), and an `evidence` block with `redundancy_removed`, `abstraction_cost`, `residual_roles`, `shared_helper` (internal callee(s) every member shares — evidence only; a shared subroutine is not proof the motif is abstracted, so it never changes `recommendation` or the rank), `estimated_loc_removed`, and `low_coherence`. Scope with `source_path_prefix` (repo-relative or repo-prefixed; a group matches when any member is under it) — the filter is applied before ranking, so `limit` is spent on groups touching that area. Shape is inferred from heuristic `call_candidate` edges, so results are evidence, not a verdict: verify in source before collapsing anything. A scope matching no snapshot returns `found:false` with an `empty_repo_scope` warning. Pass `branch` to restrict to that branch's snapshot; omit it to use the newest snapshot per repo regardless of branch (previous behavior). |
| `search_static_findings` | Search SARIF/static-analysis findings. `source_path` and `source_path_prefix` accept repo-relative paths using the same normalization as code/file tools. Empty responses include `static_runs_found` when the server can distinguish “no matching SARIF/static run” from “a run exists with no matching findings.” |
| `get_static_finding` | Fetch one SARIF/static-analysis finding with compact rule and location details. Pass `include_code_flows`, `include_raw`, or `include_run_metadata` for larger diagnostic payloads. |

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
`PCI_PROFILE=package.module:ProfileClass`.

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
- `pci index --reset <repo>` drops the inferred PCI-managed database for that
  repository/workspace scope. Explicit database URLs are not dropped.
- When no database name is configured, host tools infer a PostgreSQL database
  name from the repository or workspace path. `pci index` may create or reset
  that inferred database; `pci mcp` only connects to it read-only.
- Existing embeddings are checked for model and dimension compatibility before
  resume.
- Collections are an application-level scope used by CLI and MCP behavior; they
  are not a database security boundary.
- Private database dumps, vector indexes, and generated SARIF output should not
  be published.

Direct SQL against these tables is acceptable for local inspection, but scripts
should prefer MCP tools where possible. Schema changes should be treated as
compatibility-sensitive and covered by integration smoke tests.
