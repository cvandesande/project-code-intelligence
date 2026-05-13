# Agent Instructions

This repository is indexed by the `project-code-intelligence` MCP server. Use
that index as the first move for any task that involves understanding,
searching, or reasoning about the codebase. A single MCP call typically
replaces dozens of file reads or `grep` invocations and surfaces
relationships (edges) that text search alone cannot.

The MCP index is a navigation aid, not a source of truth. Verify important
behavior against the working tree before editing or reporting findings.

## Code-Intelligence MCP Tools

Discovery and navigation:

- **`code_intel_status`** — Start here. Inspect indexed repositories, snapshots,
  record types, languages, embedding coverage, and static-analysis findings.
  The `embedded_records` count tells you whether semantic search is usable
  for this repo (must be > 0).
- **`list_code_intel_files`** — Enumerate indexed source files filtered by
  `language`, `file_role`, `content_class`, `is_test`, `is_doc`,
  `is_generated`, `is_vendor`, `is_source`, `is_build`, `is_config`, or
  `only_skipped`. Best way to learn the shape of the codebase before
  searching.
- **`list_code_intel_parser_failures`** — Files the indexer could not parse.
  Call this before claiming you have reviewed "all of X" so you can report
  honestly what is and is not in the index.

Search:

- **`search_code_intel_text`** — Keyword/symbol/file/language/record-type/
  metadata search backed by PostgreSQL full-text search. Use before
  filesystem search. Supports `parent_record_id` for class→methods or
  function→inner-symbol navigation.
- **`search_code_intel_semantic`** — Embedding-based similarity. Only useful
  when `code_intel_status` reports `embedded_records > 0` for the repo.
  Supports the same filters as text search, including `parent_record_id`.

Record and relationship lookup:

- **`get_code_intel_record`** — Full record including display content when a
  search result needs more than the summary.
- **`related_code_intel`** — Graph edges around a `record_id` or `symbol`,
  with the source and target record bodies joined into the result so a
  single call resolves both ends of every edge. Filter by `edge_type`
  (`imports`, `calls`, `inherits`, etc.) when you only want one kind of
  relationship.

Static analysis (when SARIF has been ingested):

- **`search_static_findings`**, **`get_static_finding`**,
  **`get_static_code_flow`** — SARIF/static-analysis review.

Repo names are the stable identifiers the MCP returns (`openwrt`,
`project-code-intelligence`, etc.), not filesystem paths. The same
convention applies to `pci-mcp-smoke <path>`, which resolves the path to a
basename and queries by repo.

## Common Tasks

| Task                                                  | Tool & arguments                                                              |
| ----------------------------------------------------- | ----------------------------------------------------------------------------- |
| Find where symbol `X` is used                         | `related_code_intel(symbol="X", edge_type="calls")`                           |
| List a symbol's outgoing references                   | `related_code_intel(record_id="…", edge_type="calls")`                        |
| What does this module import?                         | `related_code_intel(record_id="…", edge_type="imports")`                      |
| Enumerate all test files                              | `list_code_intel_files(is_test=true)`                                         |
| Enumerate all Python sources, excluding generated     | `list_code_intel_files(language="python", is_generated=false)`                |
| What couldn't be parsed?                              | `list_code_intel_parser_failures()`                                           |
| All methods of a class                                | `search_code_intel_text(parent_record_id="<class_record_id>")`                |
| Find a symbol by name                                 | `search_code_intel_text(symbol="my_function")`                                |
| Conceptual search (e.g. "auth token refresh")         | `search_code_intel_semantic(query="…")` after confirming embeddings are on   |
| Records skipped during ingestion                      | `list_code_intel_files(only_skipped=true)`                                    |

## Working With This Repository

- Prefer targeted changes that match the existing architecture and style.
- Keep generated files, local database dumps, SARIF reports, embedding caches,
  model files, and private environment files out of version control.
- Do not publish code-intelligence database dumps for private repositories.
- Run the repository's documented checks before reporting completion.

## Suggested Workflow

1. Call `code_intel_status` for the repo you're working in. Note
   `embedded_records` (semantic search is only useful when > 0) and any
   non-empty `parser_failures` count.
2. If `parser_failures > 0`, call `list_code_intel_parser_failures` and keep
   the list in mind so you can disclose coverage gaps later.
3. Use `list_code_intel_files` and the search tools to locate the relevant
   records, files, or symbols. Prefer the most specific filter that fits the
   question.
4. Use `related_code_intel` to navigate the graph instead of re-searching by
   string — it returns both ends of each edge inline.
5. Open the current files from the working tree before editing. The index
   may be stale relative to uncommitted changes.
6. Make the smallest coherent change.
7. Run the relevant tests or checks documented by the repository.
8. When reporting "I've reviewed all of X", explicitly mention any parser
   failures or skipped files that fall inside X's scope.

Give a confidence level with architectural recommendations, and push back when
a proposed change would add avoidable complexity or leak private project data.
