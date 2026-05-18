# project-code-intelligence MCP

Indexes the source tree (files, records, edges, embeddings, SARIF) into
Postgres. For discovery questions — "where is X used?", "what calls Y?",
"what's in this codebase?", "find code that does Z" — the index is the
default; Read/Grep are the fallback. For known-path reads of small files,
the reverse is true.

**Before any broad file search or sub-agent dispatch, name the MCP query
you'd run first (or briefly say why Read/Grep is the right call instead).**
The choice should be visible whether you end up calling MCP or not.

**After an MCP call where Read/Grep would have been cheaper (it
over-returned, or you still had to Read multiple files anyway), say so
in one sentence at the time.** At the end of a non-trivial task,
summarize MCP usage in 3–5 lines: calls made and any misjudgments noted.

## Default to MCP for:

| Task | Tool |
|---|---|
| "Where is `Foo` defined/used?" | `search_code_intel_text query="Foo"` |
| "What's queryable here?" (record types, languages, file roles…) | `code_intel_status` (with `include_queryability=true` for full lists) |
| "List files matching a filter" | `list_code_intel_files repo=… file_role=… language=…` |
| "What calls/uses `Bar.method`?" | `related_code_intel symbol="Bar.method" direction=incoming` |
| "Find code by concept, no identifier" | `search_code_intel_semantic query="…"` |
| "Full text of record `<id>`" | `get_code_intel_record record_id=<id> include_content=true` |
| "Triage SARIF findings" | `search_static_findings`, `get_static_finding`, `get_static_code_flow` |

Start non-trivial sessions with `code_intel_status`: cheap, confirms the
index matches HEAD, lists queryable filter values.

## Fall back to Read/Grep when:

- The path is known and the file is small (Read is one cheap op).
- The index is stale: post-edit verification, uncommitted changes, or
  `snapshot_dirty` warnings — MCP's view is a snapshot.

After an MCP failure, fall back and surface it once; don't retry in a loop.

Trust `tools/list` over this prompt if schemas drift. Empty-result responses
include `empty_<dim>_scope` / `mode_inferred_enumerate` warnings that name
the bad filter and point at `code_intel_status` for valid values.
