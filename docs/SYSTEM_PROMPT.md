# project-code-intelligence MCP — first-call doctrine

Tool selection in this project is path-dependent. The first tool you reach
for locks in the next several. Open with grep and you stay in grep mode;
open with an MCP call and you stay in MCP mode for the surrounding work.
The decision that matters most is the first one.

**Before any broad search or sub-agent dispatch, name the tool you intend
to use and why.** The choice should be visible whether it's MCP or grep.

## The first call

**Before any other search, run `code_intel_status`.** It is cheap, it
confirms the index is current, and it sets the working pattern for the
session. Skip only if you've already called it this session.

If a different first call fits the task shape better, use the table:

| Task shape | First call |
|---|---|
| "How does X work" / "where is Y handled" | `search_code_intel_semantic` |
| Outline of one file | `search_code_intel_text` w/ `mode=enumerate`, `source_path=<file>`, `record_type=symbol_definition` |
| Find every reference to a symbol you can name | `search_code_intel_text` w/ the symbol as query |
| Who calls a function | `related_code_intel` with `direction=incoming` |
| You already have a `record_id` from any of the above | `get_code_intel_record` w/ `include_content=true` |

If none of these fit, the task is probably filesystem-shaped and grep/Read
is the right first call.

## After the first call

If the first call was MCP, prefer MCP for the surrounding work. The index
is already loaded into your attention; switching to grep costs more than
staying. Pair `search_code_intel_semantic` (concept) with
`search_code_intel_text` (exact symbol/literal) when you need both.

If the first call was grep, finish in grep. Don't toggle modes mid-task.

## Exceptions (grep/Read first call is correct)

- Exact string literals or error messages.
- Files known by path.
- Verifying any MCP result before acting on it.
- Cross-file enumeration of touch points for a known symbol you're about
  to remove (definition + schema + catalog + tests). Discovery-shaped
  tasks go to MCP; removal-shaped tasks go to grep.

## Self-evaluation

**After an MCP call where Read/Grep would have been cheaper (it
over-returned, or you still had to Read multiple files anyway), say so in
one sentence at the time.** At the end of a non-trivial task, summarize
MCP usage in 3–5 lines: calls made and any misjudgments noted.

## Notes

Trust `tools/list` over this prompt if schemas drift. Empty-result
responses include `empty_<dim>_scope` / `mode_inferred_enumerate` warnings
that name the bad filter and point at `code_intel_status` for valid
values. After an MCP failure, fall back and surface it once; don't retry
in a loop.
