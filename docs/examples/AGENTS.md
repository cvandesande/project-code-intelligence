# Agent Instructions

This repository is indexed by the `project-code-intelligence` MCP server. Use
that index as an early discovery tool when answering questions or planning code
changes.

## Code-Intelligence MCP

- Start with `code_intel_status` to inspect indexed repositories, snapshots,
  record types, languages, and static-analysis findings.
- Use `search_code_intel_text` for keyword, symbol, file, language, record-type,
  and metadata searches before broad filesystem searches.
- Use `get_code_intel_record` when a search result needs full display content.
- Use `related_code_intel` to find candidate relationships around a symbol or
  record.
- Use `search_static_findings`, `get_static_finding`, and
  `get_static_code_flow` when reviewing SARIF or static-analysis output.
- Use `search_code_intel_semantic` only when embeddings are configured for this
  repository.

The MCP index is a navigation aid, not a source of truth. Verify important
behavior against the working tree before editing or reporting findings.

## Working With This Repository

- Prefer targeted changes that match the existing architecture and style.
- Keep generated files, local database dumps, SARIF reports, embedding caches,
  model files, and private environment files out of version control.
- Do not publish code-intelligence database dumps for private repositories.
- Run the repository's documented checks before reporting completion.

## Suggested Workflow

1. Call `code_intel_status`.
2. Search the MCP index for the relevant symbols, files, or concepts.
3. Open the current files from the working tree before editing.
4. Make the smallest coherent change.
5. Run the relevant tests or checks documented by the repository.

Give a confidence level with architectural recommendations, and push back when
a proposed change would add avoidable complexity or leak private project data.
