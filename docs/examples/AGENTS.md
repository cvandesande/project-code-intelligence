# Code Intelligence Guide for AI Assistants

This project has a `project-code-intelligence` MCP server attached. For non-trivial code discovery, use it before broad rg/find or speculative file reads. Use rg/direct reads for known small files and final verification.

## The actual problem it solves

When you work through a codebase without tooling, you read files to find what you need. The waste isn't in the reading itself — it's in everything you load that turns out not to matter. You open a file because it might have what you're looking for. Sometimes it does, often it doesn't. That speculative loading is where token cost quietly accumulates.

The index inverts this. Instead of reading to find, you query to decide what's worth reading. The expensive part (loading file content into context) becomes the last step, not the first.

## Where this actually pays off

**Large and generated files.** Every serious codebase has files that exist to be consumed by machines, not read by people — protobuf-generated code, auto-generated clients, ORM models, build artifacts that got committed. These files can be hundreds of kilobytes. You rarely need more than one function from them. Without the index, you either read a huge slice or run multiple greps and still end up loading more than you need. With the index, you query for the symbol and get a 20-line snippet. That's not a marginal improvement on a single lookup.

**Not knowing what you're looking for.** Grep requires you to already know the word. Semantic search doesn't. "Find code that handles connection retry backoff" or "where does TLS configuration get assembled" are questions grep can't answer without you already knowing the answer. The semantic search isn't perfect, but it replaces three or four speculative file reads with one targeted query.

**Getting oriented.** Understanding the shape of an unfamiliar codebase — what languages, what's generated vs. hand-written, what's tested, where the entry points are — normally costs a lot of exploratory reading. `code_intel_status` and `list_code_intel_files` with filters make this cheap. Use them at the start of any non-trivial task.

**Finding callers.** `related_code_intel` can answer "what calls this function?" across the whole codebase in one round-trip, with file paths, line numbers, and snippets. The alternative is grep plus reading each match in context.

## How to use the snippet field

The snippet in search results is not the answer. It's what you use to decide if you need the answer. "Is this the right function?" can be resolved from a snippet. "What exactly does this do, and does it handle the edge case I care about?" still requires reading the actual code. Don't skip the read when you actually need to understand something — use the snippet to avoid reads when you don't.

## What it won't do well

Call graph edges are heuristic — they're inferred from symbol co-occurrence, not proven by a type checker or linker. They're useful for navigating to candidates, not for asserting definitive caller/callee relationships. Treat them as "probably calls" and verify in source when correctness matters.

Text search falls back through multiple strategies when full-text search finds nothing. The results are still useful, but the relevance ranking becomes less reliable. If text search returns noise, try semantic search instead — they use different mechanisms and one often succeeds where the other struggles.

## The rule

Before reading a file, ask whether the index can tell you if it's worth reading. Before grepping speculatively across the codebase, ask whether a semantic query would get you there faster. The index doesn't replace reading code — it replaces the part where you're not sure what to read yet.
