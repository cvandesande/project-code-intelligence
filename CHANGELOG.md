# Changelog

Notable changes to this project. The project has no release tags yet and tracks
one version in `pyproject.toml`, so work accumulates under `Unreleased` until
the first tagged release. Small fixes are not listed here; they are visible in
the commit history.

## Unreleased

### Added

- **`blast_radius` reports callers of class-qualified definitions.** Before this
  change a method stored as `Class.method` had no inbound call edge at all: call
  sites record the callee name, so name lookup never reached the qualified
  definition. Measured on this repository, 1,491 qualified definitions had zero
  inbound edges; now 101 of 1,510 have at least one real caller, and
  `DatabaseSettings.from_env` went from 0 reported callers to 38. A callee name
  that names its receiver binds exactly, so two classes with the same method
  name no longer block each other. See `docs/PUBLIC_API.md` for the contract.
- **`low_semantic_similarity` warning on `search_code_intel_semantic`.** Vector
  search returns nearest neighbours even when nothing relevant is indexed. When
  the closest result scores below 0.40 cosine similarity the response now says
  so and names the top value, so a weak match is not read as an answer.
- **`empty_metadata_scope` warning.** A metadata filter that matches nothing no
  longer reports an enum-filter explanation. The new warning names the metadata
  filter and states that `metadata_value` compares whole strings and
  `metadata_contains` is jsonb containment of the stored value, so a substring
  never matches.
- **`blast_radius` lists state their bound.** Caller and callee lists are capped
  at 25 and 40 entries. `inbound_count` and `callees_total` stay untruncated, and
  `callers_truncated` / `callees_truncated` mark the cut, so a short list cannot
  be mistaken for the full set. `covered_by_tests` is computed before truncation.

### Changed

- **Python call references carry their receiver.** `DatabaseSettings.from_env(...)`
  and `self.validate(...)` both name the receiver, and a name-only reference
  discarded that information. The parser now emits the qualified target, and
  `self` / `cls` resolve against the enclosing class (through nested functions and
  nested classes). `cls(...)` names the class itself. Other dotted calls keep the
  receiver and member, which resolution still falls back to by member.
- **Call-edge resolution prefers an exact name, then an unambiguous member
  name.** A bare callee name binds to a qualified definition only when exactly one
  qualified name ends in it; two candidates leave the edge unresolved instead of
  guessing a receiver type.
- **Language-scoped reserved-word filter for reference extraction.** Syntax that
  puts a parenthesis directly after a reserved word (`with (open(f) as h):`,
  `except (OSError, ValueError):`) was read as a call to that word. Reserved words
  per language are now excluded, and the list is per language on purpose, because
  `with(...)` is a call where `with` is not reserved.
- **A `def` or `class` signature is no longer a call.** A class body holds its
  methods' signature lines, so a class record previously reported itself as a
  caller of every method whose name matched any definition elsewhere in the
  repository.
- **Index rebuild required.** `PARSER_VERSION` moves from `stdlib-heuristic-v22`
  to `stdlib-heuristic-v25`. Reference extraction changed, so an existing snapshot
  keeps its old edges until `pci index` runs a full rebuild; the version change
  triggers that rebuild by itself. Measured on this repository, call-candidate
  edges went from 23,890 to 19,392 and resolved edges from 8,591 to 7,251. That
  reduction is removed noise (self-references, signature lines, reserved words),
  not lost coverage of real calls.

### Fixed

- **A definition no longer calls an unrelated symbol of the same name.** A body
  scan always sees the definition's own name on its signature line, and name-based
  resolution turned that into a call edge to any same-named definition in another
  file or language. 3,413 such self-name references existed in this repository,
  1,937 of which resolved to a record elsewhere. Recursive self-calls are dropped
  too: they carry no delete-safety information, and evidence already discarded
  them on read.

## Scope notes

- Receiver-qualified references are Python only. Go, Rust, and JavaScript or
  TypeScript would need per-language receiver-type inference, and a guess there is
  worse than an unresolved edge.
- The qualified-name fallback matches the `.` separator, which covers Python
  `Class.method` and Go `Receiver.Method`. Rust stores `Type::method` and already
  binds through exact references emitted by its own extractor.
- Member-name fallback has no language restriction, so a bare call to a common
  name (`start`, `add`, `resolve`) can still bind across languages. This is
  measured, not fixed: on this repository 10 of 7,251 resolved edges do so, and
  the same weakness exists within a single language, where a language filter would
  not help.
