# Rationale for the suggested system prompt

This document explains the design choices behind
[SYSTEM_PROMPT.md](SYSTEM_PROMPT.md). It exists so anyone tightening,
translating, or extending the prompt can do so without breaking the parts
that actually move agent behavior.

The goal of the prompt is **shifting which tool the agent reaches for
first**, not coverage. Every line in the prompt has to earn its keep
against that goal. The harder question — and the one this document answers
— is *which* lines earn their keep and which are sentimental clutter.

## The underlying problem

Coding agents have access to many tools. Their default behavior on
discovery questions ("where is `Foo` used?", "what calls this?", "find the
TLS config code") is to grep a working tree and then read several files.
That works, but on a large or unfamiliar codebase it loads a lot of files
that turn out not to be relevant. The `project-code-intelligence` MCP
server exists to cut that speculative loading down by letting the agent
query an index before reading.

The catch: the agent has to *choose* to query the index instead of grep.
That choice has two properties that shape everything else in this prompt:

1. **It is path-dependent.** The first tool reached for in a task locks in
   the next several. An agent that opens with grep stays in grep; an
   agent that opens with an MCP call stays in MCP. The first decision
   compounds.
2. **It has high run-to-run variance on borderline tasks.** The same
   prompt and the same task produce different tool choices on different
   runs. The agent generates coherent post-hoc justifications for
   whichever path it took, which makes single-run results look
   explanatory when they're often just sampling noise.

The prompt is designed around those two properties. It targets the first
call specifically, and it doesn't promise determinism.

## What was learned by trying

Several rounds of evidence shaped this prompt. The findings are listed in
order of how load-bearing they turned out to be.

1. **Channel matters more than content.** The same cheat sheet placed in
   `CLAUDE.md` (user-level context) does not reliably shift tool choice;
   placed in the system prompt, it does. This is the single most
   important finding for anyone deploying a similar prompt. If you ship
   this content as a user-level addition, expect it to underperform.

2. **The first call is the lever.** Once an agent has opened with grep,
   subsequent steps stay in grep — the file list it just produced
   becomes the next planning input, and switching costs more than
   staying. Framing the prompt around "name the first call" produces
   more behavioral pull than framing it around "use MCP when…" lists
   distributed across the workflow.

3. **Soft framing fails.** Saying "consider X" or "X is available" does
   not reliably override the agent's default tool selection. The agent
   will read the line, acknowledge the tool exists, and reach for grep
   anyway. The prompt declares MCP as the default for discovery; it does
   not invite preference.

4. **Concrete `if-task-then-tool` mappings work, but only as first-call
   defaults.** When the prompt contains specific task phrasings ("how
   does X work", "outline of one file") next to specific first calls,
   the agent pattern-matches incoming requests against those rows. The
   same mappings buried in the middle of the workflow are weaker — the
   agent has already made the first call by then.

5. **Honesty about *when not* to use MCP is load-bearing.** Without an
   explicit exceptions section, the agent over-applies MCP (e.g.,
   querying for a known small file's contents when `Read` is one cheap
   call). Crucially, the exceptions section now includes
   *removal-shaped enumeration tasks* — tracing every reference to a
   symbol you're about to delete is genuinely grep work, not MCP work.
   Omitting this exception would push the agent toward MCP on tasks
   where grep is correct, which weakens the credibility of the rest of
   the doctrine.

6. **Articulation requirement makes silent defaulting visible.** The
   "name your tool choice before broad work" sentence does not coerce
   tool selection — the agent can still pick grep — but it forces the
   decision onto the transcript. Without articulation, the agent
   dispatches silently and the user discovers the choice only by
   reading the tool log.

7. **Self-evaluation only works when it's asymmetric and anchored.**
   Asking the agent to "report whether MCP saved tokens" breaks down:
   agents can't measure tokens directly, confirmation bias pushes
   self-assessments toward "yes the tool helped", and the report itself
   costs tokens. Two narrower forms work:
   - **Negative-only mid-task reports.** Surface only the cases where
     MCP turned out worse than Read/Grep. Confirmation bias works
     *against* surfacing these, which makes them credible when they
     appear.
   - **End-of-task summary anchored by the in-line negatives.** The
     summary can't quietly omit problems because they're already on
     the record.

   Quantitative estimates ("saved ~N tokens") were considered and
   dropped. The agent can't compute them honestly.

8. **Post-hoc justifications are coherent but not causal.** When the
   agent explains why it chose a particular tool, the explanation will
   sound principled regardless of the underlying mechanism. Don't read
   single-run rationalizations as evidence that the prompt is or isn't
   working — measure tool choice across multiple runs on
   discovery-shaped tasks.

## Section-by-section: what's there and why

### Path-dependence framing (opening)

```
Tool selection in this project is path-dependent. The first tool you reach
for locks in the next several. Open with grep and you stay in grep mode;
open with an MCP call and you stay in MCP mode for the surrounding work.
The decision that matters most is the first one.
```

This is the highest-leverage paragraph in the prompt. It does two things:

- **Names the mechanism.** The agent reads a description of its own
  behavior. Whether this introspection is causal or just informational,
  the framing makes the first-call decision feel weighty.
- **Anchors everything that follows.** The "first call" framing is the
  spine the rest of the prompt hangs on — the trigger table is
  *first-call* rules, the exceptions are *first-call* exceptions.

An earlier draft opened with "MCP is the default for discovery." That's
true but generic. The path-dependence framing is specific to the actual
failure mode and reads as analysis rather than instruction.

### Articulation requirement

```
**Before any broad search or sub-agent dispatch, name the tool you intend
to use and why.** The choice should be visible whether it's MCP or grep.
```

This paragraph does not coerce MCP. It makes the choice visible. The
agent can still pick grep; it just has to say so. Two structural
consequences:

- **Silent defaulting becomes hard.** The agent can't reach for grep
  without acknowledging the alternative.
- **Path-dependence becomes auditable.** When the first-call goes wrong,
  the user can see *why* — the rationalization is on the transcript.

What this paragraph deliberately avoids:
- Mandating MCP. An agent that always uses MCP would be as broken as one
  that never does; the goal is legible choice, not coerced choice.
- Per-call articulation. Targeted at broad searches and sub-agent
  dispatch only. Read-on-known-path doesn't need narration.

### The first call section

```
Before any other search, run code_intel_status. It is cheap, it confirms
the index is current, and it sets the working pattern for the session.
Skip only if you've already called it this session.
```

`code_intel_status` is the default opener for three reasons:

- **It's the cheapest commitment.** A small status query that confirms
  the index is current.
- **It sets the pattern.** Once the agent has made one MCP call, the
  next step naturally stays in MCP — path dependence working in our
  favor.
- **It surfaces the queryable surface.** Subsequent filter calls don't
  fail with `empty_<dim>_scope` warnings because the agent has just
  seen the valid values.

The task-shape table below it covers cases where `code_intel_status`
isn't the obvious first call. Selection criteria for each row:

| Row | Why it's there |
|---|---|
| "How does X work" → `search_code_intel_semantic` | The case grep can't answer at all. Concept queries are the strongest case for MCP. |
| Outline of one file → `search_code_intel_text` w/ `mode=enumerate` | Replaces `grep -n "^def "` with bounded line ranges and snippets in one call. |
| Find references to a named symbol → `search_code_intel_text` | The default discovery question. Ranked snippets beat raw grep hits. |
| Who calls a function → `related_code_intel` | The single best demonstration of "MCP wins" — call-graph traversal is genuinely expensive with grep+Read. |
| Have a `record_id` already → `get_code_intel_record` | The natural follow-up after `search_code_intel_text`. Pre-bounded function body. |

The table is short on purpose. Long enough to pattern-match against,
short enough that the agent generalizes rather than treating it as
exhaustive.

### After the first call

```
If the first call was MCP, prefer MCP for the surrounding work...
If the first call was grep, finish in grep. Don't toggle modes mid-task.
```

This section explicitly endorses path dependence rather than fighting it.
The path-dependence framing at the top says it happens; this section
says *let it happen, consistently*.

The "pair semantic with text" line is concrete guidance for a specific
pattern that came up repeatedly: semantic for concept, text for exact
symbol. Each tool covers a gap the other has.

### Exceptions section

The four exceptions are calibrated to prevent specific failure modes:

- **Exact string literals or error messages.** grep is faster and the
  index adds nothing.
- **Files known by path.** Read is one cheap op.
- **Verifying any MCP result before acting on it.** On-disk is source of
  truth; the index is a snapshot.
- **Cross-file enumeration of touch points for removal.** This one is
  load-bearing. Without it, the agent over-applies MCP on
  delete-this-symbol tasks where the ranked, capped, parent-deduped
  output is exactly the wrong shape. The agent needs `grep -rn` for
  flat exhaustive enumeration.

The fourth exception was added after observation: the agent correctly
identified that a removal-shaped task was grep territory but couldn't
justify it from the prompt because the prompt didn't acknowledge the
case. Adding the explicit carve-out aligns the prompt with the right
behavior the agent was already producing.

### Self-evaluation requirement

```
After an MCP call where Read/Grep would have been cheaper (it over-returned,
or you still had to Read multiple files anyway), say so in one sentence at
the time. At the end of a non-trivial task, summarize MCP usage in 3–5
lines: calls made and any misjudgments noted.
```

Two halves designed to anchor each other:

- Per-turn negative reports are honest because confirmation bias works
  *against* the agent surfacing them.
- The end-of-task summary is honest because it has to reference whatever
  negatives already surfaced in the transcript — it can't quietly omit
  them.

If you compress the prompt aggressively, this is the second-most
compressible section (after the Notes trailer). Removing it doesn't
break tool selection; what it removes is the observability loop. Without
self-evaluation, you can't tell from outside the session whether the
prompt is working over time.

### Notes (trailer)

```
Trust tools/list over this prompt if schemas drift. Empty-result responses
include empty_<dim>_scope / mode_inferred_enumerate warnings... After an
MCP failure, fall back and surface it once; don't retry in a loop.
```

Three distinct purposes packed into one short paragraph:

- **`tools/list` deflection.** The schemas are versioned by the server,
  not this prompt. The prompt won't go stale silently as the server
  evolves.
- **Warning-kind hint.** When MCP returns empty results, the agent's
  default reaction is to assume MCP doesn't work and fall back. The
  hint keeps the agent on the MCP path one round more often.
- **Retry guidance.** Defensive against transient errors that agents
  tend to loop on.

The most compressible part of the prompt. Can be cut under severe budget
pressure, but the empty-result hint in particular pays for itself within
one or two real sessions.

## What was deliberately cut

The earlier version of this prompt and the project's earlier
infrastructure included several things that didn't survive contact with
evidence.

- **The generated code map (`pci-context`).** An auto-generated markdown
  map injected into context. Tested via `CLAUDE.md` injection; produced
  modest wall-time improvement but zero shift in tool selection. The
  cheat-sheet-as-doctrine in the system prompt produces the actual
  behavioral shift the map was supposed to produce, at a fraction of
  the token cost. The map was overkill.
- **`list_code_intel_parser_failures`.** Moved to a `pci-index` flag.
  Diagnostic surface for the indexer, not for code queries; doesn't
  belong in the model-facing tool palette.
- **`get_code_intel_records` (plural).** Folded into
  `get_code_intel_record` — the singular tool accepts arrays.
- **`get_static_code_flow`.** Redundant with `get_static_finding`
  passing `include_code_flows=true`.
- **Trigger rows for SARIF tools.** SARIF tools survive as a queryable
  surface but are inert until SARIF data is ingested. They're not
  worth a dedicated trigger row in the first-call doctrine; the
  workflow for security review is sufficiently different that it
  warrants its own prompt overlay rather than a row here.
- **Quantitative self-evaluation.** "Saved ~N tokens" framings.
  Agents can't compute these honestly and inviting confabulation
  degrades the rest of the report.

## When to deviate from this prompt

This prompt is calibrated for a *general-purpose coding agent on an
average-sized indexed repo*. Adjust if your situation differs:

- **Very small repo / monolingual / familiar codebase.** The
  path-dependence framing still applies but the index pays off less.
  Either drop the prompt or rewrite the first-call section to make
  `Read` the default opener.
- **Heavy SARIF / security-review workflow.** Add SARIF trigger rows
  back to the table or write a separate overlay. The generic prompt
  doesn't surface them strongly enough.
- **Read-only inspection sessions.** Drop the stale-index caveat from
  the exceptions — the index can't go stale relative to changes that
  aren't happening.
- **Different MCP client (Codex, Zed, Cline, etc.).** The prompt is
  vendor-neutral. The *channel* matters — make sure your client
  injects it at the system-prompt layer, not user-level. See
  [MCP_SETUP.md](MCP_SETUP.md) for per-client config.

## How to evaluate prompt changes

If you tighten or rewrite the prompt, measure across multiple runs.
Single-run results are unreliable; the same prompt and task can produce
different tool choices on different runs due to sampling variance.

Four probes, each run at least 3–5 times:

1. **First-call regression.** Pose a "where is `X` used?" question.
   Across multiple runs, does the agent reach for MCP more often than
   grep? A consistent grep-first pattern means the path-dependence
   framing or first-call section weakened.
2. **Discovery-vs-enumeration regression.** Pose a "remove this symbol"
   task. Across multiple runs, does the agent correctly identify it as
   grep territory? If the agent applies MCP to enumeration tasks, the
   exceptions section weakened.
3. **Articulation regression.** Give the agent a task that requires a
   broad search. Does it name the tool choice *before* acting? If it
   dispatches silently and only justifies under questioning, the
   articulation requirement weakened.
4. **Stale-index regression.** Make a small edit, then ask the agent to
   find something you just changed. Does the agent recognize the
   post-edit case and grep instead, or call MCP and report on stale
   data? If the latter, the exceptions section weakened.

The first probe is the most important and the easiest to drift on.
Run it whenever you tighten the prompt.

## Reproducing this prompt for a different MCP server

The same structure works for any read-only-discovery MCP. The pattern:

1. **Path-dependence framing.** One paragraph naming the mechanism
   (first call locks in subsequent ones). This is the spine.
2. **Articulation requirement.** One bold sentence asking the agent to
   name its tool choice before broad work. Cheap; makes silent
   defaulting visible.
3. **First-call directive.** One sentence on the cheap status /
   orientation call that opens every session.
4. **First-call task table.** 4–7 rows mapping concrete task shapes to
   concrete first calls. Cover the common queries; let `tools/list`
   handle the long tail.
5. **After-first-call rule.** Explicit endorsement of path dependence
   ("stay in the mode you opened in").
6. **Exceptions list.** 3–5 bullets covering known-path reads,
   verification, stale-index risk, and any enumeration patterns that
   are genuinely grep territory.
7. **Self-evaluation requirement.** Asymmetric negative-only in-line
   reports, plus an end-of-task summary anchored by those negatives.
8. **Notes trailer.** Defer to `tools/list` for schemas; flag any
   server-specific structured warnings; failure-handling guidance.

Three things to verify before shipping:
- The prompt is delivered via **system prompt**, not user-level context.
- The first-call directive is in the **opening third** of the document.
- The **discovery vs enumeration carve-out** is in the exceptions list.

Anything beyond these belongs in a longer-form contributor guide, not
in a system prompt that pays a token cost on every turn.
