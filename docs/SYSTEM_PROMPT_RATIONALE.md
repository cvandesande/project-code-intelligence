# Rationale for the suggested system prompt

This document explains the design choices behind
[SYSTEM_PROMPT.md](SYSTEM_PROMPT.md). It exists so anyone tightening,
translating, or extending the prompt can do so without breaking the parts
that actually move agent behavior.

The goal of the prompt is **token and time savings**, not coverage. Every
line in the prompt has to earn its keep against that goal. The harder
question — and the one this document answers — is *which* lines earn their
keep and which are sentimental clutter.

## The underlying problem

Coding agents have access to many tools. Their default behavior on
discovery questions ("where is `Foo` used?", "what calls this?", "find the
TLS config code") is to grep a working tree and then read several files.
That works, but on a large or unfamiliar codebase it loads a lot of files
that turn out not to be relevant. The `project-code-intelligence` MCP
server exists to cut that speculative loading down by letting the agent
query an index before reading.

The catch: the agent has to *choose* to query the index instead of grep.
Soft prompts ("you can use the MCP server, it might help") aren't enough
to redirect that choice. Concrete trigger conditions are — but they're
also not sufficient on their own. The current prompt evolved through two
rounds of evidence.

## What was learned by trying

Four observations shaped this prompt:

1. **Soft framing fails.** Saying "consider X" or "X is available" in a
   system prompt does not reliably override the agent's default tool
   selection. The agent will read the line, acknowledge the tool exists,
   and then reach for `Bash` + `grep` anyway because that's the
   lowest-resistance path.

2. **Concrete `if-task-then-tool` mappings work — for visible per-turn
   decisions.** When the prompt contains specific task phrasings ("Where
   is `Foo` defined/used?") next to specific tool calls
   (`search_code_intel_text query="Foo"`), the agent pattern-matches
   incoming requests against those rows. That's the behavioral pull
   abstract preferences don't have.

3. **Honesty about *when not* to use MCP is load-bearing.** Without an
   explicit fallback section, the agent over-applies the tool (e.g.,
   querying for a known small file's contents when `Read` would be one
   call against a known path). The fallback rules also pre-empt common
   mistakes that would otherwise need debugging round-trips (querying
   the stale index after edits, looping on MCP errors).

4. **Trigger maps don't reliably guide sub-agent dispatch or broad
   workflow decisions.** In a real audit session, the agent correctly
   applied the skip rules for Criticals (known paths → `Read`) but did
   not articulate an MCP plan for cross-cutting High findings until the
   user asked why. The agent's post-hoc justification was articulate,
   but whether MCP would have been used without the nudge was
   unobservable. The fix isn't more trigger rows — it's forcing the
   agent to *name its tool choice* before broad searches or sub-agent
   dispatch. Articulation makes the decision visible to both the agent
   and the user; it also tightens the agent's own reasoning loop because
   it has to commit before acting.

5. **Self-evaluation only works when it's asymmetric and anchored.**
   Asking the agent to "report whether MCP saved tokens" sounds
   reasonable but breaks down in practice: agents can't measure token
   counts directly, the report itself costs tokens, and confirmation
   bias pushes self-assessments toward "yes the tool helped". Two
   narrower forms work better:
   - **Negative-only mid-task reports**: surface only the cases where
     MCP turned out worse than Read/Grep (over-returned results, still
     required multiple file reads). The signal is asymmetric and
     confirmation bias works *against* surfacing it, which makes the
     reports credible when they appear.
   - **End-of-task summary anchored by the in-line negatives**: a short
     wrap-up listing MCP calls and any misjudgments already named in
     the transcript. The summary can't quietly omit problems because
     they're already on the record. The negatives provide a credibility
     anchor the positives lack.

   Quantitative estimates ("saved ~N tokens" or "avoided ~N file reads")
   were considered and dropped. The agent can't compute them honestly,
   and inviting confabulation degrades the rest of the report. The
   wrap-up sticks to counts and named misjudgments — both observable.

## Section-by-section: what's there and why

### Framing paragraph

```
Indexes the source tree (files, records, edges, embeddings, SARIF) into
Postgres. For discovery questions — "where is X used?", "what calls Y?",
"what's in this codebase?", "find code that does Z" — the index is the
default; Read/Grep are the fallback. For known-path reads of small files,
the reverse is true.
```

- Names what the server *is* (so the agent doesn't have to fetch
  `tools/list` just to learn the surface).
- **Names MCP as the default for discovery** — a deliberate rhetorical
  inversion. An earlier draft framed it as "wins on discovery; for known
  paths Read/Grep stay cheaper", which read as a balanced preference. In
  practice the balanced framing left enough room for the agent to
  default to native tools on cross-cutting work. The current framing
  names two regimes and assigns a default in each, so the prompt's bias
  is unambiguous from the first paragraph.
- **Names the discovery shape concretely** with four illustrative
  question types. The list is short on purpose — long enough to
  pattern-match against, short enough that the agent generalizes rather
  than treating it as exhaustive.

What this paragraph deliberately avoids:
- The word *prefer*. "Prefer" reads as a directive *and* still allows
  defaulting away when convenient. Naming a default is stronger.
- Token-saving puffery. "Save tokens" is what every tool description
  claims; the agent has heard it before and discounts it. Naming the
  *mechanism* (replace speculative file reads with a query) is more
  persuasive than naming the *outcome* (save tokens).

### Articulation requirement

```
**Before any broad file search or sub-agent dispatch, name the MCP query
you'd run first (or briefly say why Read/Grep is the right call instead).**
The choice should be visible whether you end up calling MCP or not.
```

This paragraph is the response to observation #4 above. It does two
things:

- **Forces tool-choice visibility per turn.** The agent can't quietly
  default to native tools and rely on the user to notice. Naming the
  intended MCP call (or justifying its absence) creates a checkpoint
  before broad action.
- **Covers both directions.** "The choice should be visible whether you
  end up calling MCP or not" prevents the requirement from becoming a
  forcing function on tool selection — it's a forcing function on
  reasoning. The agent can still pick `Read`; it just has to say so.

What this paragraph deliberately avoids:
- Mandating MCP. The articulation requirement is structural, not
  directive. An agent that always uses MCP would be as broken as one
  that never does; the goal is legible choice, not coerced choice.
- Per-call articulation. Targeted at "broad searches or sub-agent
  dispatch" specifically. Read-on-known-path doesn't need narration.

This is the most compressible section if budget is severe, but it's also
the section with the most observed behavioral pull. The bold formatting
is deliberate — drawing the eye to the one per-turn habit the prompt
asks for.

### Self-evaluation requirement

```
**After an MCP call where Read/Grep would have been cheaper (it
over-returned, or you still had to Read multiple files anyway), say so
in one sentence at the time.** At the end of a non-trivial task,
summarize MCP usage in 3–5 lines: calls made and any misjudgments noted.
```

This pair sits at the bottom of the prompt's reasoning arc:

- **Articulation** (before broad work) — name the intended tool choice.
- **Defaults + fallback** (during) — use the index for discovery,
  Read/Grep for known paths.
- **Self-evaluation** (after) — negative-only in-line reports, then a
  short summary at task end.

The two halves are designed to anchor each other (see observation #5).
Per-turn negative reports are honest because confirmation bias works
*against* the agent surfacing them. The end-of-task summary is honest
because it has to reference whatever negatives already surfaced in the
transcript — it can't quietly omit them.

What this section deliberately avoids:
- **Quantitative savings claims.** No "saved ~N tokens" or "avoided
  ~M file reads." The agent can't compute these honestly. The wrap-up
  sticks to counts and named misjudgments, both of which the user can
  verify against the transcript.
- **Positive in-line reports.** Asking the agent to also surface MCP
  *wins* per turn would re-introduce confirmation bias and double the
  per-turn cost. The end-of-task summary is the right place for net
  positives (implied by call count without explicit success claims).
- **Per-turn summaries on tiny tasks.** "Non-trivial task" is the same
  fuzzy qualifier as `code_intel_status`-first; in practice agents
  calibrate it well enough.

If you compress the prompt aggressively, this is the second section to
cut (after the trailer). Removing it doesn't break behavior; the
articulation requirement and trigger map still pull. But the
observability loop the section creates — *can the user tell whether
MCP was applied well across a session?* — is what makes the prompt
auditable over time.

### Trigger map (`## Default to MCP for:`)

The table is the single most important piece of the prompt for
*matching* incoming tasks to the right tool. Each row is a concrete
`if-the-task-looks-like-X-then-use-Y` rule.

The section heading is `## Default to MCP for:` — not `## Use MCP when:`.
This is a small rhetorical change but a real one: "default" implies a
baseline you'd have to argue away from. "Use when" implies a permission
you opt into.

Selection criteria for each row:

| Row | Why it's there |
|---|---|
| `"Where is Foo defined/used?"` → `search_code_intel_text` | The most common discovery question. Replaces grep + N file reads with one structured response. |
| `"What's queryable here?"` → `code_intel_status` | The authoritative source for valid filter values (languages, file roles, record types, content classes). Cheap and contextual. |
| `"List files matching a filter"` → `list_code_intel_files` | Replaces `find` + manual extension/path filtering. Boolean-filter aware. |
| `"What calls/uses Bar.method?"` → `related_code_intel` | The single best demonstration of "MCP wins" — call-graph traversal is genuinely expensive with grep+Read. |
| `"Find code by concept, no identifier"` → `search_code_intel_semantic` | The case grep can't answer at all. Surfacing this case is more about reminding the agent the option exists than about token savings. |
| `"Full text of record <id>"` → `get_code_intel_record` | The natural follow-up after `search_code_intel_text` returns matches; the row makes the two-step flow obvious. |
| `"Triage SARIF findings"` → `search_static_findings` etc. | The static-analysis surface is a genuinely separate use case; without this row the agent forgets it exists. |

What's not on the table and why:
- `get_code_intel_records` (batched fetch). Implied by `get_code_intel_record`; agents pluralize correctly without prompting.
- `list_code_intel_parser_failures`. Niche; visible from `tools/list` when relevant.
- Static-flow tools individually. Folded into one row to keep density up.

The trigger map is also where most of the prompt's *length* lives. It's
tempting to compress further; resist that urge. Removing rows reduces
behavioral pull more than it saves context — each row maps to a real
recurring task. Below ~5 rows the prompt loses the pattern-match
function and reverts to abstract preference language.

### `code_intel_status` first call

```
Start non-trivial sessions with `code_intel_status`: cheap, confirms the
index matches HEAD, lists queryable filter values.
```

Single line, high leverage:
- Reduces wasted MCP calls against stale indexes by making the freshness
  check default.
- Surfaces the queryable surface up front, which means subsequent
  filter-by-language/role calls don't fail with `empty_<dim>_scope`
  warnings and re-roundtrips.
- The "non-trivial sessions" qualifier is deliberate. Forcing this on
  every session would bloat token usage on simple tasks where one
  `Read` would settle the matter.

### Fallback section (`## Fall back to Read/Grep when:`)

The honesty section, demoted from the earlier "Skip MCP for…" framing.

Two structural changes from earlier drafts:
1. **Heading flipped.** `Skip MCP for` framed native tools as the
   exception you fall *into*. `Fall back to Read/Grep when` reinforces
   that MCP is the default and Read/Grep are the alternative.
2. **Bullet count cut from 3 to 2, plus a trailing one-line note.** The
   old three-bullet list had visual weight comparable to the trigger
   map, which let the agent's eye treat them as equally weighted
   options. The demoted form makes the trigger map clearly dominant.

The two bullets that remain:

- **The path is known and the file is small (Read is one cheap op).**
  Without this, the agent routinely calls `search_code_intel_text` when
  it already has the file path, and MCP's structured response is larger
  than the file itself.
- **The index is stale.** Post-edit verification, uncommitted changes,
  `snapshot_dirty` warnings. The index is a snapshot from the last
  `pci-index` run. The agent will confidently look at stale data after
  edits unless this is stated explicitly. This bullet has the highest
  "prevent debugging round-trip" value per word.

The trailing note ("After an MCP failure, fall back and surface it
once; don't retry in a loop.") is defensive against transient errors
that agents tend to retry on. Folding it out of the bullet list shrinks
visual weight further without losing the rule.

What's not in this section:
- A length comparison ("MCP responses are several KB"). Implied by the
  framing paragraph's "known-path reads of small files" clause.
- A defense of the trade-offs. Agents respond to rules, not to
  defensiveness.

### Trailer (`tools/list` deflection + warning kinds)

```
Trust `tools/list` over this prompt if schemas drift. Empty-result
responses include `empty_<dim>_scope` / `mode_inferred_enumerate`
warnings that name the bad filter and point at `code_intel_status` for
valid values.
```

Two distinct purposes packed into one paragraph:

- `tools/list` deflection. The schemas are versioned by the server, not
  this prompt. The deflection clause means the prompt won't go stale
  silently as the server evolves.
- Warning-kind hint. When MCP returns empty results, the agent's default
  reaction is to assume MCP doesn't work and fall back. The hint tells
  it the warnings are structured and self-pointing, which keeps the
  agent on the MCP path one round more often (and surfaces filter
  mistakes faster).

This trailer is the most compressible part of the prompt. It can be cut
if the prompt budget is severe, but the index-warnings hint in
particular pays for itself within one or two real sessions because
empty-result responses *are* common.

## What was deliberately cut

The long version of this prompt had several more sections. They were
removed because the tight version maintains behavioral pull without
them:

- **Practical workflow paragraph.** Covered by the framing, the
  articulation requirement, and the `code_intel_status`-first line.
- **Watch-outs section.** Boolean-filter quirks, `mode=enumerate` rules,
  pci-mcp scope notes. These belong in `tools/list` / the warning trailer
  rather than as standing prompt content.
- **Cost-reality paragraph.** Substance preserved in the framing
  paragraph and the fallback section.
- **Examples and case studies.** Worth keeping in a longer-form
  contributor guide if one ever exists, but they don't redirect agent
  behavior beyond what the trigger map already does.

## When to deviate from this prompt

This prompt is calibrated for a *general-purpose coding agent on an
average-sized indexed repo*. Adjust if your situation differs:

- **Very small repo / monolingual / familiar codebase.** The framing
  paragraph's "discovery over a large/unfamiliar codebase" win becomes
  the minority case. Either drop the prompt or rewrite the framing to
  make Read/Grep the default in both regimes. MCP overhead may not pay
  off here.
- **Heavy static-analysis workflow (security review, SARIF triage).**
  Promote the static-finding row higher and add `get_static_code_flow`
  as its own line.
- **Read-only inspection sessions (no edits planned).** You can drop
  the stale-index caveats; the index can't go stale relative to changes
  that aren't happening. This shortens the prompt by ~3 lines.
- **Different MCP client (Codex, Zed, Cline, etc.).** The prompt is
  vendor-neutral and should work as-is. The system-prompt slot is
  client-specific; see [MCP_SETUP.md](MCP_SETUP.md) for the per-client
  config snippets `pci-index --mcp-config <client>` emits.

## How to evaluate prompt changes

If you tighten or rewrite the prompt, measure behavior, not feel. Four
crude but effective probes:

1. **Discovery-question regression.** Pose a "where is `X` used?"
   question. Does the agent reach for MCP, or default to `grep`?
2. **Stale-index regression.** Make a small edit, then ask the agent to
   find something you just changed. Does the agent (a) call MCP, get
   stale data, and report confidently, or (b) recognize the post-edit
   case and grep instead?
3. **Articulation regression.** Give the agent a task that requires a
   broad search or sub-agent dispatch. Does it name the MCP query (or
   justify Read/Grep) *before* acting, or does it dispatch silently and
   only justify under questioning?
4. **Self-evaluation regression.** Run a multi-step task where MCP
   should be over-applied at least once (e.g., a known small file in
   the middle of cross-cutting work). Does the agent surface that
   misjudgment in one sentence at the time, and reference it in the
   end-of-task summary? If both stay silent, the self-evaluation
   habit has weakened.

If (1) goes the wrong way, the trigger map or framing baseline
weakened. If (2) goes the wrong way, the fallback section weakened. If
(3) goes the wrong way, the articulation requirement weakened or was
removed. If (4) goes the wrong way, the self-evaluation section is
either gone or got rewritten in a way that makes the in-line negatives
optional. All four regressions are common when prompts get aggressively
compressed.

## Reproducing this prompt for a different MCP server

The same structure works for any read-only-discovery MCP. The pattern:

1. **Framing paragraph**: what the server indexes, two-regime default
   declaration (MCP is the default for these tasks; native tools are
   the default for those). One sentence per regime.
2. **Articulation requirement**: one bold sentence asking the agent to
   name its tool choice before broad work. Cheap; makes silent
   defaulting visible.
3. **Self-evaluation requirement**: one paragraph asking for negative-only
   in-line reports and an end-of-task summary anchored by those negatives.
   Avoid quantitative estimates the agent can't compute.
4. **Trigger map**: 5–8 concrete task→tool rows. Cover the common
   queries; let `tools/list` handle the long tail.
5. **First-call habit**: one sentence on the cheap status/orientation
   call.
6. **Fallback list**: 2–3 bullets covering known-path reads, stale-data
   risk, and (as a trailing note) failure handling. Keep this section
   visually shorter than the trigger map.
7. **Trailer**: defer to `tools/list` for schemas; flag the
   server-specific structured warnings if any.

Anything beyond that probably belongs in a longer-form contributor
guide, not in a system prompt that pays a token cost on every turn.
