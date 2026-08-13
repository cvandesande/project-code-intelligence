# Is `project-code-intelligence` paying off?

Running this tool isn't free. It costs:

- A Postgres + pgvector container running in the background.
- An embedding service (CPU/GPU/NPU) running locally or hitting a remote
  endpoint.
- Disk for indexed records, vectors, and SARIF findings.
- Re-indexing time after non-trivial changes.

Whether those costs pay back depends on how often you ask an AI agent
discovery-shaped questions ("where is X used?", "what calls Y?", "find
code that does Z") on this codebase. There's no single number that
answers it — but there's evidence you can collect.

## What high-quality evidence looks like

**Verifiable, concrete examples beat aggregate claims.** "I used
`search_code_intel_text` on `AppendCertsFromPEM` and got 4 call sites
in one response" is something you can check against the transcript.
"MCP saved me a lot of tokens" is not.

**Negative cases matter more than positive ones.** If an agent's
retrospective only lists MCP wins, that's confirmation bias talking.
A credible retrospective includes a few cases where `Read` or `grep`
would have been cheaper, named specifically.

**Quantitative token-savings estimates are unreliable.** The agent
doesn't have direct access to a token counter and the numbers are
guesses. Discount them.

**Server-side data is best of all.** If you can see *how many MCP calls
the server actually handled* in a session — independent of any agent's
self-report — that breaks the bias loop entirely. The current `pci mcp`
server doesn't expose that yet; see "Cross-session evidence" below.

## Asking for a session retrospective

A pre-written retrospective prompt lives at
[`docs/SESSION_RETROSPECTIVE_PROMPT.md`](SESSION_RETROSPECTIVE_PROMPT.md).
Paste it at the end of a non-trivial session when you want to evaluate
MCP usage. The prompt asks the agent to classify each MCP call as a
win, loss, or wash, with a concrete one-line description of each.

What to look for in the response:

- **A tri-state tally.** Not just wins — losses and washes too. A
  retrospective with zero losses across 20+ MCP calls is suspect.
- **Concrete examples.** "I queried `X` and got `Y`; the alternative
  was `Z`." If the response is abstract ("MCP helped a lot with
  cross-cutting work"), push back: ask for a specific case.
- **An actionable next-session suggestion.** "For follow-up sessions,
  prefer `Read` for [specific question shape]" is real feedback. "Keep
  using MCP for discovery" is filler.

If the agent produces a vague or aggregate-heavy retrospective despite
the prompt, that's a signal the prompt itself didn't get followed —
not that MCP isn't helping.

## Interpreting the tally

| Pattern | What it suggests |
|---|---|
| Mostly wins, few losses, all examples concrete | MCP is pulling its weight on this kind of work. Keep it running. |
| Mostly washes | The work isn't shaped right for MCP. Either you're asking small-known-path questions, or the codebase isn't large enough to benefit. Try turning the tool off for a week and see if you miss it. |
| A few wins but more losses | The agent is over-applying MCP. Check whether the system prompt's "fall back to Read/Grep when" section is being followed; a stricter prompt may help. |
| Refuses to name any losses | Discount the retrospective; it's optimism. Ask explicitly for the three weakest MCP calls. |

## Cross-session evidence

A single-session retrospective is a data point, not a verdict. For the
"is the install paying off?" question you want a few sessions' worth.

Lightweight options:

- **Run the retrospective prompt at the end of each substantive
  session for a week.** Save the responses. Skim them at the end.
  Patterns emerge — "MCP was used on every audit task" is a different
  signal from "I rarely reached for any MCP tool."
- **Compare with-MCP and without-MCP sessions on the same task.** Run
  a comparable task twice: once with the MCP server running, once with
  it stopped (`pci doctor --stop`). Compare the agent's path through
  the task and the final answer's quality. This is the cleanest test
  because the comparison is direct, but it costs running the task
  twice.

Heavier options (not yet built):

- **Server-side query log.** The `pci mcp` server could maintain a
  rolling log of recent tool calls with timing and response sizes that
  you can `cat` independently of any agent. That breaks self-report
  bias entirely. Not currently exposed; would need a server change.

## When to stop using the tool

The install isn't paying off if, over a few non-trivial sessions:

- The retrospective consistently shows mostly washes.
- The agent rarely volunteers an MCP call without nudging, and the
  end-of-session retrospective doesn't change your mind about its
  value.
- The cost of keeping the embedding service warm exceeds whatever
  discovery savings you observe.

In those cases, `make tool-uninstall` runs `pci doctor --clean` first, then
removes the installed `pci` binary. Use `pci doctor --clean` directly when
you only want to stop local services and remove generated runtime state while
keeping the CLI installed. Re-installing later is `make tool-install` plus, for
non-bundled Postgres, `pci doctor --init-postgres`.

The honest framing: this tool earns its keep on large, unfamiliar, or
code-graph-heavy work. On a small, well-known repo, the answer might
genuinely be *"the squeeze isn't worth it for me."* The retrospective
mechanism exists so you can make that call from evidence, not vibes.
