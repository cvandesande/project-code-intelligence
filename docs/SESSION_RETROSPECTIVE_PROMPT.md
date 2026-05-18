Produce an MCP-usage retrospective for this session.

For each non-trivial MCP call, list:

1. **The question or task** that prompted the call (one line).
2. **The MCP tool** called and a one-line summary of what it returned.
3. **The alternative** you would have used (`Read` path, `grep` pattern,
   sub-agent dispatch shape).
4. **Verdict — win / loss / wash:**
   - **Win**: MCP gave a directly useful answer the alternative would
     have required several follow-up reads to reach.
   - **Loss**: the response was over-broad or you still had to Read
     multiple files after; the alternative would have been cheaper.
   - **Wash**: roughly equivalent cost; either tool would have worked.

Close with:

- **Tally**: count of wins, losses, washes.
- **One sentence** on whether MCP earned its keep this session,
  anchored by the tally.
- **One concrete case** where you'd suggest the next session use the
  alternative tool instead — a specific question shape, not a generic
  principle.

**Skip quantitative token-savings estimates**, even for the "earned its
keep" sentence. You can't compute them honestly; the win/loss/wash
tri-state and concrete examples are the credible signal.

If the session had fewer than three non-trivial MCP calls, say so and
skip the tally; one short paragraph is enough.
