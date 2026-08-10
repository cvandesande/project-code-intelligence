// PCI evidence hook (phase-3, delete-first).
//
// When an `edit` removes a definition, this injects that symbol's blast radius
// (callers, test coverage, entry-point / module-level wiring) into the tool
// result, so the agent can judge "safe to cut?" while the change is fresh --
// the delivery step that pull-only tools miss.
//
// Boundary: all query logic lives in the Python `pci-evidence` CLI (backed by
// evidence.py); this plugin only resolves removed names and injects the text.
//
// Caveat surfaced by the CLI itself: the index is keyed on the git-index blob
// sha, so it usually predates the current edit -- output is labelled approximate.
//
// Setup: needs `pci-evidence` reachable. Order: $PCI_EVIDENCE_BIN, then the repo
// `.venv/bin/pci-evidence`, then PATH.

import { removedDefinitions, evidenceBin, SOURCE_EXT } from "../lib/pci-evidence-logic.js"

const MAX_SYMBOLS = 2 // budget: ~2 x 5 lines + header stays near the ~15-line cap
const NEIGHBORS = "0" // deletion cares about impact, not semantic twins

export const PciEvidence = async ({ $, directory }) => {
  const bin = evidenceBin(directory)
  return {
    "tool.execute.after": async (input, output) => {
      if (input.tool !== "edit") return
      const args = input.args || {}
      const filePath = typeof args.filePath === "string" ? args.filePath : ""
      if (!SOURCE_EXT.test(filePath)) return
      const removed = removedDefinitions(args.oldString, args.newString)
      if (removed.length === 0) return
      const reports = []
      for (const name of removed.slice(0, MAX_SYMBOLS)) {
        const result = await $`${bin} --symbol ${name} --neighbors ${NEIGHBORS}`.quiet().nothrow()
        const text = result.stdout.toString().trim()
        if (text) reports.push(text)
      }
      if (reports.length === 0) return
      const hidden = removed.length - MAX_SYMBOLS
      const overflow = hidden > 0 ? ` (+${hidden} more removed, not shown)` : ""
      output.output +=
        `\n\n[pci blast-radius${overflow} -- you removed the symbol(s) below. ` +
        `The index likely predates this edit, so treat as approximate; ` +
        `confirm no live caller before finalizing.]\n` +
        reports.join("\n---\n")
    },
  }
}
