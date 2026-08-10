// PCI evidence hook (phase-3, delete-first).
//
// When an `edit` removes a definition, this injects that symbol's blast radius
// (callers, test coverage, wiring) into the tool result so the agent can judge
// "safe to cut?" while the change is fresh. The cheap JS gate below avoids
// spawning pci-hook on edits that remove nothing; all query/format logic lives
// in `pci-hook run` (Python).

import { SOURCE_EXT, removedDefinitions, hookBin, runHook } from "../lib/pci-evidence-logic.js"

export const PciEvidence = async ({ directory }) => {
  const bin = hookBin(directory)
  return {
    "tool.execute.after": async (input, output) => {
      if (input.tool !== "edit") return
      const args = input.args || {}
      const filePath = typeof args.filePath === "string" ? args.filePath : ""
      if (!SOURCE_EXT.test(filePath)) return
      if (removedDefinitions(args.oldString, args.newString).length === 0) return
      const event = JSON.stringify({
        filePath,
        oldString: args.oldString,
        newString: args.newString,
      })
      const text = (await runHook(bin, ["run", "--agent", "opencode", "--behavior", "evidence"], event)).trim()
      if (text) output.output += "\n\n" + text
    },
  }
}
