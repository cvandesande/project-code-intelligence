// PCI evidence hook.
//
// When an `edit` removes a definition, this injects that symbol's blast radius
// (callers, test coverage, wiring) into the tool result so the agent can judge
// "safe to cut?" while the change is fresh. When an edit ADDS one, pci-hook
// answers "does this already exist?" instead. The cheap JS gate below avoids
// spawning pci-hook on edits that touch no definition either way; all
// query/format logic lives in `pci-hook run` (Python).
//
// `write` is deliberately not handled. This is a tool.execute.AFTER hook, so by
// the time it runs the file on disk already holds the new text and there is no
// pre-write side left to diff against -- every definition in the file would look
// newly added. Supporting it needs a before-hook to capture the old content,
// which is a change to this adapter's shape, not a one-line addition. Claude is
// unaffected: its hook is registered on PreToolUse for Edit and Write both.

import { SOURCE_EXT, removedDefinitions, hookBin, runHook } from "../lib/pci-evidence-logic.js"

export const PciEvidence = async ({ directory }) => {
  const bin = hookBin(directory)
  return {
    "tool.execute.after": async (input, output) => {
      if (input.tool !== "edit") return
      const args = input.args || {}
      const filePath = typeof args.filePath === "string" ? args.filePath : ""
      if (!SOURCE_EXT.test(filePath)) return
      // Removed OR added: the gate exists to skip edits that change no definition, and
      // gating on removals alone made the add-side branch in pci-hook unreachable here --
      // neither the reminder nor the prior-art query ever fired on opencode. Added names are
      // the removal set with the sides swapped, the same way the Python runtime derives them.
      const removed = removedDefinitions(args.oldString, args.newString)
      const added = removedDefinitions(args.newString, args.oldString)
      if (removed.length === 0 && added.length === 0) return
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
