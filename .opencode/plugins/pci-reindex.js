// PCI background reindex hook (debounced, opt-in).
//
// Default reindex is the git post-commit hook (indexes the clean committed
// tree). This per-edit reindex is opt-in for power users who want working-tree
// freshness between commits; enable with PCI_REINDEX_ON_EDIT=1. On each
// edit/write to a source file it (re)arms a debounce timer, then asks
// `pci-hook run --behavior reindex` to refresh in the background; pci-hook
// serialises runs with a lock. Tunable: PCI_REINDEX_DEBOUNCE_MS (5000).

import { spawn } from "node:child_process"
import { SOURCE_EXT, hookCmd } from "../lib/pci-evidence-logic.js"

const DEBOUNCE_MS = Number(process.env.PCI_REINDEX_DEBOUNCE_MS) || 5000
const WRITE_TOOLS = new Set(["edit", "write"])

export const PciReindex = async ({ directory }) => {
  // Opt-in: default reindex path is the git post-commit hook.
  if (!process.env.PCI_REINDEX_ON_EDIT) return {}
  const cmd = hookCmd(directory)
  let timer = null
  let running = false
  let pending = false

  const runIndex = () => {
    running = true
    pending = false
    const args = [...cmd.slice(1), "run", "--agent", "opencode", "--behavior", "reindex", "--repo", directory]
    const child = spawn(cmd[0], args, {
      cwd: directory,
      stdio: "ignore",
    })
    const done = () => {
      running = false
      if (pending) arm()
    }
    child.on("error", done)
    child.on("close", done)
  }

  const fire = () => {
    timer = null
    if (running) {
      pending = true
      return
    }
    runIndex()
  }

  const arm = () => {
    if (timer !== null) clearTimeout(timer)
    timer = setTimeout(fire, DEBOUNCE_MS)
    if (typeof timer.unref === "function") timer.unref()
  }

  return {
    "tool.execute.after": async (input) => {
      if (!WRITE_TOOLS.has(input.tool)) return
      const args = input.args || {}
      const filePath = typeof args.filePath === "string" ? args.filePath : ""
      if (!SOURCE_EXT.test(filePath)) return
      arm()
    },
  }
}
