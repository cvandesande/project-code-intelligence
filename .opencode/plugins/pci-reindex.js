// PCI background reindex hook (debounced, non-blocking).
//
// On each edit/write to a source file it (re)arms a debounce timer; once the
// writes go quiet it asks `pci-hook run --behavior reindex` to refresh the
// index in the background. `pci-hook` serialises runs with a lock, so a write
// that lands mid-run triggers exactly one more pass. Tunable:
// PCI_REINDEX_DEBOUNCE_MS (5000). Timers fire only while the session lives.

import { spawn } from "node:child_process"
import { SOURCE_EXT, hookBin } from "../lib/pci-evidence-logic.js"

const DEBOUNCE_MS = Number(process.env.PCI_REINDEX_DEBOUNCE_MS) || 5000
const WRITE_TOOLS = new Set(["edit", "write"])

export const PciReindex = async ({ directory }) => {
  const bin = hookBin(directory)
  let timer = null
  let running = false
  let pending = false

  const runIndex = () => {
    running = true
    pending = false
    const child = spawn(bin, ["run", "--agent", "opencode", "--behavior", "reindex", "--repo", directory], {
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
