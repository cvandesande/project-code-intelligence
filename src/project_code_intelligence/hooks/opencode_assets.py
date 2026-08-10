"""opencode plugin assets, written verbatim by ``pci-hook install --agent opencode``.

These are the single source for the opencode adapter. The plugins are thin: a
cheap JS gate (source-file extension, removed-definition check) decides whether
to invoke ``pci-hook run``, and all evidence/reindex logic lives in Python.
Keep the gate patterns in ``LIB_JS`` in sync with ``hooks/detect.py``.
"""

from __future__ import annotations

LIB_JS = r"""// Shared helpers for the pci opencode plugins. Kept out of `plugins/` so the
// loader (which scans only `plugins/`) does not treat it as a plugin entry.
// The gate below mirrors `hooks/detect.py`; keep the two in sync.

import { existsSync } from "node:fs"
import { spawn } from "node:child_process"

// Files whose edits are worth resolving against the code index.
export const SOURCE_EXT = /\.(py|go|sh|bash|c|h|rs|js|ts|java)$/

// Definition forms across the indexed languages; each capture group is the name.
const DEF_PATTERNS = [
  /(?:^|\n)[ \t]*(?:async[ \t]+)?(?:def|class|func|type)[ \t]+([A-Za-z_]\w*)/g,
  /(?:^|\n)[ \t]*func[ \t]*\([^)]*\)[ \t]*([A-Za-z_]\w*)/g,
  /(?:^|\n)[ \t]*(?:function[ \t]+)?([A-Za-z_]\w*)[ \t]*\(\)[ \t]*\{/g,
]

export function definedNames(text) {
  const names = new Set()
  const source = typeof text === "string" ? text : ""
  for (const re of DEF_PATTERNS) {
    re.lastIndex = 0
    let match
    while ((match = re.exec(source)) !== null) names.add(match[1])
  }
  return names
}

export function removedDefinitions(oldString, newString) {
  const before = definedNames(oldString)
  const after = definedNames(newString)
  return [...before].filter((name) => !after.has(name))
}

// Resolve the pci-hook binary: explicit override, repo venv, then PATH.
export function hookBin(directory) {
  const configured = process.env.PCI_HOOK_BIN
  if (configured) return configured
  const local = `${directory}/.venv/bin/pci-hook`
  return existsSync(local) ? local : "pci-hook"
}

// Run `pci-hook` with args, feeding `input` on stdin; resolve trimmed stdout
// ("" on any error, so callers stay silent and never break the edit).
export function runHook(bin, args, input) {
  return new Promise((resolve) => {
    let child
    try {
      child = spawn(bin, args, { stdio: ["pipe", "pipe", "ignore"] })
    } catch {
      resolve("")
      return
    }
    let out = ""
    child.on("error", () => resolve(""))
    child.stdout.on("data", (chunk) => {
      out += chunk
    })
    child.on("close", () => resolve(out))
    child.stdin.end(input || "")
  })
}
"""

EVIDENCE_JS = r"""// PCI evidence hook (phase-3, delete-first).
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
"""

REINDEX_JS = r"""// PCI background reindex hook (debounced, opt-in).
//
// Default reindex is the git post-commit hook (indexes the clean committed
// tree). This per-edit reindex is opt-in for power users who want working-tree
// freshness between commits; enable with PCI_REINDEX_ON_EDIT=1. On each
// edit/write to a source file it (re)arms a debounce timer, then asks
// `pci-hook run --behavior reindex` to refresh in the background; pci-hook
// serialises runs with a lock. Tunable: PCI_REINDEX_DEBOUNCE_MS (5000).

import { spawn } from "node:child_process"
import { SOURCE_EXT, hookBin } from "../lib/pci-evidence-logic.js"

const DEBOUNCE_MS = Number(process.env.PCI_REINDEX_DEBOUNCE_MS) || 5000
const WRITE_TOOLS = new Set(["edit", "write"])

export const PciReindex = async ({ directory }) => {
  // Opt-in: default reindex path is the git post-commit hook.
  if (!process.env.PCI_REINDEX_ON_EDIT) return {}
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
"""

# Relative path within a project's .opencode dir -> file content.
OPENCODE_FILES: dict[str, str] = {
    "lib/pci-evidence-logic.js": LIB_JS,
    "plugins/pci-evidence.js": EVIDENCE_JS,
    "plugins/pci-reindex.js": REINDEX_JS,
}
