// Shared helpers for the pci opencode plugins. Kept out of `plugins/` so the
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
