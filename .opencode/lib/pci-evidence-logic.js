// Pure logic for the PCI evidence hook. Kept out of `plugins/` so opencode's
// plugin loader (which scans only `plugins/`) does not treat these helpers as
// plugin entry points. Imported by the plugin and by the node unit test.

import { existsSync } from "node:fs"

// Files whose edits are worth resolving against the code index.
export const SOURCE_EXT = /\.(py|go|sh|bash|c|h|rs|js|ts|java)$/

// Definition forms across the indexed languages; each capture group is the name.
const DEF_PATTERNS = [
  /(?:^|\n)[ \t]*(?:async[ \t]+)?(?:def|class|func|type)[ \t]+([A-Za-z_]\w*)/g, // py / go / generic
  /(?:^|\n)[ \t]*func[ \t]*\([^)]*\)[ \t]*([A-Za-z_]\w*)/g, // go method (receiver)
  /(?:^|\n)[ \t]*(?:function[ \t]+)?([A-Za-z_]\w*)[ \t]*\(\)[ \t]*\{/g, // shell function
]

// Names of definitions declared anywhere in a text blob.
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

// Definitions present before an edit but gone after it -- the deletion set.
export function removedDefinitions(oldString, newString) {
  const before = definedNames(oldString)
  const after = definedNames(newString)
  return [...before].filter((name) => !after.has(name))
}

// Resolve the pci-evidence binary: explicit override, repo venv, then PATH.
export function evidenceBin(directory) {
  const configured = process.env.PCI_EVIDENCE_BIN
  if (configured) return configured
  const local = `${directory}/.venv/bin/pci-evidence`
  return existsSync(local) ? local : "pci-evidence"
}
