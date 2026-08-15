import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { spawnSync } from "node:child_process";

const PCI = "__PCI_COMMAND__";

function run(behavior: "banner" | "evidence", event?: object): string {
  const result = spawnSync(PCI, ["hook", "run", "--target", "pi", "--behavior", behavior], {
    input: event ? JSON.stringify(event) : "",
    encoding: "utf8",
  });
  if (result.error) return `[pci hook unavailable: ${result.error.message}]`;
  return result.stdout.trim();
}

export default function (pi: ExtensionAPI) {
  pi.on("session_start", () => {
    const banner = run("banner");
    if (banner) pi.sendMessage({ customType: "pci-hook", content: banner, display: true }, { deliverAs: "nextTurn" });
  });

  pi.on("tool_call", (event) => {
    if (event.toolName !== "edit" && event.toolName !== "write") return;
    const evidence = run("evidence", {
      hook_event_name: "PreToolUse",
      tool_name: event.toolName,
      tool_input: event.input,
    });
    if (evidence) {
      pi.sendMessage({ customType: "pci-hook", content: evidence, display: true }, { deliverAs: "steer" });
    }
  });
}
