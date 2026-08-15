import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface } from "node:readline";

const PCI = "__PCI_COMMAND__";
type Pending = { resolve(value: unknown): void; reject(error: Error): void };

export default function (pi: ExtensionAPI) {
  let child: ChildProcessWithoutNullStreams | undefined;
  let nextId = 1;
  let pending = new Map<number, Pending>();
  let started = false;

  function request(method: string, params: object = {}): Promise<any> {
    if (!child) return Promise.reject(new Error("PCI MCP server is not running"));
    const id = nextId++;
    child.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
    return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
  }

  async function start(cwd: string) {
    if (started) return;
    started = true;
    child = spawn(PCI, ["mcp", "--scope", cwd], { cwd, stdio: ["pipe", "pipe", "pipe"] });
    child.stderr.on("data", (data) => process.stderr.write(`[pci mcp] ${data}`));
    child.on("exit", (code) => {
      for (const waiter of pending.values()) waiter.reject(new Error(`PCI MCP server exited (${code})`));
      pending.clear(); child = undefined; started = false;
    });
    createInterface({ input: child.stdout }).on("line", (line) => {
      try {
        const message = JSON.parse(line);
        if (typeof message.id !== "number") return;
        const waiter = pending.get(message.id);
        if (!waiter) return;
        pending.delete(message.id);
        if (message.error) waiter.reject(new Error(message.error.message ?? JSON.stringify(message.error)));
        else waiter.resolve(message.result);
      } catch (error) { process.stderr.write(`[pci mcp] invalid response: ${String(error)}\n`); }
    });
    await request("initialize", {
      protocolVersion: "2025-06-18",
      capabilities: {},
      clientInfo: { name: "pi-project-code-intelligence", version: "1" },
    });
    child.stdin.write(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }) + "\n");
    let cursor: string | undefined;
    do {
      const result = await request("tools/list", cursor ? { cursor } : {});
      for (const tool of result.tools ?? []) {
        pi.registerTool({
          name: tool.name,
          label: tool.title ?? tool.name,
          description: tool.description ?? tool.name,
          parameters: tool.inputSchema ?? { type: "object", properties: {} },
          async execute(_id, params) {
            const response = await request("tools/call", { name: tool.name, arguments: params });
            if (response.isError) throw new Error(response.content?.map((item: any) => item.text ?? "").join("\n") || "MCP tool failed");
            return { content: response.content ?? [], details: response.structuredContent ?? {} };
          },
        });
      }
      cursor = result.nextCursor;
    } while (cursor);
  }

  pi.on("session_start", async (_event, ctx) => { await start(ctx.cwd); });
  pi.on("session_shutdown", async () => {
    if (child) { child.kill(); child = undefined; }
    started = false;
  });
}
