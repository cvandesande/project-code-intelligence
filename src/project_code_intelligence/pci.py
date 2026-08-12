"""``pci``: one binary fronting every pci-* command.

Each subcommand lazily imports and calls the same entry point its legacy
``pci-<name>`` binary uses. The legacy binaries stay installed as
compatibility shims for configs, git hooks, and scripts that reference them.
"""

from __future__ import annotations

import importlib
import sys
from typing import IO, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

# subcommand -> (module, attribute, argv prefix). Lazy imports keep `pci index`
# from paying for embedding-backend dependencies it never touches.
_COMMANDS: dict[str, tuple[str, str, list[str]]] = {
    "index": ("project_code_intelligence.cli", "index_main", []),
    "analyze": ("project_code_intelligence.analyze", "main", []),
    "audit": ("project_code_intelligence.analyze", "main", ["audit"]),
    "doctor": ("project_code_intelligence.doctor", "main", []),
    "hook": ("project_code_intelligence.hooks.cli", "main", []),
    "serve": ("project_code_intelligence.server", "main", []),
    "evidence": ("project_code_intelligence.evidence", "main", []),
    "context": ("project_code_intelligence.context", "main", []),
    "ingest": ("project_code_intelligence.ingest_code_intel", "cli_main", []),
    "bench": ("project_code_intelligence.embedding.bench", "main", []),
    "smoke": ("project_code_intelligence.cli", "mcp_smoke_main", []),
    "llama-embed": ("project_code_intelligence.embedding.llama", "main", []),
    "apple-embed-server": ("project_code_intelligence.embedding.apple_embed_server", "main", []),
    "fastembed-server": ("project_code_intelligence.embedding.fastembed_server", "main", []),
}


def _usage(stream: IO[str]) -> None:
    lines = "\n  ".join(sorted(_COMMANDS))
    _ = stream.write(f"usage: pci <command> [args...]\n\ncommands:\n  {lines}\n")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in {"-h", "--help"}:
        _usage(sys.stdout if args else sys.stderr)
        return 0 if args else 2
    command, rest = args[0], args[1:]
    entry = _COMMANDS.get(command)
    if entry is None:
        _ = sys.stderr.write(f"pci: unknown command {command!r}\n")
        _usage(sys.stderr)
        return 2
    module_name, attr, prefix = entry
    func = cast("Callable[[], object]", getattr(importlib.import_module(module_name), attr))
    sys.argv = [f"pci {command}", *prefix, *rest]
    result = func()
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
