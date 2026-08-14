"""``pci``: one binary fronting every pci-* command.

Each subcommand lazily imports and calls the same entry point the legacy
``pci-<name>`` binaries used. `pci` is the only installed executable; systems
installed before the single-binary change keep their pci-* shims (same entry
points, so configs and git hooks that reference them keep working) until they
upgrade -- internal callers fall back to those names when `pci` is absent.
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
    "audit": ("project_code_intelligence.audit", "audit_main", []),
    "check": ("project_code_intelligence.check", "check_main", []),
    "doctor": ("project_code_intelligence.doctor", "main", []),
    "hook": ("project_code_intelligence.hooks.cli", "main", []),
    "mcp": ("project_code_intelligence.server", "main", []),
    "evidence": ("project_code_intelligence.evidence", "main", []),
    "context": ("project_code_intelligence.context", "main", []),
    "smoke": ("project_code_intelligence.cli", "mcp_smoke_main", []),
    "status": ("project_code_intelligence.status_cli", "main", []),
}

# `pci services <verb>` translates to the doctor flags that already start/stop
# the local database and embedding backend.
_SERVICES = {"start": ["--start"], "stop": ["--stop"], "status": []}

# `pci embed <backend>` translates to the module/attribute that runs each
# embedding daemon or helper.
_EMBED: dict[str, tuple[str, str]] = {
    "apple": ("project_code_intelligence.embedding.apple_embed_server", "main"),
    "fastembed": ("project_code_intelligence.embedding.fastembed_server", "main"),
    "llama": ("project_code_intelligence.embedding.llama", "main"),
    "bench": ("project_code_intelligence.embedding.bench", "main"),
}


def resolve(command: str, rest: list[str]) -> tuple[tuple[str, str, list[str]], list[str]] | None:
    if command == "services":
        verb = rest[0] if rest else "status"
        flags = _SERVICES.get(verb)
        if flags is None:
            return None
        return ("project_code_intelligence.doctor", "main", flags), rest[1:]
    if command == "embed":
        backend = rest[0] if rest else ""
        entry = _EMBED.get(backend)
        if entry is None:
            return None
        module_name, attr = entry
        return (module_name, attr, []), rest[1:]
    entry = _COMMANDS.get(command)
    return (entry, rest) if entry is not None else None


def _usage(stream: IO[str]) -> None:
    lines = "\n  ".join(
        sorted([
            *_COMMANDS,
            "services (start|stop|status)",
            "embed (apple|fastembed|llama|bench)",
        ])
    )
    _ = stream.write(f"usage: pci <command> [args...]\n\ncommands:\n  {lines}\n")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in {"-h", "--help"}:
        _usage(sys.stdout if args else sys.stderr)
        return 0 if args else 2
    command = args[0]
    resolved = resolve(command, args[1:])
    if resolved is None:
        _ = sys.stderr.write(f"pci: unknown command {' '.join(args[:2])!r}\n")
        _usage(sys.stderr)
        return 2
    (module_name, attr, prefix), rest = resolved
    func = cast("Callable[[], object]", getattr(importlib.import_module(module_name), attr))
    sys.argv = [f"pci {command}", *prefix, *rest]
    result = func()
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
