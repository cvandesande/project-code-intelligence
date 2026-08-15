"""Install and remove project-scoped MCP client configuration."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from project_code_intelligence import pi_support

_BEGIN = "# >>> pci mcp project-code-intelligence (managed) >>>"
_END = "# <<< pci mcp project-code-intelligence (managed) <<<"
_SERVER_NAME = "project-code-intelligence"
_TARGETS = ("codex", "claude", "opencode", "vscode", "copilot", "cline", "zed", "pi")


def _strip_managed_block(text: str) -> tuple[str, bool]:
    lines: list[str] = []
    skipping = False
    found = False
    for line in text.splitlines(keepends=True):
        if line.strip() == _BEGIN:
            skipping = True
            found = True
            continue
        if line.strip() == _END:
            skipping = False
            continue
        if not skipping:
            lines.append(line)
    return "".join(lines).rstrip(), found


def _pci_command() -> str:
    installed = Path.home() / ".local" / "bin" / "pci"
    if installed.is_file():
        return str(installed)
    return shutil.which("pci") or "pci"


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _managed_block(project: Path) -> str:
    return "\n".join((
        _BEGIN,
        f"[mcp_servers.{_SERVER_NAME}]",
        f"command = {_toml_string(_pci_command())}",
        f'args = ["mcp", "--scope", {_toml_string(str(project))}]',
        f"cwd = {_toml_string(str(project))}",
        "startup_timeout_sec = 20",
        "tool_timeout_sec = 120",
        _END,
    ))


def install_codex(project: Path, *, uninstall: bool, dry_run: bool) -> tuple[str, Path]:
    config_path = project / ".codex" / "config.toml"
    existing = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    base, had_block = _strip_managed_block(existing)
    if uninstall:
        action = "removed" if had_block else "unchanged"
        updated = base + ("\n" if base else "")
    else:
        table = f"[mcp_servers.{_SERVER_NAME}]"
        if not had_block and table in existing:
            raise ValueError(f"{config_path} already defines {table[1:-1]} outside PCI's managed block")
        action = "updated" if had_block else "installed"
        updated = (base + "\n\n" if base else "") + _managed_block(project) + "\n"
    if not dry_run:
        if uninstall and not updated.strip():
            config_path.unlink(missing_ok=True)
        else:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            _ = config_path.write_text(updated, encoding="utf-8")
    return action, config_path


def _json_target(target: str, project: Path) -> tuple[str, dict[str, object], dict[str, object]]:
    command = _pci_command()
    common: dict[str, object] = {"command": command, "args": ["mcp", "--scope", str(project)]}
    if target == "claude":
        common.update({"type": "stdio", "cwd": str(project)})
        return "mcpServers", common, {}
    if target == "opencode":
        server: dict[str, object] = {
            "type": "local",
            "command": [command, "mcp", "--scope", str(project)],
            "enabled": True,
            "cwd": str(project),
        }
        return "mcp", server, {"$schema": "https://opencode.ai/config.json"}
    if target in {"vscode", "copilot"}:
        common["type"] = "stdio"
        return "servers", common, {}
    if target == "zed":
        return "context_servers", common, {}
    common.update({"autoApprove": [], "disabled": False})
    return "mcpServers", common, {}


def _default_json_path(target: str, project: Path) -> Path:
    if target == "claude":
        return project / ".mcp.json"
    if target == "opencode":
        return project / "opencode.json"
    if target in {"vscode", "copilot"}:
        return project / ".vscode" / "mcp.json"
    if target == "zed":
        return project / ".zed" / "settings.json"
    raise ValueError("--target cline requires --config-path to its user-scoped MCP settings JSON")


def install_json_target(
    target: str, project: Path, *, config_path: Path | None, uninstall: bool, dry_run: bool
) -> tuple[str, Path]:
    path = config_path or _default_json_path(target, project)
    data = _load_json_object(path)
    section_name, server, defaults = _json_target(target, project)
    for key, value in defaults.items():
        _ = data.setdefault(key, value)
    section = _object_section(data, section_name, path)
    existed = _SERVER_NAME in section
    if uninstall:
        _ = section.pop(_SERVER_NAME, None)
        action = "removed" if existed else "unchanged"
    else:
        section[_SERVER_NAME] = server
        action = "updated" if existed else "installed"
    if section:
        data[section_name] = section
    else:
        _ = data.pop(section_name, None)
    if not dry_run:
        if uninstall and not data:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return action, path


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        loaded = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise ValueError(f"cannot merge invalid JSON in {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise TypeError(f"cannot merge non-object JSON in {path}")
    typed = cast("dict[object, object]", loaded)
    return {str(key): value for key, value in typed.items()}


def _object_section(data: dict[str, object], name: str, path: Path) -> dict[str, object]:
    raw = data.get(name)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"cannot merge non-object {name} in {path}")
    typed = cast("dict[object, object]", raw)
    return {str(key): value for key, value in typed.items()}


@dataclass
class InstallNamespace(argparse.Namespace):
    target: str = ""
    project: str = "."
    config_path: str | None = None
    uninstall: bool = False
    dry_run: bool = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pci mcp install", description="Install MCP client configuration.")
    _ = parser.add_argument("--target", required=True, choices=_TARGETS)
    _ = parser.add_argument("--project", default=".", help="Project directory (default: current directory).")
    _ = parser.add_argument("--config-path", help="Explicit client config path (required for Cline).")
    _ = parser.add_argument("--uninstall", action="store_true", help="Remove PCI's managed MCP configuration.")
    _ = parser.add_argument("--dry-run", action="store_true", help="Report the change without writing it.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parsed = _parser().parse_args(argv, namespace=InstallNamespace())
    project = Path(parsed.project).resolve()
    try:
        action, target = _run_install(parsed, project)
    except (TypeError, ValueError) as exc:
        _ = sys.stderr.write(f"pci mcp install: {exc}\n")
        return 1
    prefix = "would " if parsed.dry_run else ""
    _ = sys.stdout.write(f"{prefix}{action} {parsed.target} MCP config: {target}\n")
    if not parsed.uninstall:
        _ = sys.stdout.write(
            f"Restart {parsed.target}, then verify project-code-intelligence in its MCP server list.\n"
        )
    return 0


def _run_install(parsed: InstallNamespace, project: Path) -> tuple[str, Path]:
    config_path = Path(parsed.config_path).expanduser().resolve() if parsed.config_path else None
    if parsed.target == "codex":
        if config_path is not None:
            raise ValueError("--config-path is not supported for Codex; use --project")
        return install_codex(project, uninstall=parsed.uninstall, dry_run=parsed.dry_run)
    if parsed.target == "pi":
        if config_path is not None:
            raise ValueError("--config-path is not supported for Pi; use --project")
        return pi_support.install_extension(
            project, "mcp", pci_command=_pci_command(), uninstall=parsed.uninstall, dry_run=parsed.dry_run
        )
    return install_json_target(
        parsed.target,
        project,
        config_path=config_path,
        uninstall=parsed.uninstall,
        dry_run=parsed.dry_run,
    )
