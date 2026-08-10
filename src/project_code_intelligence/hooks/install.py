"""Install / remove the pci hooks in an agent's configuration.

opencode: write the plugin + lib files under ``<project>/.opencode``.
Claude Code: merge a ``PostToolUse`` (evidence) and ``Stop`` (reindex) handler
into ``settings.json`` (project or user scope).

Both operations are idempotent and reversible (``--uninstall``).
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from project_code_intelligence.hooks.opencode_assets import OPENCODE_FILES

# Claude Code hook wiring. Evidence fires PreToolUse so the agent sees the blast
# radius before the delete lands (preventive), not just after.
_MIN_PCI_ARGS = 4  # run --agent <name> ...
_CLAUDE_EDIT_MATCHER = "Edit|Write"
_EVIDENCE_ARGS = ["run", "--agent", "claude", "--behavior", "evidence"]
_REINDEX_ARGS = ["run", "--agent", "claude", "--behavior", "reindex", "--repo", "${CLAUDE_PROJECT_DIR}"]


@dataclass
class InstallOutcome:
    agent: str
    action: str  # "installed" | "updated" | "removed" | "unchanged"
    target: str
    rows: list[tuple[str, str]] = field(default_factory=list)


# --- shared helpers -------------------------------------------------------------


def _hook_command() -> str:
    """Absolute path to this pci-hook, so the agent config does not depend on PATH."""
    invoked = Path(sys.argv[0]) if sys.argv and sys.argv[0] else None
    if invoked is not None and invoked.name.startswith("pci-hook") and invoked.exists():
        return str(invoked.resolve())
    beside_python = Path(sys.executable).with_name("pci-hook")
    if beside_python.exists():
        return str(beside_python)
    found = shutil.which("pci-hook")
    return found or "pci-hook"


def _as_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    typed = cast("dict[object, object]", value)
    return {str(k): v for k, v in typed.items()}


def _as_list(value: object) -> list[object]:
    if not isinstance(value, list):
        return []
    return list(cast("list[object]", value))


# --- opencode -------------------------------------------------------------------


def install_opencode(project: Path, *, uninstall: bool, dry_run: bool) -> InstallOutcome:
    base = project / ".opencode"
    targets = {rel: base / rel for rel in OPENCODE_FILES}
    if uninstall:
        removed = [rel for rel, path in targets.items() if path.exists()]
        if not dry_run:
            for rel in removed:
                targets[rel].unlink(missing_ok=True)
        action = "removed" if removed else "unchanged"
        rows = [("plugin", rel) for rel in removed] or [("state", "no pci plugins present")]
        return InstallOutcome("opencode", action, str(base), rows)

    existed = all(path.exists() for path in targets.values())
    if not dry_run:
        for rel, content in OPENCODE_FILES.items():
            path = targets[rel]
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_text(content, encoding="utf-8")
    return InstallOutcome(
        "opencode",
        "updated" if existed else "installed",
        str(base),
        [("plugin", rel) for rel in OPENCODE_FILES],
    )


# --- Claude Code ----------------------------------------------------------------


def _is_pci_handler(handler: object) -> bool:
    obj = _as_object(handler)
    if obj.get("type") != "command":
        return False
    args = [item for item in _as_list(obj.get("args")) if isinstance(item, str)]
    return (
        len(args) >= _MIN_PCI_ARGS
        and args[0] == "run"
        and "--agent" in args
        and args[args.index("--agent") + 1] == "claude"
    )


def _strip_pci_groups(groups: list[object]) -> list[object]:
    """Drop our handlers from each matcher group, then drop emptied groups."""
    cleaned: list[object] = []
    for group in groups:
        obj = _as_object(group)
        handlers = [h for h in _as_list(obj.get("hooks")) if not _is_pci_handler(h)]
        if handlers:
            obj["hooks"] = handlers
            cleaned.append(obj)
    return cleaned


def _evidence_group(command: str) -> dict[str, object]:
    return {
        "matcher": _CLAUDE_EDIT_MATCHER,
        "hooks": [{"type": "command", "command": command, "args": list(_EVIDENCE_ARGS)}],
    }


def _reindex_group(command: str) -> dict[str, object]:
    return {"hooks": [{"type": "command", "command": command, "args": list(_REINDEX_ARGS), "async": True}]}


def _load_settings(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        loaded = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return {}
    return _as_object(loaded)


def install_claude(settings_path: Path, *, uninstall: bool, dry_run: bool) -> InstallOutcome:
    data = _load_settings(settings_path)
    hooks = _as_object(data.get("hooks"))
    existed = any(
        _is_pci_handler(handler)
        for event_groups in hooks.values()
        for group in _as_list(event_groups)
        for handler in _as_list(_as_object(group).get("hooks"))
    )

    # Strip our handlers from both events: evidence now lives on PreToolUse, so
    # this also migrates away any legacy PostToolUse evidence handler.
    pre = _strip_pci_groups(_as_list(hooks.get("PreToolUse")))
    post = _strip_pci_groups(_as_list(hooks.get("PostToolUse")))
    stop = _strip_pci_groups(_as_list(hooks.get("Stop")))

    if uninstall:
        action = "removed" if existed else "unchanged"
        rows = [("PreToolUse", "evidence"), ("Stop", "reindex")] if existed else [("state", "no pci hooks present")]
    else:
        command = _hook_command()
        pre.append(_evidence_group(command))
        stop.append(_reindex_group(command))
        action = "updated" if existed else "installed"
        rows = [
            ("PreToolUse", f"{_CLAUDE_EDIT_MATCHER} -> evidence"),
            ("Stop", "reindex (async)"),
            ("command", command),
        ]

    _assign_event(hooks, "PreToolUse", pre)
    _assign_event(hooks, "PostToolUse", post)
    _assign_event(hooks, "Stop", stop)
    if hooks:
        data["hooks"] = hooks
    else:
        _ = data.pop("hooks", None)

    if not dry_run:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        _ = settings_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return InstallOutcome("claude", action, str(settings_path), rows)


def _assign_event(hooks: dict[str, object], event: str, groups: list[object]) -> None:
    if groups:
        hooks[event] = groups
    else:
        _ = hooks.pop(event, None)
