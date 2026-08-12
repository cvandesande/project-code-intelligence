"""Install / remove the pci hooks in an agent's configuration.

opencode: write the plugin + lib files under ``<project>/.opencode``.
Claude Code: merge a ``PreToolUse`` evidence handler into ``settings.json``.
git: write a ``post-commit`` hook that reindexes the clean committed tree.

Reindex is a git ``post-commit`` concern, not a per-edit one: indexing runs
once per commit against the committed tree (no dirty snapshots), which matches
the snapshot-per-commit model. Evidence stays agent-specific.

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

# Claude evidence fires PreToolUse (preventive); reindex is on the git post-commit hook, not here.
_CLAUDE_EDIT_MATCHER = "Edit|Write"
_EVIDENCE_ARGS = ["run", "--target", "claude", "--behavior", "evidence"]

# Managed block markers so post-commit edits never clobber a user's own script.
_POSTCOMMIT_BEGIN = "# >>> pci-hook reindex (managed) >>>"
_POSTCOMMIT_END = "# <<< pci-hook reindex (managed) <<<"


@dataclass
class InstallOutcome:
    agent: str
    action: str  # "installed" | "updated" | "removed" | "unchanged"
    target: str
    rows: list[tuple[str, str]] = field(default_factory=list)


# --- shared helpers -------------------------------------------------------------


def _hook_command() -> list[str]:
    """Absolute command prefix invoking the hook runtime, PATH-independent.

    Prefer the consolidated ``pci hook`` via the uv-tool shim (`make
    tool-install`): it survives venv rebuilds and works from any directory.
    Legacy pci-hook binaries are the fallback for installs without the shim.
    """
    for name, extra in (("pci", ["hook"]), ("pci-hook", [])):
        tool_shim = Path.home() / ".local" / "bin" / name
        if tool_shim.exists():
            return [str(tool_shim), *extra]
    invoked = Path(sys.argv[0]) if sys.argv and sys.argv[0] else None
    if invoked is not None and invoked.name.startswith("pci-hook") and invoked.exists():
        return [str(invoked.resolve())]
    for name, extra in (("pci", ["hook"]), ("pci-hook", [])):
        beside_python = Path(sys.executable).with_name(name)
        if beside_python.exists():
            return [str(beside_python), *extra]
        found = shutil.which(name)
        if found:
            return [found, *extra]
    return ["pci", "hook"]


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
    if args and args[0] == "hook":  # consolidated `pci hook run ...` spelling
        args = args[1:]
    if not args or args[0] != "run":
        return False
    for flag in ("--target", "--agent"):  # --agent: configs from pre-consolidation installs
        if flag in args:
            index = args.index(flag)
            return len(args) > index + 1 and args[index + 1] == "claude"
    return False


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


def _evidence_group(command: list[str]) -> dict[str, object]:
    return {
        "matcher": _CLAUDE_EDIT_MATCHER,
        "hooks": [{"type": "command", "command": command[0], "args": [*command[1:], *_EVIDENCE_ARGS]}],
    }


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

    # Strip our handlers from every event; this also migrates away legacy
    # PostToolUse evidence and Stop reindex handlers from older installs.
    pre = _strip_pci_groups(_as_list(hooks.get("PreToolUse")))
    post = _strip_pci_groups(_as_list(hooks.get("PostToolUse")))
    stop = _strip_pci_groups(_as_list(hooks.get("Stop")))

    if uninstall:
        action = "removed" if existed else "unchanged"
        rows = [("PreToolUse", "evidence")] if existed else [("state", "no pci hooks present")]
    else:
        command = _hook_command()
        pre.append(_evidence_group(command))
        action = "updated" if existed else "installed"
        rows = [("PreToolUse", f"{_CLAUDE_EDIT_MATCHER} -> evidence"), ("command", " ".join(command))]

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


# --- git post-commit ------------------------------------------------------------


def _strip_managed_block(text: str) -> str:
    out: list[str] = []
    skip = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped == _POSTCOMMIT_BEGIN:
            skip = True
            continue
        if stripped == _POSTCOMMIT_END:
            skip = False
            continue
        if not skip:
            out.append(line)
    return "".join(out)


def install_git(repo: Path, *, uninstall: bool, dry_run: bool) -> InstallOutcome:
    hook_path = repo / ".git" / "hooks" / "post-commit"
    existing = hook_path.read_text(encoding="utf-8") if hook_path.exists() else ""
    had_block = _POSTCOMMIT_BEGIN in existing
    base = _strip_managed_block(existing)

    if uninstall:
        action = "removed" if had_block else "unchanged"
        rows = [("post-commit", str(hook_path))] if had_block else [("state", "no pci post-commit hook present")]
        # Drop a file that was only ever ours; keep any pre-existing user script.
        leftover_is_ours = base.strip() in {"", "#!/bin/sh"}
        if not dry_run:
            if had_block and leftover_is_ours:
                hook_path.unlink(missing_ok=True)
            elif had_block:
                _ = hook_path.write_text(base, encoding="utf-8")
        return InstallOutcome("git", action, str(hook_path), rows)

    command = _hook_command()
    if not base.strip():
        base = "#!/bin/sh\n"
    elif not base.startswith("#!"):
        base = "#!/bin/sh\n" + base
    if not base.endswith("\n"):
        base += "\n"
    # Background it so `git commit` returns immediately; pci-hook serialises with a lock.
    command_line = " ".join(command)
    reindex = f'{command_line} run --target git --behavior reindex --repo "$(git rev-parse --show-toplevel)"'
    block = f"{_POSTCOMMIT_BEGIN}\n{reindex} >/dev/null 2>&1 &\n{_POSTCOMMIT_END}\n"
    if not dry_run:
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        _ = hook_path.write_text(base + block, encoding="utf-8")
        hook_path.chmod(0o755)
    return InstallOutcome(
        "git",
        "updated" if had_block else "installed",
        str(hook_path),
        [("post-commit", str(hook_path)), ("command", command_line)],
    )
