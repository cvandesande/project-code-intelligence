"""Install / remove the pci hooks in an agent's configuration.

opencode: write the plugin + lib files under ``<project>/.opencode``.
Claude Code: merge a ``PreToolUse`` evidence handler into ``settings.json``.
Codex: merge ``PreToolUse`` evidence and ``SessionStart`` banner handlers into
``.codex/hooks.json``.
git: write a ``post-commit`` hook that reindexes the clean committed tree.

Reindex is a git ``post-commit`` concern, not a per-edit one: indexing runs
once per commit against the committed tree (no dirty snapshots), which matches
the snapshot-per-commit model. Evidence stays agent-specific.

Both operations are idempotent and reversible (``--uninstall``).
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from project_code_intelligence.console_ui import as_list, as_object
from project_code_intelligence.hooks.opencode_assets import OPENCODE_FILES

if TYPE_CHECKING:
    from collections.abc import Callable

# Claude evidence fires PreToolUse (preventive); reindex is on the git post-commit hook, not here.
_CLAUDE_EDIT_MATCHER = "Edit|Write"
_EVIDENCE_ARGS = ["run", "--target", "claude", "--behavior", "evidence"]
# The banner announces the regime once per session; matcher mirrors session (re)starts.
_CLAUDE_SESSION_MATCHER = "startup|resume|clear|compact"
_BANNER_ARGS = ["run", "--target", "claude", "--behavior", "banner"]
_CODEX_EDIT_MATCHER = "apply_patch"
_CODEX_EVIDENCE_ARGS = ["run", "--target", "codex", "--behavior", "evidence"]
_CODEX_BANNER_ARGS = ["run", "--target", "codex", "--behavior", "banner"]

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
    return _is_pci_handler_for(handler, "claude")


def _is_pci_handler_for(handler: object, target: str) -> bool:
    obj = as_object(handler)
    if obj.get("type") != "command":
        return False
    # Current spelling: one command string. Legacy spelling: command + "args"
    # list (Claude Code ignores "args", so those installs were inert).
    args = [item for item in as_list(obj.get("args")) or [] if isinstance(item, str)]
    if not args and isinstance(command := obj.get("command"), str):
        try:
            args = shlex.split(command)[1:]
        except ValueError:
            return False
    if args and args[0] == "hook":  # consolidated `pci hook run ...` spelling
        args = args[1:]
    if not args or args[0] != "run":
        return False
    for flag in ("--target", "--agent"):  # --agent: configs from pre-consolidation installs
        if flag in args:
            index = args.index(flag)
            return len(args) > index + 1 and args[index + 1] == target
    return False


def _strip_groups(groups: list[object], is_managed: Callable[[object], bool]) -> list[object]:
    cleaned: list[object] = []
    for group in groups:
        obj = as_object(group)
        handlers = [handler for handler in as_list(obj.get("hooks")) or [] if not is_managed(handler)]
        if handlers:
            obj["hooks"] = handlers
            cleaned.append(obj)
    return cleaned


def _strip_pci_groups(groups: list[object]) -> list[object]:
    """Drop our handlers from each matcher group, then drop emptied groups."""
    return _strip_groups(groups, _is_pci_handler)


def _strip_target_groups(groups: list[object], target: str) -> list[object]:
    return _strip_groups(groups, lambda handler: _is_pci_handler_for(handler, target))


def _evidence_group(command: list[str]) -> dict[str, object]:
    return {
        "matcher": _CLAUDE_EDIT_MATCHER,
        # Claude Code's hook schema takes one command string; an "args" key is ignored.
        "hooks": [
            {
                "type": "command",
                "command": shlex.join([*command, *_EVIDENCE_ARGS]),
            }
        ],
    }


def _banner_group(command: list[str]) -> dict[str, object]:
    return {
        "matcher": _CLAUDE_SESSION_MATCHER,
        "hooks": [{"type": "command", "command": shlex.join([*command, *_BANNER_ARGS])}],
    }


def _load_settings(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        loaded = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return {}
    return as_object(loaded)


def install_claude(settings_path: Path, *, uninstall: bool, dry_run: bool) -> InstallOutcome:
    data = _load_settings(settings_path)
    hooks = as_object(data.get("hooks"))
    existed = any(
        _is_pci_handler(handler)
        for event_groups in hooks.values()
        for group in as_list(event_groups) or []
        for handler in as_list(as_object(group).get("hooks")) or []
    )

    # Strip our handlers from every event; this also migrates away legacy
    # PostToolUse evidence and Stop reindex handlers from older installs.
    pre = _strip_pci_groups(as_list(hooks.get("PreToolUse")) or [])
    post = _strip_pci_groups(as_list(hooks.get("PostToolUse")) or [])
    stop = _strip_pci_groups(as_list(hooks.get("Stop")) or [])
    session = _strip_pci_groups(as_list(hooks.get("SessionStart")) or [])

    if uninstall:
        action = "removed" if existed else "unchanged"
        rows = (
            [("PreToolUse", "evidence"), ("SessionStart", "banner")] if existed else [("state", "no pci hooks present")]
        )
    else:
        command = _hook_command()
        pre.append(_evidence_group(command))
        session.append(_banner_group(command))
        action = "updated" if existed else "installed"
        rows = [
            ("PreToolUse", f"{_CLAUDE_EDIT_MATCHER} -> evidence"),
            ("SessionStart", f"{_CLAUDE_SESSION_MATCHER} -> banner"),
            ("command", " ".join(command)),
        ]

    _assign_event(hooks, "PreToolUse", pre)
    _assign_event(hooks, "PostToolUse", post)
    _assign_event(hooks, "Stop", stop)
    _assign_event(hooks, "SessionStart", session)
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


# --- Codex ----------------------------------------------------------------------


def install_codex(hooks_path: Path, *, uninstall: bool, dry_run: bool) -> InstallOutcome:
    data = _load_settings(hooks_path)
    hooks = as_object(data.get("hooks"))
    existed = any(
        _is_pci_handler_for(handler, "codex")
        for event_groups in hooks.values()
        for group in as_list(event_groups) or []
        for handler in as_list(as_object(group).get("hooks")) or []
    )
    pre = _strip_target_groups(as_list(hooks.get("PreToolUse")) or [], "codex")
    session = _strip_target_groups(as_list(hooks.get("SessionStart")) or [], "codex")

    if uninstall:
        action = "removed" if existed else "unchanged"
        rows = (
            [("PreToolUse", "evidence"), ("SessionStart", "banner")] if existed else [("state", "no pci hooks present")]
        )
    else:
        command = _hook_command()
        pre.append({
            "matcher": _CODEX_EDIT_MATCHER,
            "hooks": [{"type": "command", "command": shlex.join([*command, *_CODEX_EVIDENCE_ARGS])}],
        })
        session.append({
            "matcher": _CLAUDE_SESSION_MATCHER,
            "hooks": [{"type": "command", "command": shlex.join([*command, *_CODEX_BANNER_ARGS])}],
        })
        action = "updated" if existed else "installed"
        rows = [
            ("PreToolUse", f"{_CODEX_EDIT_MATCHER} -> evidence"),
            ("SessionStart", f"{_CLAUDE_SESSION_MATCHER} -> banner"),
            ("command", " ".join(command)),
            ("trust", "review the installed commands with /hooks in Codex"),
        ]

    _assign_event(hooks, "PreToolUse", pre)
    _assign_event(hooks, "SessionStart", session)
    if hooks:
        data["hooks"] = hooks
    else:
        _ = data.pop("hooks", None)
    if not dry_run:
        hooks_path.parent.mkdir(parents=True, exist_ok=True)
        _ = hooks_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return InstallOutcome("codex", action, str(hooks_path), rows)


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


def find_nested_repos(root: Path) -> list[Path]:
    """Git repositories nested under ``root`` (root itself excluded).

    Only directories with a real ``.git`` directory count: gitfile submodules
    keep their hooks in the superproject's git dir, so a hook written under
    them would never run. Hidden directories are not descended into.
    """
    found: list[Path] = []
    for current_dir, dir_names, _ in os.walk(root):
        current = Path(current_dir)
        dir_names[:] = [name for name in dir_names if not name.startswith(".")]
        found.extend(current / name for name in dir_names if (current / name / ".git").is_dir())
    return sorted(found)


# post-commit does not fire on a fast-forward `git merge` -- git creates no new commit
# object in that case, so the reindex signal is silently absent exactly when a finished
# worktree branch lands back on the shared checkout via ff merge. post-merge exists for
# this: git guarantees it runs after every `git merge` (fast-forward or not), so it is
# installed alongside post-commit rather than instead of it.
_GIT_HOOK_NAMES = ("post-commit", "post-merge")


def _hook_paths(repo: Path) -> tuple[Path, ...]:
    return tuple(repo / ".git" / "hooks" / name for name in _GIT_HOOK_NAMES)


def has_git_hook(repo: Path) -> bool:
    """True when ``repo``'s post-commit hook carries our managed block."""
    hook_path = repo / ".git" / "hooks" / "post-commit"
    try:
        return _POSTCOMMIT_BEGIN in hook_path.read_text(encoding="utf-8")
    except OSError:
        return False


def _install_one(hook_path: Path, *, block: str, uninstall: bool, dry_run: bool) -> tuple[str, bool]:
    """Write/strip the managed block in a single hook file. Returns (action, had_block)."""
    existing = hook_path.read_text(encoding="utf-8") if hook_path.exists() else ""
    had_block = _POSTCOMMIT_BEGIN in existing
    base = _strip_managed_block(existing)

    if uninstall:
        leftover_is_ours = base.strip() in {"", "#!/bin/sh"}
        if not dry_run:
            if had_block and leftover_is_ours:
                hook_path.unlink(missing_ok=True)
            elif had_block:
                _ = hook_path.write_text(base, encoding="utf-8")
        return ("removed" if had_block else "unchanged"), had_block

    if not base.strip():
        base = "#!/bin/sh\n"
    elif not base.startswith("#!"):
        base = "#!/bin/sh\n" + base
    if not base.endswith("\n"):
        base += "\n"
    if not dry_run:
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        _ = hook_path.write_text(base + block, encoding="utf-8")
        hook_path.chmod(0o755)
    return ("updated" if had_block else "installed"), had_block


def install_git(repo: Path, *, uninstall: bool, dry_run: bool) -> InstallOutcome:
    command = _hook_command()
    command_line = " ".join(command)
    reindex = f'{command_line} run --target git --behavior reindex-submit --repo "$(git rev-parse --show-toplevel)"'
    block = f"{_POSTCOMMIT_BEGIN}\n{reindex} >/dev/null 2>&1\n{_POSTCOMMIT_END}\n"

    results = [
        (name, hook_path, *_install_one(hook_path, block=block, uninstall=uninstall, dry_run=dry_run))
        for name, hook_path in zip(_GIT_HOOK_NAMES, _hook_paths(repo), strict=True)
    ]
    any_had_block = any(had_block for _, _, _, had_block in results)

    if uninstall:
        action = "removed" if any_had_block else "unchanged"
        rows = [(name, str(path)) for name, path, _, had_block in results if had_block] or [
            ("state", "no pci git hooks present")
        ]
        return InstallOutcome("git", action, str(repo / ".git" / "hooks"), rows)

    action = "updated" if any_had_block else "installed"
    rows = [(name, str(path)) for name, path, _, _ in results]
    rows.append(("command", command_line))
    return InstallOutcome("git", action, str(repo / ".git" / "hooks"), rows)
