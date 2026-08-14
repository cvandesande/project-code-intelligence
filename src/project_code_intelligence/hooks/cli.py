"""``pci-hook`` command: install the hooks, and serve as the hook runtime.

    pci hook install --target opencode|claude|codex|git [--project DIR | --user]
    pci hook run     --target opencode|claude|codex|git --behavior evidence|reindex

``install`` renders a Rich summary panel in the shared pci style; ``run`` is
machine-facing and writes only the agent's injection payload to stdout.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Group

from project_code_intelligence import console_ui
from project_code_intelligence.hooks import install as install_mod
from project_code_intelligence.hooks import runtime

if TYPE_CHECKING:
    from project_code_intelligence.console_ui import PillKind

_AGENTS = ("opencode", "claude", "codex", "git")
_BEHAVIORS = ("evidence", "reindex", "banner")
_COLOR_FORCE: dict[str, bool | None] = {"auto": None, "always": True, "never": False}


@dataclass
class HookNamespace(argparse.Namespace):
    command: str | None = None
    agent: str | None = None
    behavior: str | None = None
    project: str | None = None
    repo: str | None = None
    user: bool = False
    uninstall: bool = False
    dry_run: bool = False
    color: str = "auto"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pci-hook", description="Install and run pci editor-agent hooks.")
    sub = parser.add_subparsers(dest="command", required=True)

    install = sub.add_parser("install", help="Wire the pci hooks into a target's config.")
    _ = install.add_argument("--target", "--agent", dest="agent", required=True, choices=_AGENTS)
    _ = install.add_argument("--project", help="Project directory to install into (default: current directory).")
    _ = install.add_argument(
        "--user", action="store_true", help="Install Claude or Codex hooks in the user-level configuration."
    )
    _ = install.add_argument("--uninstall", action="store_true", help="Remove the pci hooks instead of adding them.")
    _ = install.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    _ = install.add_argument("--color", choices=("auto", "always", "never"), default="auto")

    run = sub.add_parser("run", help="Runtime invoked by the installed hook (reads stdin, writes stdout).")
    _ = run.add_argument("--target", "--agent", dest="agent", required=True, choices=_AGENTS)
    _ = run.add_argument("--behavior", required=True, choices=_BEHAVIORS)
    _ = run.add_argument("--repo", help="Repository root for reindex (default: current directory).")
    return parser


# --- install rendering ----------------------------------------------------------

_ACTION_PILL: dict[str, PillKind] = {
    "installed": "ok",
    "updated": "ok",
    "removed": "ok",
    "unchanged": "warn",
}


def _render_outcome(outcome: install_mod.InstallOutcome, *, dry_run: bool, color: bool) -> None:
    console = console_ui.build_console(color=color)
    pill_label = ("would " if dry_run else "") + outcome.action
    header = console_ui.header_row(f"pci hook install · {outcome.agent}", _ACTION_PILL[outcome.action], pill_label)
    grid = console_ui.section_grid()
    console_ui.add_row(grid, "target", outcome.target)
    for label, detail in outcome.rows:
        console_ui.add_row(grid, label, detail)
    console.print(console_ui.main_panel(Group(header, grid)))


def prompt_claude_scope() -> bool:
    """Ask user vs project scope. Returns True for user (global) scope."""
    cwd = Path.cwd().resolve()
    is_project = (cwd / ".git").exists() or (cwd / ".claude").is_dir()
    if not is_project:
        _ = sys.stderr.write(f"pci-hook: {cwd} does not look like a project; installing to user settings (~/.claude)\n")
        return True
    _ = sys.stderr.write(
        "Install Claude Code hooks where?\n"
        "  [g] globally (~/.claude/settings.json)\n"
        f"  [p] this project ({cwd / '.claude' / 'settings.json'})\n"
        "Choice [g/p]: "
    )
    reply = sys.stdin.readline().strip().lower()
    return reply not in {"p", "project"}


def prompt_nested_repos(nested: list[Path], root: Path, *, uninstall: bool) -> bool:
    """Ask whether to apply the git hook change to nested repos. Default: yes."""
    verb = "Also remove the reindex hook from" if uninstall else "Also install the reindex hook in"
    listing = "".join(f"  {p.relative_to(root)}\n" for p in nested)
    _ = sys.stderr.write(f"{verb} {len(nested)} nested git repo(s)?\n{listing}Choice [Y/n]: ")
    return sys.stdin.readline().strip().lower() not in {"n", "no"}


def _install_git_with_nested(parsed: HookNamespace) -> list[install_mod.InstallOutcome]:
    repo = Path(parsed.project or ".").resolve()
    outcomes = [install_mod.install_git(repo, uninstall=parsed.uninstall, dry_run=parsed.dry_run)]
    nested = install_mod.find_nested_repos(repo)
    if parsed.uninstall:
        nested = [p for p in nested if install_mod.has_git_hook(p)]
    if not nested:
        return outcomes
    if not (sys.stdin.isatty() and sys.stderr.isatty()):
        # Non-interactive: never change repos the caller did not name; point at them instead.
        outcomes[0].rows.extend(("nested repo", str(p)) for p in nested)
        outcomes[0].rows.append(("hint", "run with --project <nested repo> to wire these too"))
        return outcomes
    if prompt_nested_repos(nested, repo, uninstall=parsed.uninstall):
        outcomes.extend(install_mod.install_git(p, uninstall=parsed.uninstall, dry_run=parsed.dry_run) for p in nested)
    return outcomes


def _run_install(parsed: HookNamespace) -> int:
    if parsed.user and parsed.agent not in {"claude", "codex"}:
        _ = sys.stderr.write("pci-hook: --user applies to Claude and Codex only\n")
        return 2
    if parsed.agent == "claude":
        user_scope = parsed.user
        interactive = sys.stdin.isatty() and sys.stderr.isatty()
        if not user_scope and parsed.project is None and not parsed.uninstall and interactive:
            user_scope = prompt_claude_scope()
        if user_scope:
            settings = Path.home() / ".claude" / "settings.json"
        else:
            settings = Path(parsed.project or ".").resolve() / ".claude" / "settings.json"
        outcomes = [install_mod.install_claude(settings, uninstall=parsed.uninstall, dry_run=parsed.dry_run)]
    elif parsed.agent == "codex":
        if parsed.user:
            hooks_path = Path.home() / ".codex" / "hooks.json"
        else:
            hooks_path = Path(parsed.project or ".").resolve() / ".codex" / "hooks.json"
        outcomes = [install_mod.install_codex(hooks_path, uninstall=parsed.uninstall, dry_run=parsed.dry_run)]
    elif parsed.agent == "git":
        outcomes = _install_git_with_nested(parsed)
    else:
        project = Path(parsed.project or ".").resolve()
        outcomes = [install_mod.install_opencode(project, uninstall=parsed.uninstall, dry_run=parsed.dry_run)]
    color = console_ui.should_emit_pretty(sys.stdout, force=_COLOR_FORCE[parsed.color])
    for outcome in outcomes:
        _render_outcome(outcome, dry_run=parsed.dry_run, color=color)
    return 0


def _run_runtime(parsed: HookNamespace) -> int:
    agent = parsed.agent or "opencode"
    if parsed.behavior == "reindex":
        return runtime.run_reindex(Path(parsed.repo or ".").resolve())
    if parsed.behavior == "banner":
        return runtime.run_banner(agent)
    return runtime.run_evidence(agent)


def main(argv: list[str] | None = None) -> int:
    parsed = _parser().parse_args(argv, namespace=HookNamespace())
    if parsed.command == "install":
        return _run_install(parsed)
    return _run_runtime(parsed)


if __name__ == "__main__":
    raise SystemExit(main())
