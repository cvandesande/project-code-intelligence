"""Shared installation helpers for project-local Pi extensions."""

from __future__ import annotations

from pathlib import Path


def install_extension(
    project: Path, name: str, *, pci_command: str, uninstall: bool, dry_run: bool
) -> tuple[str, Path]:
    """Install or remove one bundled extension without touching other Pi resources."""
    target = project / ".pi" / "extensions" / f"project-code-intelligence-{name}.ts"
    existed = target.is_file()
    if uninstall:
        if not dry_run:
            target.unlink(missing_ok=True)
        return ("removed" if existed else "unchanged"), target

    asset = (Path(__file__).parent / "pi_assets" / f"{name}.ts").read_text(encoding="utf-8")
    content = asset.replace("__PCI_COMMAND__", pci_command.replace("\\", "\\\\").replace('"', '\\"'))
    if existed and target.read_text(encoding="utf-8") == content:
        return "unchanged", target
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        _ = target.write_text(content, encoding="utf-8")
    return ("updated" if existed else "installed"), target
