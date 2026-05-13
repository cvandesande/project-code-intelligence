"""Shared building blocks for Rich-rendered CLI panels (pci-doctor, pci-index, pci-mcp-smoke)."""

from __future__ import annotations

import os
import sys
from typing import IO, TYPE_CHECKING, Literal, cast

from rich.box import ROUNDED
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Mapping

PANEL_WIDTH = 78

PillKind = Literal["running", "ok", "warn", "fail"]

STATUS_PILL_STYLES: dict[PillKind, str] = {
    "running": "bold reverse cyan",
    "ok": "bold reverse green",
    "warn": "bold reverse yellow",
    "fail": "bold reverse red",
}
STATUS_GLYPHS: dict[PillKind, str] = {"running": "⚙", "ok": "✓", "warn": "⚠", "fail": "✗"}


def build_console(*, file: IO[str] | None = None, color: bool = True) -> Console:
    return Console(
        file=file or sys.stdout,
        force_terminal=color,
        color_system="truecolor" if color else None,
        width=PANEL_WIDTH,
        highlight=False,
        legacy_windows=False,
    )


def section_grid(*, min_label_width: int = 12) -> Table:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(min_width=min_label_width, no_wrap=True)
    grid.add_column(overflow="fold")
    return grid


def add_row(grid: Table, label: str, detail: str) -> None:
    grid.add_row(Text(label, style="bold"), Text(detail))


def status_pill(kind: PillKind, label: str) -> Text:
    return Text(f" {STATUS_GLYPHS[kind]} {label} ", style=STATUS_PILL_STYLES[kind])


def header_row(title: str, kind: PillKind, label: str) -> Table:
    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_column(justify="right")
    grid.add_row(Text(title, style="bold"), status_pill(kind, label))
    return grid


def main_panel(body: RenderableType) -> Panel:
    return Panel(body, box=ROUNDED, padding=(1, 2), border_style="dim", expand=True, width=PANEL_WIDTH)


def format_count(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{value:,}"


def short_sha(sha: str | None) -> str:
    return sha[:7] if isinstance(sha, str) and sha else "—"


# === JSON traversal helpers ===============================================
# These let renderers safely walk arbitrary JSON without sprinkling isinstance
# checks. Two flavors: `as_dict`/`as_list` return None on failure (good for
# pattern-matching with `if`); `as_object` returns an empty dict (good for
# chained `.get()` traversal).


def as_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    typed = cast("dict[object, object]", value)
    return {str(k): v for k, v in typed.items()}


def as_list(value: object) -> list[object] | None:
    if not isinstance(value, list):
        return None
    return list(cast("list[object]", value))


def as_object(value: object) -> dict[str, object]:
    """Like `as_dict` but returns `{}` instead of `None` for chained `.get()` calls."""
    return as_dict(value) or {}


def coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


# === TTY / color detection =================================================


def should_emit_pretty(
    stream: object | None = None,
    *,
    force: bool | None = None,
    env: Mapping[str, str] | None = None,
    isatty: bool | None = None,
) -> bool:
    """Return True if a pretty/colored rendering should be used for `stream`.

    Args:
        stream: stream whose TTY-ness to check (defaults to sys.stdout).
        force: explicit override — True forces pretty, False forces plain, None auto-detects.
        env: environment dict (defaults to os.environ); read for NO_COLOR / FORCE_COLOR / TERM.
        isatty: explicit TTY override (useful for tests); takes precedence over `stream`.
    """
    env = os.environ if env is None else env
    if force is not None:
        return force
    if "NO_COLOR" in env:
        return False
    if env.get("FORCE_COLOR"):
        return True
    if isatty is None:
        target = stream if stream is not None else sys.stdout
        isatty = bool(getattr(target, "isatty", lambda: False)())
    return isatty and env.get("TERM") != "dumb"
