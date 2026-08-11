"""Detect definitions removed by an edit -- the delete trigger for evidence.

Ported from the opencode plugin's ``pci-evidence-logic.js`` so the JS shim and
the Python runtime share one notion of "what counts as a removed symbol".
Keep the two in sync: the patterns below mirror ``DEF_PATTERNS`` there.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

# Files whose edits are worth resolving against the code index.
SOURCE_EXT = re.compile(r"\.(?:py|go|sh|bash|c|h|rs|js|ts|java)$")

# Definition forms across the indexed languages; each capture group is the name.
_DEF_PATTERNS = (
    re.compile(r"(?:^|\n)[ \t]*(?:async[ \t]+)?(?:def|class|func|type)[ \t]+([A-Za-z_]\w*)"),
    re.compile(r"(?:^|\n)[ \t]*func[ \t]*\([^)]*\)[ \t]*([A-Za-z_]\w*)"),  # go method (receiver)
    re.compile(r"(?:^|\n)[ \t]*(?:function[ \t]+)?([A-Za-z_]\w*)[ \t]*\(\)[ \t]*\{"),  # shell function
)


def is_source_path(path: str) -> bool:
    return bool(SOURCE_EXT.search(path))


def defined_names(text: str) -> set[str]:
    """Names of definitions declared anywhere in a text blob."""
    names: set[str] = set()
    for pattern in _DEF_PATTERNS:
        names.update(match.group(1) for match in pattern.finditer(text))
    return names


def removed_definitions(old_string: str, new_string: str) -> list[str]:
    """Definitions present before an edit but gone after it -- the deletion set."""
    before = defined_names(old_string)
    after = defined_names(new_string)
    return [name for name in sorted(before) if name not in after]


def _definition_name(line: str) -> str | None:
    """The name this line defines, or None. Applies _DEF_PATTERNS -- the single source of
    truth for "what is a definition" -- to one line by giving them the newline they expect."""
    for pattern in _DEF_PATTERNS:
        match = pattern.match("\n" + line)
        if match:
            return match.group(1)
    return None


def _indent_width(line: str) -> int:
    return len(line) - len(line.lstrip())


def definition_slices(text: str, names: Iterable[str], *, max_lines: int = 60) -> dict[str, str]:
    """Source text of each named definition, for use as a semantic query.

    Embedding the whole edit payload instead of the definition would break the add-side
    numbers, not merely slow them down: recall and the distance gate were both measured on
    single-definition queries, so a blended whole-file vector sits at distances the gate
    does not describe. A Write payload is the entire file, which makes slicing mandatory
    rather than an optimisation.

    ponytail: dedent scan, not a parser. A definition ends before the first non-blank line
    indented no deeper than the definition itself. That drops a C-like closing brace, which
    costs a query vector nothing, and keeps the NEXT definition's signature out -- including
    it would blend a foreign symbol name into the vector, which matters more. A decorator or
    attached comment above the definition is not included. Swap in the language parsers if a
    slice is ever used as evidence rather than as a query.
    """
    wanted = set(names)
    lines = text.splitlines()
    out: dict[str, str] = {}
    for index, line in enumerate(lines):
        name = _definition_name(line)
        if name is None or name not in wanted or name in out:
            continue
        indent = _indent_width(line)
        end = min(index + max_lines, len(lines))
        for offset in range(index + 1, end):
            candidate = lines[offset]
            if candidate.strip() and _indent_width(candidate) <= indent:
                end = offset
                break
        body = "\n".join(lines[index:end]).rstrip()
        if body:
            out[name] = body
    return out
