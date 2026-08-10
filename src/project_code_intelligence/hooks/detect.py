"""Detect definitions removed by an edit -- the delete trigger for evidence.

Ported from the opencode plugin's ``pci-evidence-logic.js`` so the JS shim and
the Python runtime share one notion of "what counts as a removed symbol".
Keep the two in sync: the patterns below mirror ``DEF_PATTERNS`` there.
"""

from __future__ import annotations

import re

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
