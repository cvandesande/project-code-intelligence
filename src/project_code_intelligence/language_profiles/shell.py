"""Portable shell-script metadata extraction."""

from __future__ import annotations

import re
import shlex
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

SHELL_FUNCTION_RE = re.compile(
    r"(?m)^\s*(?:(?:function\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(\s*\))?)|"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*\))\s*\{?"
)
SHELL_SOURCE_RE = re.compile(r"(?m)^\s*(?:source|\.)\s+([^#\s;]+)")
SHELL_EXPORT_RE = re.compile(r"(?m)^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)")
SHELL_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
SHELL_KEYWORDS = frozenset({
    "!",
    "[",
    "[[",
    "case",
    "do",
    "done",
    "elif",
    "else",
    "esac",
    "fi",
    "for",
    "function",
    "if",
    "in",
    "select",
    "then",
    "time",
    "until",
    "while",
    "{",
    "}",
})
SHELL_BUILTINS = frozenset({
    ":",
    ".",
    "break",
    "cd",
    "continue",
    "echo",
    "eval",
    "exec",
    "exit",
    "export",
    "local",
    "printf",
    "read",
    "readonly",
    "return",
    "set",
    "shift",
    "source",
    "test",
    "trap",
    "unset",
})
SHELL_SERVICE_NAMES = frozenset({
    "boot",
    "reload",
    "restart",
    "service_triggers",
    "start",
    "start_service",
    "stop",
    "stop_service",
})

SHELL_METADATA_KEYS = (
    "shell_shebang",
    "shell_functions",
    "shell_service_functions",
    "shell_sources",
    "shell_exports",
    "shell_commands",
)


def shell_function_names(text: str) -> list[str]:
    names: list[str] = []
    for match in SHELL_FUNCTION_RE.finditer(text):
        name = match.group(1) or match.group(2)
        if name and name not in SHELL_KEYWORDS:
            names.append(name)
    return unique_limited(names)


def strip_shell_word(value: str) -> str:
    return value.strip().strip("\"'")


def shell_command_from_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or SHELL_FUNCTION_RE.match(stripped):
        return None
    try:
        words = shlex.split(stripped, comments=True, posix=True)
    except ValueError:
        return None
    for word in words:
        if SHELL_ASSIGNMENT_RE.match(word):
            continue
        command = word.rsplit("/", 1)[-1]
        if command in SHELL_KEYWORDS or command in SHELL_BUILTINS:
            return None
        return command
    return None


def shell_file_metadata(_path: str, text: str) -> JsonObject:
    lines = text.splitlines()
    shebang = lines[0] if lines and lines[0].startswith("#!") else None
    functions = shell_function_names(text)
    return compact_metadata({
        "shell_shebang": shebang,
        "shell_functions": functions,
        "shell_service_functions": [name for name in functions if name in SHELL_SERVICE_NAMES],
        "shell_sources": unique_limited(strip_shell_word(match.group(1)) for match in SHELL_SOURCE_RE.finditer(text)),
        "shell_exports": unique_limited(match.group(1) for match in SHELL_EXPORT_RE.finditer(text)),
        "shell_commands": unique_limited(
            command for line in lines if (command := shell_command_from_line(line)) is not None
        ),
    })


SHELL_PROFILE = LanguageProfile(
    name="shell",
    languages=frozenset({"shell"}),
    metadata_keys=SHELL_METADATA_KEYS,
    file_metadata=shell_file_metadata,
)
