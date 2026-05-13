"""Portable Go metadata extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

GO_PACKAGE_RE = re.compile(r"(?m)^\s*package\s+([A-Za-z_][A-Za-z0-9_]*)")
GO_FUNC_RE = re.compile(r"(?m)^\s*func\s+(?:\(([^)]*)\)\s*)?([A-Za-z_][A-Za-z0-9_]*)(?:\[[^\]]+\])?\s*\(")
GO_TYPE_RE = re.compile(r"(?m)^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\s+(struct|interface)\b")
GO_IMPORT_LINE_RE = re.compile(r'import\s+(?:"([^"]+)"|(?:[A-Za-z_][A-Za-z0-9_]*|\.)\s+"([^"]+)")')
GO_IMPORT_PATH_RE = re.compile(r'"([^"]+)"')

GO_METADATA_KEYS = (
    "go_package",
    "go_imports",
    "go_functions",
    "go_methods",
    "go_receiver_types",
    "go_structs",
    "go_interfaces",
)


def go_import_paths(text: str) -> list[str]:
    imports: list[str] = []
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if in_block:
            if stripped.startswith(")"):
                in_block = False
                continue
            imports.extend(match.group(1) for match in GO_IMPORT_PATH_RE.finditer(stripped))
            continue
        if stripped.startswith("import") and stripped.endswith("("):
            in_block = True
            continue
        match = GO_IMPORT_LINE_RE.match(stripped)
        if match:
            imports.append(match.group(1) or match.group(2))
    return unique_limited(imports)


def receiver_type(receiver: str) -> str | None:
    names = [match.group(0) for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", receiver.replace("*", " "))]
    if not names:
        return None
    return names[-1]


def go_file_metadata(_path: str, text: str) -> JsonObject:
    package_match = GO_PACKAGE_RE.search(text)
    functions: list[str] = []
    methods: list[str] = []
    receivers: list[str] = []
    for match in GO_FUNC_RE.finditer(text):
        name = match.group(2)
        receiver = match.group(1)
        if receiver is None:
            functions.append(name)
            continue
        methods.append(name)
        typed_receiver = receiver_type(receiver)
        if typed_receiver:
            receivers.append(typed_receiver)
    structs: list[str] = []
    interfaces: list[str] = []
    for match in GO_TYPE_RE.finditer(text):
        if match.group(2) == "struct":
            structs.append(match.group(1))
        else:
            interfaces.append(match.group(1))
    return compact_metadata({
        "go_package": package_match.group(1) if package_match else None,
        "go_imports": go_import_paths(text),
        "go_functions": unique_limited(functions),
        "go_methods": unique_limited(methods),
        "go_receiver_types": unique_limited(receivers),
        "go_structs": unique_limited(structs),
        "go_interfaces": unique_limited(interfaces),
    })


GO_PROFILE = LanguageProfile(
    name="go",
    languages=frozenset({"go"}),
    metadata_keys=GO_METADATA_KEYS,
    file_metadata=go_file_metadata,
)
