"""Portable JavaScript and TypeScript metadata extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

JS_IMPORT_FROM_RE = re.compile(r"""(?m)^\s*import\s+(?:type\s+)?(?:.+?\s+from\s+)?["']([^"']+)["']""")
JS_REQUIRE_RE = re.compile(r"""(?:require|import)\(\s*["']([^"']+)["']\s*\)""")
JS_FUNCTION_RE = re.compile(r"(?m)^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
JS_ARROW_FUNCTION_RE = re.compile(
    r"(?m)^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*="
    r"\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][A-Za-z0-9_$]*)\s*=>"
)
JS_CLASS_RE = re.compile(r"(?m)^\s*(?:export\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)\b")
JS_EXPORT_RE = re.compile(
    r"(?m)^\s*export\s+(?:default\s+)?(?:async\s+)?(?:function|class|const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)"
)
TS_INTERFACE_RE = re.compile(r"(?m)^\s*(?:export\s+)?interface\s+([A-Za-z_$][A-Za-z0-9_$]*)\b")
TS_TYPE_RE = re.compile(r"(?m)^\s*(?:export\s+)?type\s+([A-Za-z_$][A-Za-z0-9_$]*)\b")
TS_ENUM_RE = re.compile(r"(?m)^\s*(?:export\s+)?enum\s+([A-Za-z_$][A-Za-z0-9_$]*)\b")

JAVASCRIPT_METADATA_KEYS = (
    "js_imports",
    "js_exports",
    "js_functions",
    "js_classes",
    "ts_interfaces",
    "ts_types",
    "ts_enums",
)


def js_file_metadata(_path: str, text: str) -> JsonObject:
    imports = [
        *(match.group(1) for match in JS_IMPORT_FROM_RE.finditer(text)),
        *(match.group(1) for match in JS_REQUIRE_RE.finditer(text)),
    ]
    functions = [
        *(match.group(1) for match in JS_FUNCTION_RE.finditer(text)),
        *(match.group(1) for match in JS_ARROW_FUNCTION_RE.finditer(text)),
    ]
    return compact_metadata({
        "js_imports": unique_limited(imports),
        "js_exports": unique_limited(match.group(1) for match in JS_EXPORT_RE.finditer(text)),
        "js_functions": unique_limited(functions),
        "js_classes": unique_limited(match.group(1) for match in JS_CLASS_RE.finditer(text)),
        "ts_interfaces": unique_limited(match.group(1) for match in TS_INTERFACE_RE.finditer(text)),
        "ts_types": unique_limited(match.group(1) for match in TS_TYPE_RE.finditer(text)),
        "ts_enums": unique_limited(match.group(1) for match in TS_ENUM_RE.finditer(text)),
    })


JAVASCRIPT_PROFILE = LanguageProfile(
    name="javascript",
    languages=frozenset({"javascript", "typescript"}),
    metadata_keys=JAVASCRIPT_METADATA_KEYS,
    file_metadata=js_file_metadata,
)
