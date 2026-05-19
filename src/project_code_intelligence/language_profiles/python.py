"""Portable Python metadata extraction."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

PYTHON_METADATA_KEYS = (
    "python_module",
    "python_imports",
    "python_classes",
    "python_functions",
    "python_decorators",
    "python_has_async",
)


def python_module_name(path: str) -> str:
    parts = list(Path(path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(part for part in parts if part not in {"src", "lib"}) or Path(path).stem


def decorator_name(node: ast.expr) -> str | None:
    match node:
        case ast.Name():
            return node.id
        case ast.Attribute():
            parent = decorator_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        case ast.Call():
            return decorator_name(node.func)
        case _:
            return None


def import_name(node: ast.Import | ast.ImportFrom) -> list[str]:
    match node:
        case ast.Import():
            return [alias.name for alias in node.names]
        case ast.ImportFrom():
            module = "." * node.level + (node.module or "")
            return [module] if module else []


def python_file_metadata(path: str, text: str) -> JsonObject:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {"python_module": python_module_name(path)}
    imports: list[str] = []
    classes: list[str] = []
    functions: list[str] = []
    decorators: list[str] = []
    has_async = False
    for node in ast.walk(tree):
        match node:
            case ast.Import() | ast.ImportFrom():
                imports.extend(import_name(node))
            case ast.ClassDef():
                classes.append(node.name)
                decorators.extend(name for item in node.decorator_list if (name := decorator_name(item)))
            case ast.AsyncFunctionDef():
                functions.append(node.name)
                decorators.extend(name for item in node.decorator_list if (name := decorator_name(item)))
                has_async = True
            case ast.FunctionDef():
                functions.append(node.name)
                decorators.extend(name for item in node.decorator_list if (name := decorator_name(item)))
            case _:
                pass
    return compact_metadata({
        "python_module": python_module_name(path),
        "python_imports": unique_limited(imports),
        "python_classes": unique_limited(classes),
        "python_functions": unique_limited(functions),
        "python_decorators": unique_limited(decorators),
        "python_has_async": has_async,
    })


PYTHON_PROFILE = LanguageProfile(
    name="python",
    languages=frozenset({"python"}),
    metadata_keys=PYTHON_METADATA_KEYS,
    file_metadata=python_file_metadata,
)
