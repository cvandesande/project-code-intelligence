"""Python AST parser."""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING

from project_code_intelligence.parsers.core import SymbolChunkSpec, make_symbol_chunk
from project_code_intelligence.records import line_window_records, non_call_keywords

if TYPE_CHECKING:
    from project_code_intelligence.models import IntelEdge, IntelFile, IntelRecord

PythonDefinition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
PYTHON_SYMBOL_MAX_LINES = 220
PYTHON_SYMBOL_MAX_BODY_CHARS = 5600
PYTHON_SYMBOL_TRUNCATE_CHARS = 5560
# Dotted call expressions: `DatabaseSettings.from_env(`, `self.validate(`, `os.path.join(`.
# The optional declaration prefix marks a signature (`def start(self):`, `class Cache(Base):`)
# rather than a call, and is dropped as such: a class body holds its methods' signature lines, so
# a name-only scan made the class record call whatever `start` resolved to elsewhere in the repo.
PYTHON_CALL_SITE_RE = re.compile(
    r"\b(?:(?P<declaration>def|class)\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*\("
)


PYTHON_NON_CALL_KEYWORDS = non_call_keywords("python")


def python_class_owner(qualified_name: str, *, is_class: bool) -> str:
    """Class a ``self``/``cls`` receiver points at for a definition.

    Walks the qualified name's parents and stops at the first lowercase component, which is a
    nested function rather than a class: ``Foo.a.inner`` and ``Foo.a`` both belong to ``Foo``,
    ``Outer.Inner.method`` to ``Outer.Inner``. A class definition owns itself.
    """
    if is_class:
        return qualified_name
    class_parts: list[str] = []
    for component in qualified_name.split(".")[:-1]:
        if not component[:1].isupper():
            break
        class_parts.append(component)
    return ".".join(class_parts)


def python_referenced_symbols(body: str, *, class_owner: str) -> list[str]:
    """Call targets for one Python definition, qualified wherever the text names the receiver.

    ``DatabaseSettings.from_env(...)`` and ``self.validate(...)`` both state their receiver, and
    a name-only reference throws that away: the resolver can then only guess, and refuses when
    two classes define the same method name. ``self``/``cls`` are rewritten against the enclosing
    class so the target equals the stored symbol, ``cls(...)`` names the class itself, and other
    dotted calls keep the receiver and the member (``os.path.join`` -> ``path.join``), which the
    resolver still falls back to by member. A ``def``/``class`` signature is a definition, not a
    call, so it yields no reference.
    """
    refs: set[str] = set()
    for match in PYTHON_CALL_SITE_RE.finditer(body):
        if match.group("declaration") is not None:
            continue
        parts = [part.strip() for part in match.group("name").split(".") if part.strip()]
        if not parts or parts[0] in PYTHON_NON_CALL_KEYWORDS:
            continue
        if parts[0] in {"self", "cls"} and class_owner:
            # `cls(...)`/`self(...)` is the receiver itself; otherwise the called member is the
            # component right after it, so `self.a.b()` is a call on `a`, not on `b`.
            refs.add(class_owner if len(parts) == 1 else f"{class_owner}.{parts[1]}")
        elif len(parts) == 1:
            refs.add(parts[0])
        else:
            refs.add(f"{parts[-2]}.{parts[-1]}")
    return sorted(refs)[:160]


def python_node_start_lineno(node: PythonDefinition) -> int:
    line_start = node.lineno
    decorators = node.decorator_list
    if decorators:
        line_start = min(line_start, *(decorator.lineno for decorator in decorators))
    return int(line_start)


def iter_python_definitions(tree: ast.AST) -> list[tuple[PythonDefinition, str]]:
    definitions: list[tuple[PythonDefinition, str]] = []

    def visit(node: ast.AST, parents: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualified_name = ".".join([*parents, child.name])
                definitions.append((child, qualified_name))
                visit(child, [*parents, child.name])
            else:
                visit(child, parents)

    visit(tree, [])
    return sorted(
        definitions,
        key=lambda item: (python_node_start_lineno(item[0]), getattr(item[0], "col_offset", 0), item[1]),
    )


def python_records(
    intel_file: IntelFile, text: str, max_chars: int, overlap_lines: int
) -> tuple[list[IntelRecord], list[IntelEdge]]:
    records: list[IntelRecord] = []
    edges: list[IntelEdge] = []
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return line_window_records(intel_file, text, max_chars, overlap_lines), edges
    for node, qualified_name in iter_python_definitions(tree):
        line_start = python_node_start_lineno(node)
        line_end = int(getattr(node, "end_lineno", getattr(node, "lineno", line_start)) or line_start)
        body_lines = lines[line_start - 1 : min(line_end, line_start + PYTHON_SYMBOL_MAX_LINES - 1)]
        body = "\n".join(body_lines)
        truncated = line_end - line_start + 1 > PYTHON_SYMBOL_MAX_LINES or len(body) > PYTHON_SYMBOL_MAX_BODY_CHARS
        if len(body) > PYTHON_SYMBOL_MAX_BODY_CHARS:
            body = body[:PYTHON_SYMBOL_TRUNCATE_CHARS].rstrip() + "\n# symbol candidate truncated"
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        symbol, chunk, symbol_edges = make_symbol_chunk(
            intel_file,
            SymbolChunkSpec(
                language_label="Python",
                name=qualified_name,
                kind=kind,
                line_start=line_start,
                line_end=line_start + len(body_lines) - 1,
                body=body,
                referenced_symbols=python_referenced_symbols(
                    body,
                    class_owner=python_class_owner(qualified_name, is_class=isinstance(node, ast.ClassDef)),
                ),
                metadata={
                    "body_truncated": truncated,
                    "python_ast_parser": True,
                    "python_symbol_name": node.name,
                    "qualified_symbol": qualified_name,
                },
                confidence_kind="high_confidence_fact",
            ),
        )
        records.extend([symbol, chunk])
        edges.extend(symbol_edges)
    if not records:
        records.extend(line_window_records(intel_file, text, max_chars, overlap_lines))
    return records, edges
