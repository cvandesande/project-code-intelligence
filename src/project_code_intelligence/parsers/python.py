"""Python AST parser."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from project_code_intelligence.parsers.core import SymbolChunkSpec, make_symbol_chunk
from project_code_intelligence.records import line_window_records

if TYPE_CHECKING:
    from project_code_intelligence.models import IntelEdge, IntelFile, IntelRecord

PythonDefinition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
PYTHON_SYMBOL_MAX_LINES = 220
PYTHON_SYMBOL_MAX_BODY_CHARS = 5600
PYTHON_SYMBOL_TRUNCATE_CHARS = 5560


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
