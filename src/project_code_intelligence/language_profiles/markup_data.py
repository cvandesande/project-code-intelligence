"""Metadata extraction for XML and SQL files."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

XML_TAG_RE = re.compile(r"<\s*([A-Za-z_][A-Za-z0-9_.:-]*)\b")
XML_ATTR_RE = re.compile(r"\s([A-Za-z_][A-Za-z0-9_.:-]*)\s*=")
SQL_TABLE_RE = re.compile(
    r"(?is)\b(?:create|alter|drop|truncate)\s+table(?:\s+if\s+(?:not\s+)?exists)?\s+([A-Za-z_][A-Za-z0-9_.$\"]*)"
)
SQL_INSERT_RE = re.compile(r"(?is)\binsert\s+into\s+([A-Za-z_][A-Za-z0-9_.$\"]*)")
SQL_INDEX_RE = re.compile(
    r"(?is)\bcreate\s+(?:unique\s+)?index(?:\s+if\s+not\s+exists)?\s+([A-Za-z_][A-Za-z0-9_.$\"]*)"
)
SQL_FUNCTION_RE = re.compile(r"(?is)\bcreate\s+(?:or\s+replace\s+)?function\s+([A-Za-z_][A-Za-z0-9_.$\"]*)")
SQL_OPERATION_RE = re.compile(r"(?is)\b(create|alter|drop|truncate|insert|update|delete|select)\b")

MARKUP_DATA_METADATA_KEYS = (
    "xml_root",
    "xml_elements",
    "xml_attributes",
    "sql_operations",
    "sql_tables",
    "sql_indexes",
    "sql_functions",
)


def xml_metadata(text: str) -> JsonObject:
    elements = [match.group(1) for match in XML_TAG_RE.finditer(text) if not match.group(1).startswith("?")]
    return compact_metadata({
        "xml_root": elements[0] if elements else None,
        "xml_elements": unique_limited(elements),
        "xml_attributes": unique_limited(match.group(1) for match in XML_ATTR_RE.finditer(text)),
    })


def sql_metadata(text: str) -> JsonObject:
    tables = [
        *(match.group(1).strip('"') for match in SQL_TABLE_RE.finditer(text)),
        *(match.group(1).strip('"') for match in SQL_INSERT_RE.finditer(text)),
    ]
    return compact_metadata({
        "sql_operations": unique_limited(match.group(1).lower() for match in SQL_OPERATION_RE.finditer(text)),
        "sql_tables": unique_limited(tables),
        "sql_indexes": unique_limited(match.group(1).strip('"') for match in SQL_INDEX_RE.finditer(text)),
        "sql_functions": unique_limited(match.group(1).strip('"') for match in SQL_FUNCTION_RE.finditer(text)),
    })


def markup_data_metadata(_path: str, text: str) -> JsonObject:
    if text.lstrip().startswith("<"):
        return xml_metadata(text)
    return sql_metadata(text)


MARKUP_DATA_PROFILE = LanguageProfile(
    name="markup-data",
    languages=frozenset({"sql", "xml"}),
    metadata_keys=MARKUP_DATA_METADATA_KEYS,
    file_metadata=markup_data_metadata,
)
