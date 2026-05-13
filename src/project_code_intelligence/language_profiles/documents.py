"""Metadata extraction for Markdown and reStructuredText documents."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

MARKDOWN_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]+\]\(([^)]+)\)")
MARKDOWN_FENCE_RE = re.compile(r"(?m)^```+([A-Za-z0-9_+.-]+)?")
RST_HEADING_RE = re.compile(r"(?m)^(.+)\n[=\-~^`:#*+]{3,}\s*$")
RST_LINK_RE = re.compile(r"`[^`<]+<([^>]+)>`_")
RST_CODE_BLOCK_RE = re.compile(r"(?m)^\s*\.\.\s+code-block::\s+([A-Za-z0-9_+.-]+)")

DOC_METADATA_KEYS = (
    "doc_headings",
    "doc_links",
    "doc_fenced_languages",
)


def doc_file_metadata(path: str, text: str) -> JsonObject:
    suffix = Path(path).suffix.lower()
    if suffix == ".rst":
        headings = [match.group(1).strip() for match in RST_HEADING_RE.finditer(text)]
        links = [match.group(1).strip() for match in RST_LINK_RE.finditer(text)]
        code_languages = [match.group(1).strip() for match in RST_CODE_BLOCK_RE.finditer(text)]
    else:
        headings = [match.group(1).strip() for match in MARKDOWN_HEADING_RE.finditer(text)]
        links = [match.group(1).strip() for match in MARKDOWN_LINK_RE.finditer(text)]
        code_languages = [match.group(1).strip() for match in MARKDOWN_FENCE_RE.finditer(text) if match.group(1)]
    return compact_metadata({
        "doc_headings": unique_limited(headings),
        "doc_links": unique_limited(links),
        "doc_fenced_languages": unique_limited(code_languages),
    })


DOC_PROFILE = LanguageProfile(
    name="documents",
    languages=frozenset({"doc"}),
    metadata_keys=DOC_METADATA_KEYS,
    file_metadata=doc_file_metadata,
)
