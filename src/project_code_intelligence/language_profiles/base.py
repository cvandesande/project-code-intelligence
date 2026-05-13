"""Composable language metadata profiles.

Project profiles decide how a repository should be interpreted. Language
profiles add portable facts for common source languages regardless of project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from project_code_intelligence.models import JsonObject

MAX_METADATA_ITEMS = 80


@dataclass(frozen=True)
class LanguageProfile:
    """Metadata extractor for one or more source languages."""

    name: str
    languages: frozenset[str]
    metadata_keys: tuple[str, ...]
    file_metadata: Callable[[str, str], JsonObject]


def unique_limited(values: Iterable[str], limit: int = MAX_METADATA_ITEMS) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        value = raw_value.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= limit:
            break
    return out


def compact_metadata(metadata: JsonObject) -> JsonObject:
    return {key: value for key, value in metadata.items() if value}
