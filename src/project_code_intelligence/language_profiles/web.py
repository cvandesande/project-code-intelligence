"""Metadata extraction for HTML, CSS, SCSS, Vue, and Svelte files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

HTML_TAG_RE = re.compile(r"<\s*([A-Za-z][A-Za-z0-9:_-]*)\b")
HTML_ATTR_RE = re.compile(r"\s([A-Za-z_:][A-Za-z0-9_:.-]*)\s*=")
HTML_LINK_RE = re.compile(r"""(?i)\b(?:href|src)\s*=\s*["']([^"']+)["']""")
COMPONENT_BLOCK_RE = re.compile(r"(?is)<(template|script|style)\b([^>]*)>")
ATTR_VALUE_RE = re.compile(r"""(?i)\b([A-Za-z_:][A-Za-z0-9_:.-]*)\s*=\s*["']([^"']+)["']""")

CSS_IMPORT_RE = re.compile(r"""(?im)^\s*@import\s+(?:url\()?["']?([^"')\s;]+)""")
CSS_VARIABLE_RE = re.compile(r"(?m)(--[A-Za-z0-9_-]+)\s*:")
CSS_SELECTOR_RE = re.compile(r"(?m)^\s*([^@{}\n][^{}\n]*?)\s*\{")
SCSS_MIXIN_RE = re.compile(r"(?m)^\s*@mixin\s+([A-Za-z_][A-Za-z0-9_-]*)")
SCSS_INCLUDE_RE = re.compile(r"(?m)^\s*@include\s+([A-Za-z_][A-Za-z0-9_-]*)")
COMPONENT_IMPORT_RE = re.compile(r"""(?m)^\s*import\s+(?:.+?\s+from\s+)?["']([^"']+)["']""")

WEB_METADATA_KEYS = (
    "html_root",
    "html_elements",
    "html_custom_elements",
    "html_attributes",
    "html_links",
    "css_imports",
    "css_selectors",
    "css_variables",
    "scss_mixins",
    "scss_includes",
    "component_blocks",
    "component_script_languages",
    "component_style_languages",
    "component_imports",
)


def attrs_by_name(attrs: str) -> dict[str, str]:
    return {match.group(1).lower(): match.group(2) for match in ATTR_VALUE_RE.finditer(attrs)}


def html_metadata(text: str) -> JsonObject:
    elements = [match.group(1).lower() for match in HTML_TAG_RE.finditer(text)]
    custom_elements = [element for element in elements if "-" in element]
    return compact_metadata({
        "html_root": elements[0] if elements else None,
        "html_elements": unique_limited(elements),
        "html_custom_elements": unique_limited(custom_elements),
        "html_attributes": unique_limited(match.group(1).lower() for match in HTML_ATTR_RE.finditer(text)),
        "html_links": unique_limited(match.group(1) for match in HTML_LINK_RE.finditer(text)),
    })


def split_selectors(value: str) -> list[str]:
    return [selector.strip() for selector in value.split(",")]


def css_metadata(text: str) -> JsonObject:
    selectors: list[str] = []
    for match in CSS_SELECTOR_RE.finditer(text):
        selector_text = match.group(1).strip()
        if selector_text.startswith(("@", "from ", "to ")):
            continue
        selectors.extend(split_selectors(selector_text))
    return compact_metadata({
        "css_imports": unique_limited(match.group(1) for match in CSS_IMPORT_RE.finditer(text)),
        "css_selectors": unique_limited(selectors),
        "css_variables": unique_limited(match.group(1) for match in CSS_VARIABLE_RE.finditer(text)),
        "scss_mixins": unique_limited(match.group(1) for match in SCSS_MIXIN_RE.finditer(text)),
        "scss_includes": unique_limited(match.group(1) for match in SCSS_INCLUDE_RE.finditer(text)),
    })


def component_metadata(text: str) -> JsonObject:
    blocks: list[str] = []
    script_languages: list[str] = []
    style_languages: list[str] = []
    for match in COMPONENT_BLOCK_RE.finditer(text):
        block = match.group(1).lower()
        attrs = attrs_by_name(match.group(2))
        blocks.append(block)
        language = attrs.get("lang")
        if block == "script" and language:
            script_languages.append(language)
        elif block == "style" and language:
            style_languages.append(language)
    return compact_metadata({
        "component_blocks": unique_limited(blocks),
        "component_script_languages": unique_limited(script_languages),
        "component_style_languages": unique_limited(style_languages),
        "component_imports": unique_limited(match.group(1) for match in COMPONENT_IMPORT_RE.finditer(text)),
    })


def web_file_metadata(path: str, text: str) -> JsonObject:
    suffix = Path(path).suffix.lower()
    if suffix in {".css", ".scss", ".sass"}:
        return css_metadata(text)
    if suffix in {".vue", ".svelte"}:
        return compact_metadata({**html_metadata(text), **css_metadata(text), **component_metadata(text)})
    return html_metadata(text)


WEB_PROFILE = LanguageProfile(
    name="web",
    languages=frozenset({"css", "html", "scss", "svelte", "vue"}),
    metadata_keys=WEB_METADATA_KEYS,
    file_metadata=web_file_metadata,
)
