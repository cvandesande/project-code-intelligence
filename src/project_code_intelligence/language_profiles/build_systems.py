"""Metadata extraction for CMake and Meson build files."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

CMAKE_COMMAND_RE = re.compile(r"(?mi)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
CMAKE_PROJECT_RE = re.compile(r"(?mi)^\s*project\s*\(\s*([A-Za-z_][A-Za-z0-9_.-]*)")
CMAKE_TARGET_RE = re.compile(r"(?mi)^\s*add_(?:executable|library)\s*\(\s*([A-Za-z_][A-Za-z0-9_.-]*)")
CMAKE_PACKAGE_RE = re.compile(r"(?mi)^\s*find_package\s*\(\s*([A-Za-z_][A-Za-z0-9_.-]*)")

MESON_PROJECT_RE = re.compile(r"""(?m)^\s*project\s*\(\s*['"]([^'"]+)['"]""")
MESON_TARGET_RE = re.compile(
    r"""(?m)^\s*(?:executable|library|shared_library|static_library)\s*\(\s*['"]([^'"]+)['"]"""
)
MESON_DEPENDENCY_RE = re.compile(r"""(?m)\bdependency\s*\(\s*['"]([^'"]+)['"]""")

BUILD_SYSTEM_METADATA_KEYS = (
    "cmake_commands",
    "cmake_projects",
    "cmake_targets",
    "cmake_packages",
    "meson_projects",
    "meson_targets",
    "meson_dependencies",
)


def cmake_metadata(text: str) -> JsonObject:
    return compact_metadata({
        "cmake_commands": unique_limited(match.group(1).lower() for match in CMAKE_COMMAND_RE.finditer(text)),
        "cmake_projects": unique_limited(match.group(1) for match in CMAKE_PROJECT_RE.finditer(text)),
        "cmake_targets": unique_limited(match.group(1) for match in CMAKE_TARGET_RE.finditer(text)),
        "cmake_packages": unique_limited(match.group(1) for match in CMAKE_PACKAGE_RE.finditer(text)),
    })


def meson_metadata(text: str) -> JsonObject:
    return compact_metadata({
        "meson_projects": unique_limited(match.group(1) for match in MESON_PROJECT_RE.finditer(text)),
        "meson_targets": unique_limited(match.group(1) for match in MESON_TARGET_RE.finditer(text)),
        "meson_dependencies": unique_limited(match.group(1) for match in MESON_DEPENDENCY_RE.finditer(text)),
    })


def build_system_metadata(path: str, text: str) -> JsonObject:
    if path.endswith(("meson.build", "meson_options.txt")):
        return meson_metadata(text)
    return cmake_metadata(text)


BUILD_SYSTEM_PROFILE = LanguageProfile(
    name="build-systems",
    languages=frozenset({"cmake", "meson"}),
    metadata_keys=BUILD_SYSTEM_METADATA_KEYS,
    file_metadata=build_system_metadata,
)
