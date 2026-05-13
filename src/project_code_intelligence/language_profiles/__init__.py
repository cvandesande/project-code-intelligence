"""Composable source-language metadata profiles."""

from project_code_intelligence.language_profiles.base import LanguageProfile
from project_code_intelligence.language_profiles.registry import (
    LANGUAGE_PROFILES,
    language_metadata_for_file,
    language_metadata_keys,
)

__all__ = [
    "LANGUAGE_PROFILES",
    "LanguageProfile",
    "language_metadata_for_file",
    "language_metadata_keys",
]
