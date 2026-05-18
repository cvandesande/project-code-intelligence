"""Code-intelligence ingestion profiles."""

from __future__ import annotations

from project_code_intelligence.code_profiles.base import CodeIntelProfile, GenericProfile
from project_code_intelligence.code_profiles.registry import load_profile

__all__ = ["CodeIntelProfile", "GenericProfile", "load_profile"]
