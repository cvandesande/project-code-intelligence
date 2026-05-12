"""Code-intelligence ingestion profiles."""

from __future__ import annotations

from project_code_intelligence.code_profiles.base import CodeIntelProfile, GenericProfile
from project_code_intelligence.code_profiles.example import ExampleProfile
from project_code_intelligence.code_profiles.registry import load_profile

__all__ = ["CodeIntelProfile", "ExampleProfile", "GenericProfile", "load_profile"]
