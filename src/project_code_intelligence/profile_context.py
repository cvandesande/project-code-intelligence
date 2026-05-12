"""Selected code-intelligence profile for the current ingestion run."""

from __future__ import annotations

from project_code_intelligence.code_profiles import CodeIntelProfile, GenericProfile

active_profile: CodeIntelProfile = GenericProfile()


def set_active_profile(profile: CodeIntelProfile) -> None:
    global active_profile  # noqa: PLW0603 - active profile is process-wide ingest configuration.
    active_profile = profile


def repo_role_for(repo: str) -> str:
    return active_profile.repo_role(repo)
