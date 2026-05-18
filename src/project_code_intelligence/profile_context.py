"""Selected code-intelligence profile for the current ingestion run.

The active profile is a process-wide singleton mutated by `set_active_profile`.
Storage lives on the `_ProfileState` class (a single class attribute) so the
setter doesn't need the `global` keyword. Read access through
`profile_context.active_profile` is preserved via PEP 562 module-level
`__getattr__`; the `TYPE_CHECKING` declaration below gives the type-checker
the same signature that the module attribute previously had.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from project_code_intelligence.code_profiles import CodeIntelProfile, GenericProfile

if TYPE_CHECKING:
    # Tells type-checkers that `profile_context.active_profile` is a
    # CodeIntelProfile. At runtime, the read resolves through __getattr__.
    active_profile: CodeIntelProfile


class _ProfileState:
    profile: CodeIntelProfile = GenericProfile()


def set_active_profile(profile: CodeIntelProfile) -> None:
    _ProfileState.profile = profile


def repo_role_for(repo: str) -> str:
    return _ProfileState.profile.repo_role(repo)


def __getattr__(name: str) -> CodeIntelProfile:
    if name == "active_profile":
        return _ProfileState.profile
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
