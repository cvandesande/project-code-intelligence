"""Code profile registry and dynamic profile loading."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import cast

from project_code_intelligence.code_profiles.base import CodeIntelProfile, GenericProfile
from project_code_intelligence.code_profiles.example import ExampleProfile
from project_code_intelligence.exceptions import ProfileLoadError

ProfileFactory = Callable[[], object]


PROFILE_TYPES = {
    "default": GenericProfile,
    "example": ExampleProfile,
    "generic": GenericProfile,
    "none": GenericProfile,
}


def load_profile(name: str | None) -> CodeIntelProfile:
    profile_name = (name or "generic").strip()
    profile_type = PROFILE_TYPES.get(profile_name)
    if profile_type:
        return profile_type()

    if ":" not in profile_name:
        known = ", ".join(sorted(PROFILE_TYPES))
        raise ProfileLoadError(f"unknown code-intel profile {profile_name!r}; known profiles: {known}")

    module_name, attr_name = profile_name.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ProfileLoadError(f"could not import code-intel profile module {module_name!r}") from exc
    try:
        value = cast("object", getattr(module, attr_name))
    except AttributeError as exc:
        raise ProfileLoadError(f"code-intel profile {profile_name!r} was not found") from exc
    if isinstance(value, CodeIntelProfile):
        return value
    if isinstance(value, type):
        if issubclass(value, CodeIntelProfile):
            return value()
    elif callable(value):
        loaded = cast("ProfileFactory", value)()
        if isinstance(loaded, CodeIntelProfile):
            return loaded
    raise ProfileLoadError(f"{profile_name!r} did not resolve to a CodeIntelProfile")
