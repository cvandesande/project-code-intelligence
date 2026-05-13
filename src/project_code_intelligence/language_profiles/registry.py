"""Registry for portable language metadata profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.bazel import BAZEL_PROFILE
from project_code_intelligence.language_profiles.beam import BEAM_PROFILE
from project_code_intelligence.language_profiles.build_systems import BUILD_SYSTEM_PROFILE
from project_code_intelligence.language_profiles.c_family import C_FAMILY_PROFILE
from project_code_intelligence.language_profiles.csharp import C_SHARP_PROFILE
from project_code_intelligence.language_profiles.documents import DOC_PROFILE
from project_code_intelligence.language_profiles.go import GO_PROFILE
from project_code_intelligence.language_profiles.graphql import GRAPHQL_PROFILE
from project_code_intelligence.language_profiles.groovy import GROOVY_PROFILE
from project_code_intelligence.language_profiles.infra import INFRA_PROFILE
from project_code_intelligence.language_profiles.javascript import JAVASCRIPT_PROFILE
from project_code_intelligence.language_profiles.jvm import JVM_PROFILE
from project_code_intelligence.language_profiles.lua import LUA_PROFILE
from project_code_intelligence.language_profiles.markup_data import MARKUP_DATA_PROFILE
from project_code_intelligence.language_profiles.openwrt_formats import OPENWRT_FORMAT_PROFILE
from project_code_intelligence.language_profiles.perl import PERL_PROFILE
from project_code_intelligence.language_profiles.php import PHP_PROFILE
from project_code_intelligence.language_profiles.powershell import POWERSHELL_PROFILE
from project_code_intelligence.language_profiles.protobuf_objc import PROTOBUF_OBJC_PROFILE
from project_code_intelligence.language_profiles.python import PYTHON_PROFILE
from project_code_intelligence.language_profiles.ruby import RUBY_PROFILE
from project_code_intelligence.language_profiles.rust import RUST_PROFILE
from project_code_intelligence.language_profiles.scala import SCALA_PROFILE
from project_code_intelligence.language_profiles.shell import SHELL_PROFILE
from project_code_intelligence.language_profiles.swift import SWIFT_PROFILE
from project_code_intelligence.language_profiles.web import WEB_PROFILE
from project_code_intelligence.language_profiles.zig import ZIG_PROFILE

if TYPE_CHECKING:
    from project_code_intelligence.language_profiles.base import LanguageProfile
    from project_code_intelligence.models import JsonObject

LANGUAGE_PROFILES: tuple[LanguageProfile, ...] = (
    BAZEL_PROFILE,
    BEAM_PROFILE,
    BUILD_SYSTEM_PROFILE,
    C_FAMILY_PROFILE,
    C_SHARP_PROFILE,
    DOC_PROFILE,
    GO_PROFILE,
    GRAPHQL_PROFILE,
    GROOVY_PROFILE,
    INFRA_PROFILE,
    JAVASCRIPT_PROFILE,
    JVM_PROFILE,
    LUA_PROFILE,
    MARKUP_DATA_PROFILE,
    OPENWRT_FORMAT_PROFILE,
    PERL_PROFILE,
    PHP_PROFILE,
    POWERSHELL_PROFILE,
    PROTOBUF_OBJC_PROFILE,
    PYTHON_PROFILE,
    RUBY_PROFILE,
    RUST_PROFILE,
    SCALA_PROFILE,
    SHELL_PROFILE,
    SWIFT_PROFILE,
    WEB_PROFILE,
    ZIG_PROFILE,
)


def language_metadata_for_file(path: str, language: str, text: str | None) -> JsonObject:
    if text is None:
        return {}
    metadata: JsonObject = {}
    for profile in LANGUAGE_PROFILES:
        if language in profile.languages:
            metadata.update(profile.file_metadata(path, text))
    return metadata


def language_metadata_keys() -> list[str]:
    keys: list[str] = []
    for profile in LANGUAGE_PROFILES:
        for key in profile.metadata_keys:
            if key not in keys:
                keys.append(key)
    return keys
