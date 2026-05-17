"""Names used indirectly through type checkers, frameworks, or dynamic loading.

Vulture scans token usage without type-checker or framework semantics. Keep this
file narrow and only add entries after manual triage.
"""

from __future__ import annotations

from http.client import HTTPResponse

from typing_extensions import LiteralString

from project_code_intelligence.code_profiles.openwrt import OpenWrtProfile
from project_code_intelligence.db import AutocommitConnect
from project_code_intelligence.embedding import apple_embed_server as _apple_embed_server


class _FrameworkHooks:
    read_only = None
    server_version = None
    model_config = None
    only_skipped = None
    include_snapshots = None
    include_record_types = None
    include_queryability = None
    include_breakdowns = None
    include_static_summary = None
    include_runtime = None
    require_exactly_one_record_selector = None
    empty_optional_strings_are_omitted = None


def _uses_protocol_keywords(*, autocommit: object, row_factory: object) -> tuple[object, object]:
    return autocommit, row_factory


_WHITELIST = (
    LiteralString,
    HTTPResponse,
    AutocommitConnect,
    OpenWrtProfile,
    _apple_embed_server._AutoTokenizerClass,
    _FrameworkHooks.read_only,
    _FrameworkHooks.server_version,
    _FrameworkHooks.model_config,
    _FrameworkHooks.only_skipped,
    _FrameworkHooks.include_snapshots,
    _FrameworkHooks.include_record_types,
    _FrameworkHooks.include_queryability,
    _FrameworkHooks.include_breakdowns,
    _FrameworkHooks.include_static_summary,
    _FrameworkHooks.include_runtime,
    _FrameworkHooks.require_exactly_one_record_selector,
    _FrameworkHooks.empty_optional_strings_are_omitted,
    _uses_protocol_keywords,
)
