"""Example project-specific code-intelligence profile.

Profiles are the extension point for project vocabulary that should not live in
the generic indexer. Copy this file into your own module, rename the class, and
select it with a fully qualified profile name such as:

    PROJECT_CODE_INTELLIGENCE_PROFILE=my_project.code_profile:MyProjectProfile

Keep private profiles out of the public registry unless they are intended to be
part of the distributed package.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast

from project_code_intelligence.code_profiles.base import CodeIntelProfile, ProfileRecord

SERVICE_PATH_PARTS = 2
SERVICE_AREA_PATH_PARTS = 3

if TYPE_CHECKING:
    from typing_extensions import override

    from project_code_intelligence.models import IntelEdge, JsonObject
else:
    _T = TypeVar("_T")

    def override(method: _T) -> _T:
        return method


def string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in cast("list[object]", value)]
    return []


class ExampleProfile(CodeIntelProfile):
    """Small profile showing common project-level customizations."""

    name = "example"
    version = "v1"
    default_repos = (".",)

    @override
    def classify_file(self, path: str, language: str, classification: JsonObject) -> JsonObject:
        updated = dict(classification)
        parts = path.split("/")
        if len(parts) >= SERVICE_PATH_PARTS and parts[0] == "services":
            updated["file_role"] = "service"
        elif parts and parts[0] in {"deploy", "infra"} and language in {"json", "yaml", "toml"}:
            updated["file_role"] = "deployment"
        elif parts[:2] == ["docs", "architecture"]:
            updated["file_role"] = "architecture-doc"
        return updated

    @override
    def file_metadata(self, path: str, language: str, classification: JsonObject) -> JsonObject:
        del language, classification
        parts = path.split("/")
        metadata: JsonObject = {}
        if len(parts) >= SERVICE_PATH_PARTS and parts[0] == "services":
            metadata["service"] = parts[1]
            if len(parts) >= SERVICE_AREA_PATH_PARTS:
                metadata["service_area"] = parts[2]
        if parts and parts[0] in {"deploy", "infra"}:
            metadata["deployment_area"] = parts[1] if len(parts) >= SERVICE_PATH_PARTS else Path(path).stem
        return metadata

    @override
    def extra_records(
        self,
        path: str,
        source_path: str,
        language: str,
        text: str,
    ) -> tuple[list[ProfileRecord], list[IntelEdge]]:
        if language != "yaml" or not path.startswith(("deploy/", "infra/")):
            return [], []

        records: list[ProfileRecord] = []
        for match in re.finditer(r"(?m)^\s*name:\s*([A-Za-z0-9_.-]+)\s*$", text):
            name = match.group(1)
            line = text.count("\n", 0, match.start()) + 1
            records.append({
                "record_type": "deployment_object",
                "record_id": f"{source_path}::deployment_object::{name}::{line:06d}",
                "title": f"deployment object {name}",
                "summary": f"Deployment object {name} declared in {source_path}",
                "body": match.group(0).strip(),
                "line_start": line,
                "line_end": line,
                "symbol": name,
                "symbol_kind": "deployment_object",
                "metadata": {"deployment_object": name},
                "confidence_kind": "heuristic_candidate",
            })
        return records, []

    @override
    def security_context(self, path: str, language: str, file_role: str, content_class: str) -> JsonObject:
        context = super().security_context(path, language, file_role, content_class)
        if file_role == "service":
            context["security_contexts"] = sorted({*string_list(context.get("security_contexts")), "service_code"})
        if file_role == "deployment":
            context["boundary_candidates"] = sorted({
                *string_list(context.get("boundary_candidates")),
                "deployment_boundary",
            })
        return context

    @override
    def embedding_metadata_keys(self) -> list[str]:
        return [
            *super().embedding_metadata_keys(),
            "service",
            "service_area",
            "deployment_area",
            "deployment_object",
        ]
