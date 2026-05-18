"""Metadata extraction for infrastructure and container configuration files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

DOCKER_INSTRUCTION_RE = re.compile(r"(?im)^\s*([A-Z]+)\s+(.+)$")
DOCKER_FROM_RE = re.compile(r"(?im)^\s*FROM\s+([^\s]+)(?:\s+AS\s+([A-Za-z0-9_.-]+))?")
DOCKER_EXPOSE_RE = re.compile(r"(?im)^\s*EXPOSE\s+(.+)$")
DOCKER_COPY_RE = re.compile(r"(?im)^\s*(?:COPY|ADD)\s+(?:--[^\s]+\s+)*(.+)$")
DOCKER_ENTRYPOINT_RE = re.compile(r"(?im)^\s*(?:ENTRYPOINT|CMD)\s+(.+)$")

HCL_BLOCK_RE = re.compile(
    r'(?m)^\s*(resource|data|module|variable|output|provider|source)\s+"([^"]+)"(?:\s+"([^"]+)")?'
)
PACKER_BLOCK_RE = re.compile(r"(?m)^\s*(packer|build)\s*\{")

INFRA_METADATA_KEYS = (
    "docker_base_images",
    "docker_stages",
    "docker_instructions",
    "docker_exposed_ports",
    "docker_copy_sources",
    "docker_entrypoints",
    "terraform_resources",
    "terraform_data_sources",
    "terraform_modules",
    "terraform_variables",
    "terraform_outputs",
    "terraform_providers",
    "packer_sources",
    "packer_blocks",
    "packer_variables",
)


def docker_copy_source(value: str) -> str:
    parts = value.rsplit(None, 1)
    return parts[0] if len(parts) > 1 else value


def dockerfile_metadata(text: str) -> JsonObject:
    exposed_ports: list[str] = []
    for match in DOCKER_EXPOSE_RE.finditer(text):
        exposed_ports.extend(match.group(1).split())
    return compact_metadata({
        "docker_base_images": unique_limited(match.group(1) for match in DOCKER_FROM_RE.finditer(text)),
        "docker_stages": unique_limited(match.group(2) for match in DOCKER_FROM_RE.finditer(text) if match.group(2)),
        "docker_instructions": unique_limited(match.group(1).upper() for match in DOCKER_INSTRUCTION_RE.finditer(text)),
        "docker_exposed_ports": unique_limited(exposed_ports),
        "docker_copy_sources": unique_limited(
            docker_copy_source(match.group(1)) for match in DOCKER_COPY_RE.finditer(text)
        ),
        "docker_entrypoints": unique_limited(match.group(1).strip() for match in DOCKER_ENTRYPOINT_RE.finditer(text)),
    })


def hcl_metadata(text: str, *, packer: bool) -> JsonObject:
    resources: list[str] = []
    data_sources: list[str] = []
    modules: list[str] = []
    variables: list[str] = []
    outputs: list[str] = []
    providers: list[str] = []
    packer_sources: list[str] = []
    for match in HCL_BLOCK_RE.finditer(text):
        kind, first, second = match.group(1), match.group(2), match.group(3)
        if kind == "resource" and second:
            resources.append(f"{first}.{second}")
        elif kind == "data" and second:
            data_sources.append(f"{first}.{second}")
        elif kind == "module":
            modules.append(first)
        elif kind == "variable":
            variables.append(first)
        elif kind == "output":
            outputs.append(first)
        elif kind == "provider":
            providers.append(first)
        elif kind == "source" and second:
            packer_sources.append(f"{first}.{second}")
    return compact_metadata({
        "terraform_resources": unique_limited(resources) if not packer else [],
        "terraform_data_sources": unique_limited(data_sources) if not packer else [],
        "terraform_modules": unique_limited(modules) if not packer else [],
        "terraform_variables": unique_limited(variables) if not packer else [],
        "terraform_outputs": unique_limited(outputs) if not packer else [],
        "terraform_providers": unique_limited(providers) if not packer else [],
        "packer_sources": unique_limited(packer_sources),
        "packer_blocks": unique_limited(match.group(1) for match in PACKER_BLOCK_RE.finditer(text)) if packer else [],
        "packer_variables": unique_limited(variables) if packer else [],
    })


_DOCKERFILE_VARIANT_PREFIXES: tuple[str, ...] = (
    "dockerfile.",
    "dockerfile-",
    "containerfile.",
    "containerfile-",
)


def infra_metadata(path: str, text: str) -> JsonObject:
    file_name = Path(path).name.lower()
    normalized_path = path.lower()
    if (
        file_name in {"dockerfile", "containerfile"}
        or file_name.endswith(".dockerfile")
        or file_name.startswith(_DOCKERFILE_VARIANT_PREFIXES)
        or text.lstrip().lower().startswith("# syntax=docker/dockerfile")
    ):
        return dockerfile_metadata(text)
    return hcl_metadata(text, packer=normalized_path.endswith((".pkr.hcl", ".pkrvars.hcl")))


INFRA_PROFILE = LanguageProfile(
    name="infra",
    languages=frozenset({"dockerfile", "packer", "terraform"}),
    metadata_keys=INFRA_METADATA_KEYS,
    file_metadata=infra_metadata,
)
