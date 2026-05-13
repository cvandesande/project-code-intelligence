"""Metadata extraction for Protobuf and Objective-C family files."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence.language_profiles.base import LanguageProfile, compact_metadata, unique_limited

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

PROTO_PACKAGE_RE = re.compile(r"(?m)^\s*package\s+([A-Za-z_][A-Za-z0-9_.]*)\s*;")
PROTO_IMPORT_RE = re.compile(r'(?m)^\s*import\s+(?:public\s+|weak\s+)?["]([^"]+)["]\s*;')
PROTO_MESSAGE_RE = re.compile(r"(?m)^\s*message\s+([A-Za-z_][A-Za-z0-9_]*)\b")
PROTO_SERVICE_RE = re.compile(r"(?m)^\s*service\s+([A-Za-z_][A-Za-z0-9_]*)\b")
PROTO_RPC_RE = re.compile(r"(?m)^\s*rpc\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
PROTO_ENUM_RE = re.compile(r"(?m)^\s*enum\s+([A-Za-z_][A-Za-z0-9_]*)\b")

OBJC_IMPORT_RE = re.compile(r'(?m)^\s*#\s*import\s+[<"]([^>"]+)[>"]')
OBJC_INTERFACE_RE = re.compile(r"(?m)^\s*@interface\s+([A-Za-z_][A-Za-z0-9_]*)")
OBJC_IMPLEMENTATION_RE = re.compile(r"(?m)^\s*@implementation\s+([A-Za-z_][A-Za-z0-9_]*)")
OBJC_PROTOCOL_RE = re.compile(r"(?m)^\s*@protocol\s+([A-Za-z_][A-Za-z0-9_]*)")
OBJC_METHOD_RE = re.compile(r"(?m)^\s*[+-]\s*\([^)]*\)\s*([A-Za-z_][A-Za-z0-9_:]*)")

PROTOBUF_OBJC_METADATA_KEYS = (
    "proto_package",
    "proto_imports",
    "proto_messages",
    "proto_services",
    "proto_rpcs",
    "proto_enums",
    "objc_imports",
    "objc_interfaces",
    "objc_implementations",
    "objc_protocols",
    "objc_methods",
)


def protobuf_metadata(text: str) -> JsonObject:
    package_match = PROTO_PACKAGE_RE.search(text)
    return compact_metadata({
        "proto_package": package_match.group(1) if package_match else None,
        "proto_imports": unique_limited(match.group(1) for match in PROTO_IMPORT_RE.finditer(text)),
        "proto_messages": unique_limited(match.group(1) for match in PROTO_MESSAGE_RE.finditer(text)),
        "proto_services": unique_limited(match.group(1) for match in PROTO_SERVICE_RE.finditer(text)),
        "proto_rpcs": unique_limited(match.group(1) for match in PROTO_RPC_RE.finditer(text)),
        "proto_enums": unique_limited(match.group(1) for match in PROTO_ENUM_RE.finditer(text)),
    })


def objc_metadata(text: str) -> JsonObject:
    return compact_metadata({
        "objc_imports": unique_limited(match.group(1) for match in OBJC_IMPORT_RE.finditer(text)),
        "objc_interfaces": unique_limited(match.group(1) for match in OBJC_INTERFACE_RE.finditer(text)),
        "objc_implementations": unique_limited(match.group(1) for match in OBJC_IMPLEMENTATION_RE.finditer(text)),
        "objc_protocols": unique_limited(match.group(1) for match in OBJC_PROTOCOL_RE.finditer(text)),
        "objc_methods": unique_limited(match.group(1) for match in OBJC_METHOD_RE.finditer(text)),
    })


def protobuf_objc_metadata(_path: str, text: str) -> JsonObject:
    if "syntax =" in text or re.search(r"(?m)^\s*(message|service|rpc)\s+", text):
        return protobuf_metadata(text)
    return objc_metadata(text)


PROTOBUF_OBJC_PROFILE = LanguageProfile(
    name="protobuf-objc",
    languages=frozenset({"objective_c", "objective_cpp", "protobuf"}),
    metadata_keys=PROTOBUF_OBJC_METADATA_KEYS,
    file_metadata=protobuf_objc_metadata,
)
