"""Parsers for project metadata, docs, shell, and structured files."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, cast

from project_code_intelligence import profile_context
from project_code_intelligence.common import sha256_text
from project_code_intelligence.parser_core import first_sentence, make_profile_record, string_items
from project_code_intelligence.parser_security import security_api_refs
from project_code_intelligence.records import (
    RecordSpec,
    common_extracts,
    extract_referenced_symbols,
    line_for_offset_with_index,
    line_offsets,
    line_window_records,
    make_record,
)

if TYPE_CHECKING:
    from project_code_intelligence.code_profiles.base import ProfileRecord
    from project_code_intelligence.models import IntelEdge, IntelFile, IntelRecord, JsonObject


def kconfig_block_starts(text: str) -> list[int]:
    starts = [match.start() for match in re.finditer(r"(?m)^\s*(?:menuconfig|config|choice|menu|comment)\b", text)]
    return [*starts, len(text)]


def kconfig_record_spec(
    intel_file: IntelFile,
    block: str,
    start: int,
    offsets: list[int],
) -> tuple[RecordSpec, list[str]]:
    first = block.splitlines()[0].strip()
    kind, _, name = first.partition(" ")
    symbol = name.strip().strip('"') or kind
    line_start = line_for_offset_with_index(offsets, start)
    line_end = line_start + block.count("\n")
    configs = [f"CONFIG_{symbol}"] if kind in {"config", "menuconfig"} else []
    deps = [match.group(1) for match in re.finditer(r"^\s*(?:depends on|select|imply)\s+(.+)$", block, re.MULTILINE)]
    common = common_extracts(block)
    metadata: JsonObject = {
        **common,
        "config_symbols": sorted(set(configs + string_items(common.get("config_symbols")))),
        "config_dependencies": deps[:80],
    }
    return (
        RecordSpec(
            record_type="config_symbol" if kind in {"config", "menuconfig"} else "code_chunk",
            record_id=f"{intel_file.source_path}::{kind}::{symbol}::{line_start:06d}",
            title=f"{kind} {symbol} in {intel_file.source_path}:{line_start}-{line_end}",
            summary=f"Kconfig {kind} {symbol}",
            body=block,
            line_start=line_start,
            line_end=line_end,
            symbol=f"CONFIG_{symbol}" if kind in {"config", "menuconfig"} else symbol,
            symbol_kind="kconfig" if kind in {"config", "menuconfig"} else kind,
            metadata=metadata,
        ),
        deps,
    )


def kconfig_dependency_record(intel_file: IntelFile, parent: IntelRecord, dep: str) -> IntelRecord:
    return make_record(
        intel_file,
        RecordSpec(
            record_type="config_dependency",
            record_id=f"{parent.record_id}::dep::{sha256_text(dep)[:12]}",
            title=f"{parent.symbol} dependency",
            summary=f"Kconfig dependency for {parent.symbol}: {dep}",
            body=dep,
            line_start=parent.line_start,
            line_end=parent.line_end,
            symbol=parent.symbol,
            symbol_kind="kconfig_dependency",
            metadata={"dependency": dep},
            confidence_kind="high_confidence_fact",
            parent_record_id=parent.record_id,
        ),
    )


def kconfig_records(intel_file: IntelFile, text: str) -> tuple[list[IntelRecord], list[IntelEdge]]:
    starts = kconfig_block_starts(text)
    offsets = line_offsets(text)
    records: list[IntelRecord] = []
    for idx, start in enumerate(starts[:-1]):
        block = text[start : starts[idx + 1]].strip()
        if not block:
            continue
        spec, deps = kconfig_record_spec(intel_file, block, start, offsets)
        record = make_record(intel_file, spec)
        records.append(record)
        records.extend(kconfig_dependency_record(intel_file, record, dep) for dep in deps)
    return records, []


def split_by_make_blocks(text: str) -> list[tuple[str, str | None, int, int, str]]:
    lines = text.splitlines()
    blocks: list[tuple[str, str | None, int, int, str]] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        define = re.match(r"^\s*define\s+((Package|KernelPackage)/)?([A-Za-z0-9_+./-]+)", line)
        if define:
            start = idx
            idx += 1
            while idx < len(lines) and not re.match(r"^\s*endef\b", lines[idx]):
                idx += 1
            end = min(idx, len(lines) - 1)
            kind = define.group(2) or "define"
            name = define.group(3)
            blocks.append((kind, name, start + 1, end + 1, "\n".join(lines[start : end + 1])))
        build = re.search(r"\$\(eval\s+\$\(call\s+BuildPackage,([A-Za-z0-9_+.-]+)\)\)", line)
        if build:
            blocks.append(("BuildPackage", build.group(1), idx + 1, idx + 1, line))
        idx += 1
    return blocks


def make_records(
    intel_file: IntelFile, text: str, max_chars: int, overlap_lines: int
) -> tuple[list[IntelRecord], list[IntelEdge]]:
    records: list[IntelRecord] = []
    edges: list[IntelEdge] = []
    chunks = split_by_make_blocks(text)
    if not chunks:
        return line_window_records(intel_file, text, max_chars, overlap_lines), edges
    pkg_meta = profile_context.active_profile.make_metadata(intel_file.repo_rel_path, text)
    for ordinal, (kind, name, line_start, line_end, body) in enumerate(chunks, 1):
        metadata = {**common_extracts(body), **pkg_meta, "make_block_kind": kind}
        record_type, symbol, symbol_kind = profile_context.active_profile.make_block_record(
            kind, name, intel_file.repo_rel_path, body
        )
        records.append(
            make_record(
                intel_file,
                RecordSpec(
                    record_type=record_type,
                    record_id=f"{intel_file.source_path}::{kind}::{name or ordinal}::{line_start:06d}",
                    title=f"{kind} {name or ordinal} in {intel_file.source_path}:{line_start}-{line_end}",
                    summary=f"Makefile {kind} block {name or ordinal}",
                    body=body,
                    line_start=line_start,
                    line_end=line_end,
                    symbol=symbol,
                    symbol_kind=symbol_kind,
                    metadata=metadata,
                ),
            )
        )
    records.extend(
        (
            make_record(
                intel_file,
                RecordSpec(
                    record_type="package_install_file",
                    record_id=f"{intel_file.source_path}::install::{sha256_text(path)[:16]}",
                    title=f"install file {path}",
                    summary=f"Package install path {path}",
                    body=path,
                    line_start=None,
                    line_end=None,
                    metadata={"installed_file": path, **pkg_meta},
                    confidence_kind="high_confidence_fact",
                ),
            )
        )
        for path in string_items(pkg_meta.get("installed_files"))
    )
    return records, edges


def dts_records(
    intel_file: IntelFile, text: str, max_chars: int, overlap_lines: int
) -> tuple[list[IntelRecord], list[IntelEdge]]:
    records = line_window_records(intel_file, text, max_chars, overlap_lines)
    edges: list[IntelEdge] = []
    offsets = line_offsets(text)
    for match in re.finditer(r'compatible\s*=\s*((?:"[^"]+"\s*,?\s*)+);', text):
        compat = [item.group(1) for item in re.finditer(r'"([^"]+)"', match.group(1))]
        line = line_for_offset_with_index(offsets, match.start())
        for item in compat:
            records.append(
                make_record(
                    intel_file,
                    RecordSpec(
                        record_type="dts_compatible",
                        record_id=f"{intel_file.source_path}::compatible::{item}::{line:06d}",
                        title=f"DTS compatible {item}",
                        summary=f"DTS compatible string {item}",
                        body=match.group(0),
                        line_start=line,
                        line_end=line,
                        symbol=item,
                        symbol_kind="dts_compatible",
                        metadata={"dts_compatibles": compat},
                        confidence_kind="high_confidence_fact",
                    ),
                )
            )
    for match in re.finditer(r"(?m)^\s*([A-Za-z0-9_,@&/-]+)\s*:\s*([A-Za-z0-9_,@&/-]+)\s*\{", text):
        line = line_for_offset_with_index(offsets, match.start())
        records.append(
            make_record(
                intel_file,
                RecordSpec(
                    record_type="dts_node",
                    record_id=f"{intel_file.source_path}::node::{match.group(1)}::{line:06d}",
                    title=f"DTS node {match.group(1)}",
                    summary=f"DTS node label {match.group(1)} name {match.group(2)}",
                    body=match.group(0),
                    line_start=line,
                    line_end=line,
                    symbol=match.group(1),
                    symbol_kind="dts_node",
                    metadata={"node_name": match.group(2)},
                ),
            )
        )
    return records, edges


def shell_records(
    intel_file: IntelFile, text: str, max_chars: int, overlap_lines: int
) -> tuple[list[IntelRecord], list[IntelEdge]]:
    records: list[IntelRecord] = []
    edges: list[IntelEdge] = []
    matches = list(re.finditer(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{?", text))
    offsets = line_offsets(text)
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[match.start() : end].strip()
        name = match.group(1)
        line_start = line_for_offset_with_index(offsets, match.start())
        line_end = line_start + body.count("\n")
        metadata = {
            **common_extracts(body),
            "symbols_defined": [name],
            "symbols_referenced": extract_referenced_symbols(body),
            "security_sensitive_apis": security_api_refs(body),
        }
        rtype = (
            "service_entrypoint"
            if name
            in {"start", "stop", "reload", "restart", "boot", "service_triggers", "start_service", "stop_service"}
            else "symbol_definition"
        )
        records.append(
            make_record(
                intel_file,
                RecordSpec(
                    record_type=rtype,
                    record_id=f"{intel_file.source_path}::shell_fn::{name}::{line_start:06d}",
                    title=f"shell function {name}",
                    summary=f"Shell function {name} in {intel_file.source_path}",
                    body=body[:5000],
                    line_start=line_start,
                    line_end=line_end,
                    symbol=name,
                    symbol_kind="shell_function",
                    metadata=metadata,
                ),
            )
        )
    for spec in profile_context.active_profile.shell_service_records(
        intel_file.repo_rel_path, intel_file.source_path, text
    ):
        body = spec.get("body", text[:5000])
        metadata: JsonObject = {**common_extracts(body), **spec.get("metadata", {})}
        merged_spec: ProfileRecord = {**spec, "metadata": metadata}
        records.append(make_profile_record(intel_file, merged_spec, default_body=body))
    if not records:
        records.extend(line_window_records(intel_file, text, max_chars, overlap_lines))
    return records, edges


def doc_records(intel_file: IntelFile, text: str, max_chars: int) -> tuple[list[IntelRecord], list[IntelEdge]]:
    records: list[IntelRecord] = []
    starts = list(re.finditer(r"(?m)^\s{0,3}#{1,4}\s+(.+?)\s*$", text))
    if not starts:
        return line_window_records(intel_file, text, max_chars, 4), []
    offsets = line_offsets(text)
    for idx, match in enumerate(starts):
        end = starts[idx + 1].start() if idx + 1 < len(starts) else len(text)
        body = text[match.start() : end].strip()
        if not body:
            continue
        line_start = line_for_offset_with_index(offsets, match.start())
        title_text = match.group(1)
        records.append(
            make_record(
                intel_file,
                RecordSpec(
                    record_type="doc_section",
                    record_id=f"{intel_file.source_path}::doc::{line_start:06d}",
                    title=f"{intel_file.source_path}: {title_text}",
                    summary=first_sentence(body, title_text),
                    body=body[:6000],
                    line_start=line_start,
                    line_end=line_start + body.count("\n"),
                    metadata=common_extracts(body),
                ),
            )
        )
    return records, []


def json_like_records(
    intel_file: IntelFile, text: str, max_chars: int, overlap_lines: int
) -> tuple[list[IntelRecord], list[IntelEdge]]:
    records: list[IntelRecord] = []
    if intel_file.language == "json":
        try:
            payload = cast("object", json.loads(text))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            payload_obj = cast("JsonObject", payload)
            for key, value in payload_obj.items():
                body = json.dumps({key: value}, indent=2, sort_keys=True)[:6000]
                records.append(
                    make_record(
                        intel_file,
                        RecordSpec(
                            record_type="resource_object",
                            record_id=f"{intel_file.source_path}::json::{key}",
                            title=f"JSON resource {key}",
                            summary=f"JSON object key {key}",
                            body=body,
                            line_start=None,
                            line_end=None,
                            symbol=key,
                            symbol_kind="json_key",
                            metadata={"resource_key": key},
                        ),
                    )
                )
    if records:
        return records, []
    return line_window_records(intel_file, text, max_chars, overlap_lines), []


def kconfig_parser(
    intel_file: IntelFile,
    text: str,
    _max_chars: int,
    _overlap_lines: int,
) -> tuple[list[IntelRecord], list[IntelEdge]]:
    return kconfig_records(intel_file, text)


def doc_parser(
    intel_file: IntelFile,
    text: str,
    max_chars: int,
    _overlap_lines: int,
) -> tuple[list[IntelRecord], list[IntelEdge]]:
    return doc_records(intel_file, text, max_chars)
