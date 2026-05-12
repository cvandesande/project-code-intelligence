"""Patch/diff record extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from project_code_intelligence.records import (
    RecordSpec,
    common_extracts,
    line_for_offset_with_index,
    line_offsets,
    line_window_records,
    make_record,
)

if TYPE_CHECKING:
    from project_code_intelligence.models import IntelEdge, IntelFile, IntelRecord


@dataclass(frozen=True)
class PatchSectionSpec:
    old_path: str
    new_path: str
    body: str
    line_start: int
    line_end: int


@dataclass(frozen=True)
class PatchHunkSpec:
    new_path: str
    body: str
    line_start: int
    hunk_index: int
    parent_record_id: str


def patch_touched_record(intel_file: IntelFile, spec: PatchSectionSpec) -> IntelRecord:
    return make_record(
        intel_file,
        RecordSpec(
            record_type="patch_touched_file",
            record_id=f"{intel_file.source_path}::patch_file::{spec.new_path}",
            title=f"patch touches {spec.new_path}",
            summary=f"Patch {intel_file.source_path} touches {spec.new_path}",
            body=spec.body[:4000],
            line_start=spec.line_start,
            line_end=spec.line_end,
            metadata={"old_path": spec.old_path, "new_path": spec.new_path},
        ),
    )


def patch_hunk_record(intel_file: IntelFile, spec: PatchHunkSpec) -> IntelRecord:
    added = re.findall(
        r"^\+\s*(?:static\s+)?[A-Za-z_][\w\s\*]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        spec.body,
        re.MULTILINE,
    )
    removed = re.findall(
        r"^-\s*(?:static\s+)?[A-Za-z_][\w\s\*]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        spec.body,
        re.MULTILINE,
    )
    metadata = {
        **common_extracts(spec.body),
        "patch_target_path": spec.new_path,
        "patch_added_symbols": sorted(set(added)),
        "patch_removed_symbols": sorted(set(removed)),
    }
    return make_record(
        intel_file,
        RecordSpec(
            record_type="patch_hunk",
            record_id=f"{intel_file.source_path}::hunk::{spec.new_path}::{spec.hunk_index + 1:04d}",
            title=f"patch hunk {spec.new_path} #{spec.hunk_index + 1}",
            summary=f"Patch hunk for {spec.new_path}",
            body=spec.body[:5000],
            line_start=spec.line_start,
            line_end=spec.line_start + spec.body.count("\n"),
            metadata=metadata,
            parent_record_id=spec.parent_record_id,
        ),
    )


def patch_section_records(intel_file: IntelFile, spec: PatchSectionSpec) -> list[IntelRecord]:
    touched_record = patch_touched_record(intel_file, spec)
    records = [touched_record]
    hunks = list(re.finditer(r"(?m)^@@ .*?@@.*$", spec.body))
    for hunk_index, hunk in enumerate(hunks):
        hunk_end = hunks[hunk_index + 1].start() if hunk_index + 1 < len(hunks) else len(spec.body)
        hunk_body = spec.body[hunk.start() : hunk_end]
        hunk_line = spec.line_start + spec.body.count("\n", 0, hunk.start())
        records.append(
            patch_hunk_record(
                intel_file,
                PatchHunkSpec(
                    new_path=spec.new_path,
                    body=hunk_body,
                    line_start=hunk_line,
                    hunk_index=hunk_index,
                    parent_record_id=touched_record.record_id,
                ),
            )
        )
    return records


def patch_records(intel_file: IntelFile, text: str) -> tuple[list[IntelRecord], list[IntelEdge]]:
    records: list[IntelRecord] = []
    edges: list[IntelEdge] = []
    sections = list(re.finditer(r"(?m)^diff --git a/(.*?) b/(.*?)$", text))
    if not sections:
        return line_window_records(intel_file, text, 3000, 4), edges
    offsets = line_offsets(text)
    for idx, section in enumerate(sections):
        end = sections[idx + 1].start() if idx + 1 < len(sections) else len(text)
        body = text[section.start() : end]
        old_path, new_path = section.group(1), section.group(2)
        line_start = line_for_offset_with_index(offsets, section.start())
        line_end = line_for_offset_with_index(offsets, end)
        records.extend(
            patch_section_records(
                intel_file,
                PatchSectionSpec(
                    old_path=old_path,
                    new_path=new_path,
                    body=body,
                    line_start=line_start,
                    line_end=line_end,
                ),
            )
        )
    return records, edges
