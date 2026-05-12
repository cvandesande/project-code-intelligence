"""Built-in security pattern parser."""

from __future__ import annotations

import re

from project_code_intelligence import profile_context
from project_code_intelligence.models import CHUNKER_VERSION, IntelFile, IntelRecord, JsonObject
from project_code_intelligence.records import RecordSpec, make_record


def security_api_refs(text: str) -> list[str]:
    refs: list[str] = []
    for pattern, rule_id, _severity, _confidence, _summary in profile_context.active_profile.security_patterns():
        if re.search(pattern, text):
            refs.append(rule_id)
    return sorted(set(refs))


def security_context(intel_file: IntelFile) -> JsonObject:
    return profile_context.active_profile.security_context(
        intel_file.repo_rel_path,
        intel_file.language,
        intel_file.file_role,
        intel_file.content_class,
    )


def security_pattern_anchor(pattern: str) -> str | None:
    match = re.search(r"\\b([A-Za-z_][A-Za-z0-9_]*)", pattern)
    return match.group(1) if match else None


def security_records(intel_file: IntelFile, text: str) -> list[IntelRecord]:
    if not profile_context.active_profile.should_scan_security(
        intel_file.repo_rel_path,
        intel_file.language,
        intel_file.file_role,
    ):
        return []
    records: list[IntelRecord] = []
    lines = text.splitlines()
    context = security_context(intel_file)
    seen: set[tuple[int, str]] = set()
    for lineno, line in enumerate(lines, 1):
        window = "\n".join(lines[lineno - 1 : min(len(lines), lineno + 2)])
        for pattern, rule_id, severity, confidence_kind, summary in profile_context.active_profile.security_patterns():
            matched = re.search(pattern, line)
            if not matched:
                anchor = security_pattern_anchor(pattern)
                matched = bool(anchor and re.search(rf"\b{re.escape(anchor)}\b", line) and re.search(pattern, window))
            if matched:
                key = (lineno, rule_id)
                if key in seen:
                    continue
                seen.add(key)
                body = window.strip()
                records.append(
                    make_record(
                        intel_file,
                        RecordSpec(
                            record_type="security_pattern",
                            record_id=f"{intel_file.source_path}::security::{rule_id}::{lineno:06d}",
                            title=f"{rule_id} at {intel_file.source_path}:{lineno}",
                            summary=summary,
                            body=body,
                            line_start=lineno,
                            line_end=lineno + body.count("\n"),
                            metadata={"evidence": body, "pattern": pattern, **context},
                            confidence_kind=confidence_kind,
                            confidence=0.45,
                            tool="builtin-pattern-scan",
                            rule_id=rule_id,
                            severity=severity,
                            analyzer="project-code-intelligence-builtin-security-patterns",
                            analyzer_version=CHUNKER_VERSION,
                        ),
                    )
                )
    return records
