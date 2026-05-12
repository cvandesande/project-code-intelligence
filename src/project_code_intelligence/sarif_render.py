"""Render normalized SARIF findings as code-intelligence records."""

from __future__ import annotations

from typing import cast

from project_code_intelligence.common import sha256_text
from project_code_intelligence.models import (
    PARSER_VERSION,
    IntelFile,
    IntelRecord,
    JsonObject,
    JsonValue,
    StaticCodeFlowStep,
    StaticFinding,
    StaticLocation,
    StaticRule,
    StaticRun,
)
from project_code_intelligence.records import markdown_fence_for
from project_code_intelligence.sarif_types import (
    SarifFileRecordContext,
    SarifFlowRecordContext,
    SarifRecordRenderContext,
    SarifRuleRecordContext,
)


def rule_for_finding(run: StaticRun, finding: StaticFinding) -> StaticRule | None:
    for rule in run.rules:
        if rule.rule_id == finding.rule_id:
            return rule
    if finding.rule_index is not None and 0 <= finding.rule_index < len(run.rules):
        return run.rules[finding.rule_index]
    return None


def metadata_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in cast("list[object]", value) if isinstance(item, (str, int, float))]
    if isinstance(value, (str, int, float)) and str(value):
        return [str(value)]
    return []


def rule_security_metadata(rule: StaticRule | None) -> JsonObject:
    if rule is None:
        return {}
    properties = rule.properties
    metadata: JsonObject = {}
    for key in ("tags", "precision", "security-severity", "security_severity", "securitySeverity", "problem.severity"):
        value = properties.get(key)
        if value not in (None, "", [], {}):
            metadata[key] = value
    cwe_values = (
        metadata_list(properties.get("cwe"))
        or metadata_list(properties.get("cwe_id"))
        or metadata_list(properties.get("cwes"))
    )
    if not cwe_values:
        cwe_values = [
            tag for tag in metadata_list(properties.get("tags")) if tag.lower().startswith(("cwe-", "external/cwe/"))
        ]
    if cwe_values:
        metadata["cwe"] = cwe_values
    return metadata


def first_present(metadata: JsonObject, keys: tuple[str, ...]) -> JsonValue:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def format_location_summary(source_path: str | None, line: int | None, message: str | None) -> str:
    location = source_path or "(unknown path)"
    if line is not None:
        location = f"{location}:{line}"
    return f"{location}: {message}" if message else location


def summarize_location_steps(items: list[StaticLocation], limit: int = 8) -> list[str]:
    return [
        format_location_summary(location.source_path, location.line_start, location.message)
        for location in items[:limit]
    ]


def summarize_code_flow_steps(steps: list[StaticCodeFlowStep], limit: int = 8) -> list[str]:
    if len(steps) <= limit:
        selected: list[StaticCodeFlowStep | None] = list(steps)
    else:
        selected = [*steps[:3], None, *steps[-3:]]
    return [
        "..." if step is None else format_location_summary(step.source_path, step.line_start, step.message)
        for step in selected
    ]


def fenced_markdown_block(label: str, body: str, info: str = "") -> list[str]:
    fence_marker = markdown_fence_for(body)
    return [f"{label}:", f"{fence_marker}{info}", body, fence_marker]


def sarif_file_record_context(
    run: StaticRun,
    finding: StaticFinding,
    file_by_source_path: dict[str, IntelFile],
) -> SarifFileRecordContext:
    source_path = finding.primary_source_path or run.sarif_path
    intel_file = file_by_source_path.get(source_path)
    language = intel_file.language if intel_file else "sarif"
    file_role = intel_file.file_role if intel_file else "static-analysis"
    content_class = intel_file.content_class if intel_file else "analysis"
    location_text = f"{source_path}:{finding.line_start or 1}" if source_path else run.sarif_path
    return SarifFileRecordContext(
        source_path=source_path,
        language=language,
        file_role=file_role,
        content_class=content_class,
        location_text=location_text,
    )


def sarif_rule_record_context(run: StaticRun, finding: StaticFinding) -> SarifRuleRecordContext:
    rule = rule_for_finding(run, finding)
    rule_metadata = rule_security_metadata(rule)
    security_severity = first_present(
        rule_metadata,
        ("security-severity", "security_severity", "securitySeverity"),
    )
    tags = metadata_list(rule_metadata.get("tags"))
    cwe_values = metadata_list(rule_metadata.get("cwe"))
    return SarifRuleRecordContext(
        rule=rule,
        rule_metadata=rule_metadata,
        severity=finding.properties.get("severity") or finding.level,
        security_severity=security_severity,
        tags=tags,
        cwe_values=cwe_values,
    )


def code_flow_endpoint(steps: list[StaticCodeFlowStep], *, source: bool) -> str | None:
    if not steps:
        return None
    step = steps[0] if source else steps[-1]
    return format_location_summary(step.source_path, step.line_start, step.message)


def sarif_path_mappings(finding: StaticFinding) -> list[str]:
    location_mappings = {
        mapping
        for location in finding.locations
        for mapping in [location.properties.get("path_mapping")]
        if isinstance(mapping, str)
    }
    flow_mappings = {
        mapping
        for step in finding.code_flows
        for mapping in [step.properties.get("path_mapping")]
        if isinstance(mapping, str)
    }
    return sorted(location_mappings | flow_mappings)


def sarif_primary_path_mapping(finding: StaticFinding) -> object:
    if finding.locations and isinstance(finding.locations[0].properties.get("path_mapping"), str):
        return finding.locations[0].properties.get("path_mapping")
    return None


def sarif_flow_record_context(finding: StaticFinding) -> SarifFlowRecordContext:
    code_flow_summary = summarize_code_flow_steps(finding.code_flows)
    return SarifFlowRecordContext(
        code_flow_summary=code_flow_summary,
        code_flow_source=code_flow_endpoint(finding.code_flows, source=True),
        code_flow_sink=code_flow_endpoint(finding.code_flows, source=False),
        location_summary=summarize_location_steps(finding.locations),
        path_mappings=sarif_path_mappings(finding),
        primary_path_mapping=sarif_primary_path_mapping(finding),
        suppressed=bool(finding.suppressions),
    )


def sarif_record_metadata(
    run: StaticRun,
    finding: StaticFinding,
    rule_context: SarifRuleRecordContext,
    flow_context: SarifFlowRecordContext,
) -> JsonObject:
    rule = rule_context.rule
    metadata = {
        "static_analysis_tool": run.tool_name,
        "static_analysis_tool_version": run.tool_version,
        "sarif_path": run.sarif_path,
        "sarif_sha256": run.sarif_sha256,
        "finding_key": finding.finding_key,
        "rule_id": finding.rule_id,
        "rule_name": rule.name if rule else None,
        "rule_short_description": rule.short_description if rule else None,
        "rule_full_description": rule.full_description if rule else None,
        "rule_help_uri": rule.help_uri if rule else None,
        "rule_default_level": rule.default_level if rule else None,
        "rule_tags": rule_context.tags,
        "rule_precision": rule_context.rule_metadata.get("precision"),
        "rule_security_severity": rule_context.security_severity,
        "rule_cwe": rule_context.cwe_values,
        "rule_properties": rule.properties if rule else {},
        "level": finding.level,
        "kind": finding.kind,
        "baseline_state": finding.baseline_state,
        "primary_uri": finding.primary_uri,
        "primary_path_mapping": flow_context.primary_path_mapping,
        "path_mappings": flow_context.path_mappings,
        "suppressed": flow_context.suppressed,
        "fingerprints": finding.fingerprints,
        "suppressions_count": len(finding.suppressions),
        "locations_count": len(finding.locations),
        "code_flow_steps": len(finding.code_flows),
        "code_flow_source": flow_context.code_flow_source,
        "code_flow_sink": flow_context.code_flow_sink,
        "code_flow_summary": flow_context.code_flow_summary,
        "location_summary": flow_context.location_summary,
    }
    return cast("JsonObject", {key: value for key, value in metadata.items() if value not in (None, {}, [])})


def append_sarif_rule_body_lines(body_lines: list[str], render: SarifRecordRenderContext) -> None:
    rule = render.rule.rule
    if rule and rule.name:
        body_lines.append(f"Rule name: {rule.name}")
    if rule and rule.help_uri:
        body_lines.append(f"Help URI: {rule.help_uri}")
    if render.rule.rule_metadata.get("precision"):
        body_lines.append(f"Precision: {render.rule.rule_metadata['precision']}")
    if render.rule.security_severity:
        body_lines.append(f"Security severity: {render.rule.security_severity}")
    if render.rule.cwe_values:
        body_lines.append(f"CWE: {', '.join(render.rule.cwe_values[:12])}")
    if render.rule.tags:
        body_lines.append(f"Tags: {', '.join(render.rule.tags[:20])}")


def append_sarif_detail_blocks(body_lines: list[str], render: SarifRecordRenderContext) -> None:
    rule = render.rule.rule
    body_lines.append("")
    body_lines.extend(fenced_markdown_block("Message", render.finding.message))
    if rule and rule.short_description:
        body_lines.extend(["", *fenced_markdown_block("Rule short description", rule.short_description)])
    if rule and rule.full_description:
        body_lines.extend(["", *fenced_markdown_block("Rule full description", rule.full_description)])
    if render.finding.locations and render.finding.locations[0].snippet:
        body_lines.extend(["", *fenced_markdown_block("Snippet", render.finding.locations[0].snippet)])
    if render.flow.location_summary:
        body_lines.extend(["", *fenced_markdown_block("Locations", "\n".join(render.flow.location_summary))])
    if render.flow.code_flow_summary:
        body_lines.extend(["", *fenced_markdown_block("Code flow", "\n".join(render.flow.code_flow_summary))])


def sarif_record_body_lines(render: SarifRecordRenderContext) -> list[str]:
    body_lines = [
        f"Tool: {render.run.tool_name}",
        f"Rule: {render.finding.rule_id}",
        f"Severity: {render.rule.severity or 'unknown'}",
        f"Location: {render.file.location_text}",
    ]
    append_sarif_rule_body_lines(body_lines, render)
    append_sarif_detail_blocks(body_lines, render)
    return body_lines


def sarif_embedding_text(render: SarifRecordRenderContext) -> str:
    rule = render.rule.rule
    return "\n".join([
        render.title,
        render.summary,
        f"tool: {render.run.tool_name}",
        f"rule_id: {render.finding.rule_id}",
        f"rule_name: {rule.name if rule and rule.name else ''}",
        f"rule_short_description: {rule.short_description if rule and rule.short_description else ''}",
        f"rule_full_description: {rule.full_description if rule and rule.full_description else ''}",
        f"rule_help_uri: {rule.help_uri if rule and rule.help_uri else ''}",
        f"rule_tags: {', '.join(render.rule.tags[:40])}",
        f"rule_precision: {render.rule.rule_metadata.get('precision') or ''}",
        f"rule_security_severity: {render.rule.security_severity or ''}",
        f"rule_cwe: {', '.join(render.rule.cwe_values[:20])}",
        f"severity: {render.rule.severity or ''}",
        f"security_severity: {render.rule.security_severity or ''}",
        f"suppressed: {render.flow.suppressed}",
        f"path_mapping: {render.flow.primary_path_mapping or ''}",
        f"path_mappings: {', '.join(render.flow.path_mappings)}",
        f"source_path: {render.file.source_path}",
        f"message: {render.finding.message}",
        "locations:\n" + "\n".join(render.flow.location_summary),
        f"code_flow_source: {render.flow.code_flow_source or ''}",
        f"code_flow_sink: {render.flow.code_flow_sink or ''}",
        "code_flow:\n" + "\n".join(render.flow.code_flow_summary),
        f"code_flow_steps: {len(render.finding.code_flows)}",
    ])


def sarif_display_content(render: SarifRecordRenderContext, body_lines: list[str]) -> str:
    display_lines = [
        f"# {render.title}",
        "",
        f"- Repo: `{render.repo}`",
        f"- Source: `{render.file.source_path}`",
        f"- Tool: `{render.run.tool_name}`",
        f"- Rule: `{render.finding.rule_id}`",
        "- Confidence: `tool_finding`",
    ]
    if render.flow.primary_path_mapping:
        display_lines.append(f"- Path mapping: `{render.flow.primary_path_mapping}`")
    if render.flow.suppressed:
        display_lines.append("- Suppressed: `true`")
    return "\n".join(display_lines) + "\n\n" + "\n".join(body_lines)


def sarif_record_for_finding(
    collection: str,
    repo: str,
    run: StaticRun,
    finding: StaticFinding,
    file_by_source_path: dict[str, IntelFile],
) -> IntelRecord:
    file_context = sarif_file_record_context(run, finding, file_by_source_path)
    rule_context = sarif_rule_record_context(run, finding)
    flow_context = sarif_flow_record_context(finding)
    title = f"{run.tool_name} {finding.rule_id} at {file_context.location_text}"
    summary = f"{run.tool_name} static-analysis finding {finding.rule_id}: {finding.message[:180]}"
    render_context = SarifRecordRenderContext(
        title=title,
        summary=summary,
        repo=repo,
        run=run,
        finding=finding,
        file=file_context,
        rule=rule_context,
        flow=flow_context,
    )
    body_lines = sarif_record_body_lines(render_context)
    embedding_text = sarif_embedding_text(render_context)
    record_hash = sha256_text(
        "\n".join([run.tool_name, run.sarif_path, finding.finding_key, finding.rule_id, file_context.source_path])
    )[:24]
    return IntelRecord(
        collection=collection,
        source_path=file_context.source_path,
        language=file_context.language,
        file_role=file_context.file_role,
        content_class=file_context.content_class,
        record_type="static_finding",
        record_id=f"{file_context.source_path}::static_finding::{record_hash}",
        title=title,
        summary=summary,
        embedding_text=embedding_text,
        display_content=sarif_display_content(render_context, body_lines),
        line_start=finding.line_start,
        line_end=finding.line_end,
        symbol=finding.rule_id,
        symbol_kind="static_analysis_rule",
        confidence_kind="tool_finding",
        tool=run.tool_name,
        rule_id=finding.rule_id,
        severity=str(rule_context.severity) if rule_context.severity else None,
        analyzer=run.tool_name,
        analyzer_version=run.tool_version,
        parser="sarif",
        parser_version=PARSER_VERSION,
        metadata=sarif_record_metadata(run, finding, rule_context, flow_context),
    )
