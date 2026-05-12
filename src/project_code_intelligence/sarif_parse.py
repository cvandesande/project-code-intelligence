"""SARIF JSON normalization."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from project_code_intelligence.common import sha256_text
from project_code_intelligence.models import (
    JsonObject,
    JsonValue,
    StaticCodeFlowStep,
    StaticFinding,
    StaticLocation,
    StaticRule,
)
from project_code_intelligence.sarif_paths import SarifPathContext, resolve_sarif_source_path

if TYPE_CHECKING:
    from project_code_intelligence.sarif_types import SarifResultContext, SarifToolMetadata


def sarif_message_text(message: object) -> str:
    if not isinstance(message, dict):
        return ""
    message_obj = cast("dict[str, object]", message)
    value = message_obj.get("text") or message_obj.get("markdown") or ""
    return str(value)


def json_object(value: object) -> JsonObject:
    return cast("JsonObject", value) if isinstance(value, dict) else {}


def json_array(value: object) -> list[JsonValue]:
    return cast("list[JsonValue]", value) if isinstance(value, list) else []


def sarif_description_text(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    value_obj = cast("dict[str, object]", value)
    text = value_obj.get("text") or value_obj.get("markdown")
    return str(text) if text else None


def sarif_region(location: JsonObject) -> JsonObject:
    physical = location.get("physicalLocation")
    if not isinstance(physical, dict):
        return {}
    region = physical.get("region")
    return json_object(region)


def sarif_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def sarif_artifact_location(location: JsonObject) -> JsonObject:
    physical = location.get("physicalLocation")
    if not isinstance(physical, dict):
        return {}
    artifact = physical.get("artifactLocation")
    if not isinstance(artifact, dict):
        return {}
    return json_object(artifact)


def sarif_artifact_uri(location: JsonObject) -> str | None:
    artifact = sarif_artifact_location(location)
    uri = artifact.get("uri")
    return str(uri) if uri else None


def sarif_artifact_uri_base_id(location: JsonObject) -> str | None:
    artifact = sarif_artifact_location(location)
    uri_base_id = artifact.get("uriBaseId")
    return str(uri_base_id) if uri_base_id else None


def sarif_original_uri_base_ids(run: JsonObject) -> dict[str, str]:
    raw = run.get("originalUriBaseIds")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        uri = value.get("uri")
        if uri:
            out[str(key)] = str(uri)
    return out


def sarif_location(
    context: SarifPathContext,
    location: JsonObject,
    *,
    ordinal: int,
    location_kind: str,
) -> tuple[StaticLocation, str | None]:
    uri = sarif_artifact_uri(location)
    uri_base_id = sarif_artifact_uri_base_id(location)
    resolution = resolve_sarif_source_path(context, uri, uri_base_id=uri_base_id)
    region = sarif_region(location)
    snippet = json_object(region.get("snippet"))
    properties = dict(json_object(location.get("properties")))
    _ = properties.setdefault("path_mapping", resolution.path_mapping)
    if uri_base_id:
        _ = properties.setdefault("uriBaseId", uri_base_id)
        if uri_base_id in context.uri_base_ids:
            _ = properties.setdefault("resolvedUriBase", context.uri_base_ids[uri_base_id])
    snippet_text = snippet.get("text")
    return (
        StaticLocation(
            ordinal=ordinal,
            location_kind=location_kind,
            source_path=resolution.source_path,
            uri=uri,
            message=sarif_message_text(location.get("message")),
            line_start=sarif_int(region.get("startLine")),
            line_end=sarif_int(region.get("endLine")) or sarif_int(region.get("startLine")),
            column_start=sarif_int(region.get("startColumn")),
            column_end=sarif_int(region.get("endColumn")),
            snippet=str(snippet_text) if snippet_text else None,
            properties=properties,
        ),
        resolution.repo,
    )


def sarif_rule_items(run: JsonObject) -> list[StaticRule]:
    tool = json_object(run.get("tool"))
    driver = json_object(tool.get("driver"))
    rules = json_array(driver.get("rules"))
    out: list[StaticRule] = []
    for item in rules:
        if not isinstance(item, dict):
            continue
        default_config = json_object(item.get("defaultConfiguration"))
        properties = json_object(item.get("properties"))
        rule_id = str(item.get("id") or item.get("name") or "unknown")
        out.append(
            StaticRule(
                rule_id=rule_id,
                name=str(item["name"]) if item.get("name") else None,
                short_description=sarif_description_text(item.get("shortDescription")),
                full_description=sarif_description_text(item.get("fullDescription")),
                default_level=str(default_config["level"]) if default_config.get("level") else None,
                help_uri=str(item["helpUri"]) if item.get("helpUri") else None,
                properties=properties,
                metadata={key: item[key] for key in ("help", "relationships") if key in item},
            )
        )
    return out


def sarif_tool_metadata(run: JsonObject) -> SarifToolMetadata:
    tool = json_object(run.get("tool"))
    driver = json_object(tool.get("driver"))
    return {
        "tool_name": str(driver.get("name") or "unknown"),
        "tool_version": str(driver["version"]) if driver.get("version") else None,
        "semantic_version": str(driver["semanticVersion"]) if driver.get("semanticVersion") else None,
        "information_uri": str(driver["informationUri"]) if driver.get("informationUri") else None,
    }


def sarif_automation_id(run: JsonObject) -> str | None:
    automation = run.get("automationDetails")
    if isinstance(automation, dict) and automation.get("id"):
        return str(automation["id"])
    return None


def sarif_finding_key(result: JsonObject, run_index: int, result_index: int) -> str:
    fingerprints = result.get("partialFingerprints") or result.get("fingerprints")
    if isinstance(fingerprints, dict) and fingerprints:
        key, value = min(fingerprints.items())
        return sha256_text(f"{key}:{value}")[:32]
    stable = json.dumps(
        {
            "run": run_index,
            "idx": result_index,
            "rule": result.get("ruleId"),
            "message": sarif_message_text(result.get("message")),
            "locations": result.get("locations"),
        },
        sort_keys=True,
        default=str,
    )
    return sha256_text(stable)[:32]


def parse_sarif_locations(context: SarifPathContext, result: JsonObject) -> tuple[list[StaticLocation], str | None]:
    locations_raw = json_array(result.get("locations"))
    locations: list[StaticLocation] = []
    primary_repo = context.default_repo
    for idx, item in enumerate(locations_raw):
        if not isinstance(item, dict):
            continue
        location, repo = sarif_location(
            context,
            item,
            ordinal=idx,
            location_kind="primary" if idx == 0 else "additional",
        )
        locations.append(location)
        if idx == 0 and repo:
            primary_repo = repo
    related_raw = json_array(result.get("relatedLocations"))
    for idx, item in enumerate(related_raw, start=len(locations)):
        if not isinstance(item, dict):
            continue
        location, _repo = sarif_location(
            SarifPathContext(
                root=context.root,
                repos=context.repos,
                default_repo=primary_repo,
                uri_base_ids=context.uri_base_ids,
                known_source_paths=context.known_source_paths,
            ),
            item,
            ordinal=idx,
            location_kind="related",
        )
        locations.append(location)
    return locations, primary_repo


def parse_sarif_code_flows(context: SarifPathContext, result: JsonObject) -> list[StaticCodeFlowStep]:
    code_flow_steps: list[StaticCodeFlowStep] = []
    code_flows = json_array(result.get("codeFlows"))
    for flow_idx, flow in enumerate(code_flows):
        if not isinstance(flow, dict):
            continue
        thread_flows = json_array(flow.get("threadFlows"))
        for thread_idx, thread in enumerate(thread_flows):
            if not isinstance(thread, dict):
                continue
            flow_locations = json_array(thread.get("locations"))
            for step_idx, flow_location in enumerate(flow_locations):
                if not isinstance(flow_location, dict):
                    continue
                location = flow_location.get("location")
                if not isinstance(location, dict):
                    continue
                static_location, _repo = sarif_location(
                    context,
                    location,
                    ordinal=step_idx,
                    location_kind="code_flow",
                )
                step_properties = dict(json_object(flow_location.get("properties")))
                for key in ("uriBaseId", "resolvedUriBase"):
                    if key in static_location.properties:
                        _ = step_properties.setdefault(key, static_location.properties[key])
                code_flow_steps.append(
                    StaticCodeFlowStep(
                        flow_index=flow_idx,
                        thread_index=thread_idx,
                        step_index=step_idx,
                        source_path=static_location.source_path,
                        uri=static_location.uri,
                        message=static_location.message,
                        line_start=static_location.line_start,
                        line_end=static_location.line_end,
                        column_start=static_location.column_start,
                        column_end=static_location.column_end,
                        importance=str(flow_location["importance"]) if flow_location.get("importance") else None,
                        properties=step_properties,
                    )
                )
    return code_flow_steps


def sarif_rule_id(result: JsonObject, rules: list[StaticRule]) -> tuple[str, int | None]:
    rule_index_value = result.get("ruleIndex")
    rule_index = (
        rule_index_value if isinstance(rule_index_value, int) and not isinstance(rule_index_value, bool) else None
    )
    rule_id = str(result.get("ruleId") or "")
    if not rule_id and rule_index is not None and 0 <= rule_index < len(rules):
        rule_id = rules[rule_index].rule_id
    if not rule_id:
        rule_id = f"rule_index_{rule_index if rule_index is not None else 'unknown'}"
    return rule_id, rule_index


def parse_sarif_result(
    context: SarifResultContext,
    result: JsonObject,
    result_index: int,
) -> tuple[StaticFinding, str | None]:
    locations, primary_repo = parse_sarif_locations(context.path_context, result)
    code_flow_steps = parse_sarif_code_flows(
        SarifPathContext(
            root=context.path_context.root,
            repos=context.path_context.repos,
            default_repo=primary_repo,
            uri_base_ids=context.path_context.uri_base_ids,
            known_source_paths=context.path_context.known_source_paths,
        ),
        result,
    )
    primary = locations[0] if locations else None
    fingerprints = result.get("partialFingerprints") or result.get("fingerprints") or {}
    properties = json_object(result.get("properties"))
    rule_id, rule_index = sarif_rule_id(result, context.rules)
    finding = StaticFinding(
        finding_key=sarif_finding_key(result, context.run_index, result_index),
        rule_id=rule_id,
        rule_index=rule_index,
        level=str(result["level"]) if result.get("level") else None,
        kind=str(result["kind"]) if result.get("kind") else None,
        message=sarif_message_text(result.get("message")) or "(no SARIF message)",
        baseline_state=str(result["baselineState"]) if result.get("baselineState") else None,
        primary_source_path=primary.source_path if primary else None,
        primary_uri=primary.uri if primary else None,
        line_start=primary.line_start if primary else None,
        line_end=primary.line_end if primary else None,
        column_start=primary.column_start if primary else None,
        column_end=primary.column_end if primary else None,
        fingerprints=fingerprints if isinstance(fingerprints, dict) else {},
        suppressions=json_array(result.get("suppressions")),
        properties=properties,
        raw_result={key: result[key] for key in ("rank", "taxa", "fixes") if key in result},
        locations=locations,
        code_flows=code_flow_steps,
    )
    return finding, primary_repo
