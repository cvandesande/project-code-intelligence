"""SARIF discovery, normalization, and record conversion."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from project_code_intelligence import profile_context
from project_code_intelligence.common import sha256_bytes
from project_code_intelligence.exceptions import SarifFileTooLargeError, SarifLoadError
from project_code_intelligence.models import SarifIngest, StaticRun
from project_code_intelligence.sarif.discovery import discover_sarif_files, explicit_sarif_patterns, repo_for_sarif_file
from project_code_intelligence.sarif.parse import (
    json_array,
    parse_sarif_result,
    sarif_automation_id,
    sarif_original_uri_base_ids,
    sarif_rule_items,
    sarif_tool_metadata,
)
from project_code_intelligence.sarif.paths import (
    SarifPathContext,
    relative_to_or_none,
    resolve_sarif_source_path,
    source_path_from_sarif_uri,
)
from project_code_intelligence.sarif.render import sarif_record_for_finding
from project_code_intelligence.sarif.types import (
    LoadedSarifFile,
    SarifIngestContext,
    SarifIngestState,
    SarifResultContext,
    SarifRunContext,
)

if TYPE_CHECKING:
    from pathlib import Path

    from project_code_intelligence.models import JsonObject, StaticFinding

__all__ = [
    "SarifIngestContext",
    "SarifPathContext",
    "discover_sarif_files",
    "explicit_sarif_patterns",
    "ingest_sarif",
    "relative_to_or_none",
    "resolve_sarif_source_path",
    "source_path_from_sarif_uri",
]


def sarif_file_bytes(context: SarifIngestContext, sarif_path: Path) -> bytes:
    size = sarif_path.stat().st_size
    if size > context.max_bytes:
        raise SarifFileTooLargeError(path=str(sarif_path), size_bytes=size, limit_bytes=context.max_bytes)
    return sarif_path.read_bytes()


def load_sarif_file(context: SarifIngestContext, sarif_path: Path) -> LoadedSarifFile:
    try:
        data = sarif_file_bytes(context, sarif_path)
        sarif_value = cast("object", json.loads(data.decode("utf-8")))
    except SarifFileTooLargeError as exc:
        context_error = {"source_path": str(sarif_path), "parser": "sarif", "error": "sarif_file_too_large"}
        raise SarifLoadError(context=context_error) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        context_error = {"source_path": str(sarif_path), "parser": "sarif", "error": str(exc)}
        raise SarifLoadError(context=context_error) from exc
    if not isinstance(sarif_value, dict):
        context_error = {"source_path": str(sarif_path), "parser": "sarif", "error": "SARIF root is not an object"}
        raise TypeError(json.dumps(context_error, sort_keys=True))
    return LoadedSarifFile(
        sarif_path=sarif_path,
        source_path=relative_to_or_none(sarif_path, context.root) or str(sarif_path),
        sarif_hash=sha256_bytes(data),
        default_repo=repo_for_sarif_file(context.root, context.repos, sarif_path),
        sarif=cast("JsonObject", sarif_value),
    )


def append_sarif_load_failure(failures: list[JsonObject], exc: RuntimeError, sarif_path: Path) -> None:
    try:
        failure = cast("JsonObject", json.loads(str(exc)))
    except json.JSONDecodeError:
        failure = cast("JsonObject", {"source_path": str(sarif_path), "parser": "sarif", "error": str(exc)})
    failures.append(failure)


def sarif_static_run(run_context: SarifRunContext, repo: str) -> StaticRun:
    uri_base_ids = run_context.result_context.path_context.uri_base_ids
    return StaticRun(
        repo=repo,
        sarif_path=run_context.loaded.source_path,
        sarif_sha256=run_context.loaded.sarif_hash,
        run_index=run_context.run_index,
        tool_name=run_context.tool_meta["tool_name"],
        tool_version=run_context.tool_meta["tool_version"],
        semantic_version=run_context.tool_meta["semantic_version"],
        information_uri=run_context.tool_meta["information_uri"],
        automation_id=sarif_automation_id(run_context.run),
        metadata={
            "sarif_version": run_context.loaded.sarif.get("version"),
            "schema": run_context.loaded.sarif.get("$schema"),
            "originalUriBaseIds": uri_base_ids,
            "versionControlProvenance": json_array(run_context.run.get("versionControlProvenance"))[:3],
            "invocations": json_array(run_context.run.get("invocations"))[:3],
        },
        rules=run_context.rules,
    )


def ensure_sarif_static_run(state: SarifIngestState, run_context: SarifRunContext, repo: str) -> StaticRun:
    key = (run_context.loaded.source_path, run_context.run_index, repo)
    static_run = state.runs_by_key.get(key)
    if static_run is None:
        static_run = sarif_static_run(run_context, repo)
        state.runs_by_key[key] = static_run
    return static_run


def annotate_static_source_origin(
    context: SarifIngestContext,
    state: SarifIngestState,
    finding: StaticFinding,
) -> None:
    metadata = profile_context.active_profile.static_source_origin_metadata(
        finding.primary_source_path,
        context.repos,
        state.known_source_paths,
    )
    if metadata:
        finding.properties = {**finding.properties, **metadata}


def ingest_sarif_result(
    context: SarifIngestContext,
    state: SarifIngestState,
    run_context: SarifRunContext,
    result: JsonObject,
    result_index: int,
) -> None:
    finding, result_repo = parse_sarif_result(run_context.result_context, result, result_index)
    repo = result_repo or run_context.loaded.default_repo
    if repo not in context.repos:
        state.failures.append({
            "source_path": finding.primary_source_path or run_context.loaded.source_path,
            "parser": "sarif",
            "error": f"SARIF finding did not map to an indexed repo: {repo!r}",
            "rule_id": finding.rule_id,
        })
        return
    annotate_static_source_origin(context, state, finding)
    static_run = ensure_sarif_static_run(state, run_context, repo)
    static_run.findings.append(finding)
    state.records_by_repo.setdefault(repo, []).append(
        sarif_record_for_finding(
            context.collection,
            repo,
            static_run,
            finding,
            context.file_by_source_path,
        )
    )


def sarif_run_context(
    context: SarifIngestContext,
    state: SarifIngestState,
    loaded: LoadedSarifFile,
    run_index: int,
    run: JsonObject,
) -> SarifRunContext:
    rules = sarif_rule_items(run)
    path_context = SarifPathContext(
        root=context.root,
        repos=context.repos,
        default_repo=loaded.default_repo,
        uri_base_ids=sarif_original_uri_base_ids(run),
        known_source_paths=state.known_source_paths,
    )
    result_context = SarifResultContext(path_context=path_context, rules=rules, run_index=run_index)
    return SarifRunContext(
        loaded=loaded,
        run_index=run_index,
        run=run,
        tool_meta=sarif_tool_metadata(run),
        rules=rules,
        result_context=result_context,
    )


def ingest_sarif_run(
    context: SarifIngestContext,
    state: SarifIngestState,
    loaded: LoadedSarifFile,
    run_index: int,
    run: JsonObject,
) -> None:
    run_context = sarif_run_context(context, state, loaded, run_index, run)
    for result_index, result in enumerate(json_array(run.get("results"))):
        if isinstance(result, dict):
            ingest_sarif_result(context, state, run_context, result, result_index)


def ingest_sarif(context: SarifIngestContext, sarif_files: list[Path]) -> SarifIngest:
    state = SarifIngestState(
        runs_by_key={},
        records_by_repo={repo: [] for repo in context.repos},
        failures=[],
        known_source_paths=set(context.file_by_source_path),
    )
    for sarif_path in sarif_files:
        try:
            loaded = load_sarif_file(context, sarif_path)
        except RuntimeError as exc:
            append_sarif_load_failure(state.failures, exc, sarif_path)
            continue
        for run_index, run in enumerate(json_array(loaded.sarif.get("runs"))):
            if isinstance(run, dict):
                ingest_sarif_run(context, state, loaded, run_index, run)
    return SarifIngest(
        runs=list(state.runs_by_key.values()),
        records_by_repo=state.records_by_repo,
        failures=state.failures,
    )
