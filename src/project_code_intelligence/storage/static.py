"""Persistence for normalized static-analysis findings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from project_code_intelligence import db
from project_code_intelligence.storage.schema import row_int

if TYPE_CHECKING:
    from project_code_intelligence.models import (
        Snapshot,
        StaticCodeFlowStep,
        StaticFinding,
        StaticLocation,
        StaticRule,
        StaticRun,
    )


def _empty_static_counts() -> dict[str, int]:
    return {
        "static_runs": 0,
        "static_rules": 0,
        "static_findings": 0,
        "static_locations": 0,
        "static_code_flow_steps": 0,
    }


def _insert_static_run(conn: db.DbConnection, snapshot_id: int, snapshot: Snapshot, run: StaticRun) -> int:
    row = conn.execute(
        """
        INSERT INTO project_code_intel_static_runs (
            snapshot_id, collection, repo, commit_sha, sarif_path, sarif_sha256,
            run_index, tool_name, tool_version, semantic_version,
            information_uri, automation_id, metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (snapshot_id, sarif_path, sarif_sha256, run_index)
        DO UPDATE SET tool_name = EXCLUDED.tool_name,
                      tool_version = EXCLUDED.tool_version,
                      semantic_version = EXCLUDED.semantic_version,
                      information_uri = EXCLUDED.information_uri,
                      automation_id = EXCLUDED.automation_id,
                      metadata = EXCLUDED.metadata
        RETURNING id
        """,
        [
            snapshot_id,
            snapshot.collection,
            snapshot.repo,
            snapshot.commit_sha,
            run.sarif_path,
            run.sarif_sha256,
            run.run_index,
            run.tool_name,
            run.tool_version,
            run.semantic_version,
            run.information_uri,
            run.automation_id,
            db.compact_json(run.metadata, default=str),
        ],
    ).fetchone()
    return row_int(db.require_row(row, "insert static run"), "id")


def _insert_static_rule(conn: db.DbConnection, run_id: int, snapshot: Snapshot, rule: StaticRule) -> None:
    _ = conn.execute(
        """
        INSERT INTO project_code_intel_static_rules (
            run_id, collection, repo, rule_id, name, short_description,
            full_description, default_level, help_uri, properties, metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
        ON CONFLICT (run_id, rule_id)
        DO UPDATE SET name = EXCLUDED.name,
                      short_description = EXCLUDED.short_description,
                      full_description = EXCLUDED.full_description,
                      default_level = EXCLUDED.default_level,
                      help_uri = EXCLUDED.help_uri,
                      properties = EXCLUDED.properties,
                      metadata = EXCLUDED.metadata
        """,
        [
            run_id,
            snapshot.collection,
            snapshot.repo,
            rule.rule_id,
            rule.name,
            rule.short_description,
            rule.full_description,
            rule.default_level,
            rule.help_uri,
            db.compact_json(rule.properties, default=str),
            db.compact_json(rule.metadata, default=str),
        ],
    )


def _insert_static_finding(
    conn: db.DbConnection, run_id: int, snapshot_id: int, snapshot: Snapshot, finding: StaticFinding
) -> int:
    row = conn.execute(
        """
        INSERT INTO project_code_intel_static_findings (
            run_id, snapshot_id, collection, repo, commit_sha, finding_key,
            rule_id, rule_index, level, kind, message, baseline_state,
            primary_source_path, primary_uri, line_start, line_end,
            column_start, column_end, fingerprints, suppressions,
            properties, raw_result
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)
        ON CONFLICT (run_id, finding_key)
        DO UPDATE SET level = EXCLUDED.level,
                      kind = EXCLUDED.kind,
                      message = EXCLUDED.message,
                      baseline_state = EXCLUDED.baseline_state,
                      primary_source_path = EXCLUDED.primary_source_path,
                      primary_uri = EXCLUDED.primary_uri,
                      line_start = EXCLUDED.line_start,
                      line_end = EXCLUDED.line_end,
                      column_start = EXCLUDED.column_start,
                      column_end = EXCLUDED.column_end,
                      fingerprints = EXCLUDED.fingerprints,
                      suppressions = EXCLUDED.suppressions,
                      properties = EXCLUDED.properties,
                      raw_result = EXCLUDED.raw_result
        RETURNING id
        """,
        [
            run_id,
            snapshot_id,
            snapshot.collection,
            snapshot.repo,
            snapshot.commit_sha,
            finding.finding_key,
            finding.rule_id,
            finding.rule_index,
            finding.level,
            finding.kind,
            finding.message,
            finding.baseline_state,
            finding.primary_source_path,
            finding.primary_uri,
            finding.line_start,
            finding.line_end,
            finding.column_start,
            finding.column_end,
            db.compact_json(finding.fingerprints, default=str),
            db.compact_json(finding.suppressions, default=str),
            db.compact_json(finding.properties, default=str),
            db.compact_json(finding.raw_result, default=str),
        ],
    ).fetchone()
    return row_int(db.require_row(row, "insert static finding"), "id")


def _insert_static_locations(conn: db.DbConnection, finding_id: int, locations: list[StaticLocation]) -> int:
    inserted = 0
    for location in locations:
        _ = conn.execute(
            """
            INSERT INTO project_code_intel_static_locations (
                finding_id, ordinal, location_kind, source_path, uri, message,
                line_start, line_end, column_start, column_end, snippet, properties
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            [
                finding_id,
                location.ordinal,
                location.location_kind,
                location.source_path,
                location.uri,
                location.message,
                location.line_start,
                location.line_end,
                location.column_start,
                location.column_end,
                location.snippet,
                db.compact_json(location.properties, default=str),
            ],
        )
        inserted += 1
    return inserted


def _insert_static_code_flow_steps(conn: db.DbConnection, finding_id: int, code_flows: list[StaticCodeFlowStep]) -> int:
    inserted = 0
    for step in code_flows:
        _ = conn.execute(
            """
            INSERT INTO project_code_intel_static_code_flows (
                finding_id, flow_index, thread_index, step_index, source_path,
                uri, message, line_start, line_end, column_start, column_end,
                importance, properties
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            [
                finding_id,
                step.flow_index,
                step.thread_index,
                step.step_index,
                step.source_path,
                step.uri,
                step.message,
                step.line_start,
                step.line_end,
                step.column_start,
                step.column_end,
                step.importance,
                db.compact_json(step.properties, default=str),
            ],
        )
        inserted += 1
    return inserted


def insert_static_runs(
    conn: db.DbConnection,
    *,
    snapshot_ids_by_repo: dict[str, int],
    snapshot_by_repo: dict[str, Snapshot],
    runs: list[StaticRun],
) -> dict[str, int]:
    counts = _empty_static_counts()
    for run in runs:
        snapshot_id = snapshot_ids_by_repo.get(run.repo)
        snapshot = snapshot_by_repo.get(run.repo)
        if snapshot_id is None or snapshot is None:
            continue
        run_id = _insert_static_run(conn, snapshot_id, snapshot, run)
        counts["static_runs"] += 1
        for rule in run.rules:
            _insert_static_rule(conn, run_id, snapshot, rule)
            counts["static_rules"] += 1
        for finding in run.findings:
            finding_id = _insert_static_finding(conn, run_id, snapshot_id, snapshot, finding)
            counts["static_findings"] += 1
            # Both DELETEs run before any child INSERTs so a concurrent reader never
            # sees half-replaced child rows. The test asserts on this ordering.
            _ = conn.execute("DELETE FROM project_code_intel_static_locations WHERE finding_id = %s", [finding_id])
            _ = conn.execute("DELETE FROM project_code_intel_static_code_flows WHERE finding_id = %s", [finding_id])
            counts["static_locations"] += _insert_static_locations(conn, finding_id, finding.locations)
            counts["static_code_flow_steps"] += _insert_static_code_flow_steps(conn, finding_id, finding.code_flows)
    return counts
