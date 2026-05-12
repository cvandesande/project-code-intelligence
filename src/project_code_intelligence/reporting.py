"""Human-readable ingest report construction."""

from __future__ import annotations

from collections import Counter

from project_code_intelligence import profile_context
from project_code_intelligence.models import CHUNKER_VERSION, SCHEMA_VERSION, SOURCE_TYPE, JsonObject, RepoIngest


def report_ingests(ingests: list[RepoIngest], *, embeddings: bool) -> JsonObject:
    files = [item for ingest in ingests for item in ingest.files]
    records = [item for ingest in ingests for item in ingest.records]
    skipped = Counter(file.skipped_reason for file in files if file.skipped_reason)
    report_metadata_keys = profile_context.active_profile.report_metadata_keys()
    return {
        "source_type": SOURCE_TYPE,
        "schema_version": SCHEMA_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "profile": profile_context.active_profile.name,
        "profile_version": profile_context.active_profile.version,
        "collections": sorted({ingest.snapshot.collection for ingest in ingests}),
        "modes": dict(sorted(Counter(ingest.mode for ingest in ingests).items())),
        "repos": [ingest.snapshot.repo for ingest in ingests],
        "snapshots": [
            {
                "collection": ingest.snapshot.collection,
                "repo": ingest.snapshot.repo,
                "branch": ingest.snapshot.branch,
                "commit_sha": ingest.snapshot.commit_sha,
                "tree_sha": ingest.snapshot.tree_sha,
                "dirty": ingest.snapshot.dirty,
                "mode": ingest.mode,
                "previous_snapshot_id": ingest.previous_snapshot_id,
                "changed_files": len(ingest.changed_paths),
                "unchanged_files": len(ingest.unchanged_paths),
                "deleted_files": len(ingest.deleted_paths),
            }
            for ingest in ingests
        ],
        "files": len(files),
        "changed_files": sum(len(ingest.changed_paths) for ingest in ingests),
        "unchanged_files": sum(len(ingest.unchanged_paths) for ingest in ingests),
        "deleted_files": sum(len(ingest.deleted_paths) for ingest in ingests),
        "parseable_files": sum(1 for file in files if not file.skipped_reason),
        "parsed_files": sum(
            1
            for ingest in ingests
            for file in ingest.files
            if file.source_path in ingest.changed_paths and not file.skipped_reason
        ),
        "skipped_files": sum(1 for file in files if file.skipped_reason),
        "skipped_reasons": dict(sorted(skipped.items())),
        "records": len(records),
        "edges": sum(len(ingest.edges) for ingest in ingests),
        "parser_failures": sum(len(ingest.parser_failures) for ingest in ingests),
        "records_by_type": dict(sorted(Counter(record.record_type for record in records).items())),
        "files_by_language": dict(sorted(Counter(file.language for file in files).items())),
        "files_by_class": dict(sorted(Counter(file.content_class for file in files).items())),
        "confidence_by_kind": dict(sorted(Counter(record.confidence_kind for record in records).items())),
        "embeddings": embeddings,
        "examples": [
            {
                "record_type": record.record_type,
                "record_id": record.record_id,
                "title": record.title,
                "summary": record.summary,
                "confidence_kind": record.confidence_kind,
                "metadata": {key: value for key in report_metadata_keys if (value := record.metadata.get(key))},
            }
            for record in records[:12]
        ],
    }
