CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS project_code_intel_schema_migrations (
    version text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS project_code_intel_snapshots (
    id bigserial PRIMARY KEY,
    collection text NOT NULL DEFAULT 'default',
    repo text NOT NULL,
    repo_role text NOT NULL,
    branch text,
    commit_sha text NOT NULL,
    tree_sha text NOT NULL,
    dirty boolean NOT NULL DEFAULT false,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS project_code_intel_files (
    id bigserial PRIMARY KEY,
    snapshot_id bigint NOT NULL REFERENCES project_code_intel_snapshots(id)
        ON DELETE CASCADE,
    collection text NOT NULL DEFAULT 'default',
    repo text NOT NULL,
    repo_role text NOT NULL,
    branch text,
    commit_sha text NOT NULL,
    tree_sha text NOT NULL,
    source_path text NOT NULL,
    git_blob_sha text,
    file_sha256 text,
    size_bytes bigint,
    language text NOT NULL,
    file_role text NOT NULL,
    content_class text NOT NULL,
    is_generated boolean NOT NULL DEFAULT false,
    is_vendor boolean NOT NULL DEFAULT false,
    is_test boolean NOT NULL DEFAULT false,
    is_source boolean NOT NULL DEFAULT false,
    is_build boolean NOT NULL DEFAULT false,
    is_config boolean NOT NULL DEFAULT false,
    is_doc boolean NOT NULL DEFAULT false,
    skipped_reason text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (snapshot_id, source_path)
);

-- Records v2: content-addressed, no snapshot/branch/commit columns. A record
-- row is identified by its content (collection, repo, source_path,
-- file_sha256, record_type, record_id, embedding_text_hash), not by which
-- snapshot produced it. Snapshot scoping happens on read via a join through
-- project_code_intel_files on (collection, repo, source_path, file_sha256).
-- On a content conflict (same key, same reparsed file), the insert refreshes
-- the fields that can drift independently of embedding_text -- summary,
-- display, metadata, confidence/severity/tool fields, parser/chunker
-- versions -- rather than freezing them at first write; that keeps the
-- write bounded to files actually reparsed instead of copying every row.
--
-- file_sha256 is NOT NULL with '' as the sentinel for "no hash" (inventory
-- read failures), not NULL: Postgres unique constraints treat NULLs as
-- distinct, so a NULL here would never dedupe against itself across
-- ingests. project_code_intel_files.file_sha256 stays nullable; joins from
-- records to files compare `r.file_sha256 = coalesce(f.file_sha256, '')`.
CREATE TABLE IF NOT EXISTS project_code_intel_records (
    id bigserial PRIMARY KEY,
    collection text NOT NULL DEFAULT 'default',
    repo text NOT NULL,
    source_path text NOT NULL,
    file_sha256 text NOT NULL DEFAULT '',
    language text NOT NULL,
    file_role text NOT NULL,
    content_class text NOT NULL,
    record_type text NOT NULL,
    record_id text NOT NULL,
    parent_record_id text,
    title text NOT NULL,
    summary text NOT NULL,
    embedding_text text NOT NULL,
    display_content text NOT NULL,
    embedding_text_hash text NOT NULL,
    display_hash text NOT NULL,
    line_start integer,
    line_end integer,
    byte_start integer,
    byte_end integer,
    symbol text,
    symbol_kind text,
    confidence_kind text NOT NULL DEFAULT 'high_confidence_fact',
    confidence numeric,
    tool text,
    rule_id text,
    severity text,
    analyzer text,
    analyzer_version text,
    parser text,
    parser_version text,
    chunker_version text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding vector,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    search_document tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', title), 'A') ||
        setweight(to_tsvector('english', summary), 'B') ||
        setweight(to_tsvector('english', embedding_text), 'C') ||
        setweight(to_tsvector('simple', source_path), 'A') ||
        setweight(to_tsvector('simple', coalesce(symbol, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(rule_id, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(record_type, '')), 'B')
    ) STORED,
    UNIQUE (collection, repo, source_path, file_sha256, record_type, record_id, embedding_text_hash)
);

CREATE TABLE IF NOT EXISTS project_code_intel_edges (
    id bigserial PRIMARY KEY,
    snapshot_id bigint NOT NULL REFERENCES project_code_intel_snapshots(id)
        ON DELETE CASCADE,
    collection text NOT NULL DEFAULT 'default',
    repo text NOT NULL,
    commit_sha text NOT NULL,
    source_record_id text NOT NULL,
    target_record_id text,
    edge_type text NOT NULL,
    source_symbol text,
    target_symbol text,
    source_path text,
    target_path text,
    confidence_kind text NOT NULL DEFAULT 'approximate_fact',
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS project_code_intel_parser_failures (
    id bigserial PRIMARY KEY,
    snapshot_id bigint NOT NULL REFERENCES project_code_intel_snapshots(id)
        ON DELETE CASCADE,
    collection text NOT NULL DEFAULT 'default',
    repo text NOT NULL,
    commit_sha text NOT NULL,
    source_path text NOT NULL,
    language text,
    parser text NOT NULL,
    error text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (snapshot_id, source_path, parser, error)
);

CREATE TABLE IF NOT EXISTS project_code_intel_static_runs (
    id bigserial PRIMARY KEY,
    snapshot_id bigint NOT NULL REFERENCES project_code_intel_snapshots(id)
        ON DELETE CASCADE,
    collection text NOT NULL DEFAULT 'default',
    repo text NOT NULL,
    commit_sha text NOT NULL,
    sarif_path text NOT NULL,
    sarif_sha256 text NOT NULL,
    run_index integer NOT NULL,
    tool_name text NOT NULL,
    tool_version text,
    semantic_version text,
    information_uri text,
    automation_id text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (snapshot_id, sarif_path, sarif_sha256, run_index)
);

CREATE TABLE IF NOT EXISTS project_code_intel_static_rules (
    id bigserial PRIMARY KEY,
    run_id bigint NOT NULL REFERENCES project_code_intel_static_runs(id)
        ON DELETE CASCADE,
    collection text NOT NULL DEFAULT 'default',
    repo text NOT NULL,
    rule_id text NOT NULL,
    name text,
    short_description text,
    full_description text,
    default_level text,
    help_uri text,
    properties jsonb NOT NULL DEFAULT '{}'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (run_id, rule_id)
);

CREATE TABLE IF NOT EXISTS project_code_intel_static_findings (
    id bigserial PRIMARY KEY,
    run_id bigint NOT NULL REFERENCES project_code_intel_static_runs(id)
        ON DELETE CASCADE,
    snapshot_id bigint NOT NULL REFERENCES project_code_intel_snapshots(id)
        ON DELETE CASCADE,
    collection text NOT NULL DEFAULT 'default',
    repo text NOT NULL,
    commit_sha text NOT NULL,
    finding_key text NOT NULL,
    rule_id text NOT NULL,
    rule_index integer,
    level text,
    kind text,
    message text NOT NULL,
    baseline_state text,
    primary_source_path text,
    primary_uri text,
    line_start integer,
    line_end integer,
    column_start integer,
    column_end integer,
    fingerprints jsonb NOT NULL DEFAULT '{}'::jsonb,
    suppressions jsonb NOT NULL DEFAULT '[]'::jsonb,
    properties jsonb NOT NULL DEFAULT '{}'::jsonb,
    raw_result jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (run_id, finding_key)
);

CREATE TABLE IF NOT EXISTS project_code_intel_static_locations (
    id bigserial PRIMARY KEY,
    finding_id bigint NOT NULL REFERENCES project_code_intel_static_findings(id)
        ON DELETE CASCADE,
    ordinal integer NOT NULL,
    location_kind text NOT NULL,
    source_path text,
    uri text,
    message text,
    line_start integer,
    line_end integer,
    column_start integer,
    column_end integer,
    snippet text,
    properties jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS project_code_intel_static_code_flows (
    id bigserial PRIMARY KEY,
    finding_id bigint NOT NULL REFERENCES project_code_intel_static_findings(id)
        ON DELETE CASCADE,
    flow_index integer NOT NULL,
    thread_index integer NOT NULL,
    step_index integer NOT NULL,
    source_path text,
    uri text,
    message text,
    line_start integer,
    line_end integer,
    column_start integer,
    column_end integer,
    importance text,
    properties jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- pci check regression ratchet: one frozen baseline per (collection, repo,
-- branch), the same identity scheme snapshots use. `branch` is nullable for a
-- detached HEAD, matching project_code_intel_snapshots.branch -- NULL is a
-- distinct bucket, not a branch literally named "HEAD". Postgres treats every
-- NULL as distinct under UNIQUE, so this constraint only dedupes named
-- branches; storage/check.py does the NULL-safe find-or-insert itself rather
-- than relying on ON CONFLICT for the detached-HEAD case. Additive-only
-- compatibility event; no existing table is touched.
CREATE TABLE IF NOT EXISTS project_code_intel_check_baselines (
    id bigserial PRIMARY KEY,
    collection text NOT NULL,
    repo text NOT NULL,
    branch text,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (collection, repo, branch)
);

CREATE TABLE IF NOT EXISTS project_code_intel_check_baseline_findings (
    id bigserial PRIMARY KEY,
    baseline_id bigint NOT NULL REFERENCES project_code_intel_check_baselines(id)
        ON DELETE CASCADE,
    fingerprint text NOT NULL,
    rule_id text NOT NULL,
    level text,
    tool_name text,
    message text,
    primary_source_path text,
    line_start integer,
    line_end integer,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (baseline_id, fingerprint)
);

CREATE TABLE IF NOT EXISTS project_code_intel_index_runs (
    id bigserial PRIMARY KEY,
    collection text NOT NULL DEFAULT 'default',
    repos jsonb NOT NULL DEFAULT '[]'::jsonb,
    repo_modes jsonb NOT NULL DEFAULT '{}'::jsonb,
    pid integer,
    host text,
    phase text,
    progress jsonb NOT NULL DEFAULT '{}'::jsonb,
    started_at timestamptz NOT NULL DEFAULT now(),
    heartbeat_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    exit_code integer,
    interrupted boolean NOT NULL DEFAULT false,
    error text
);

ALTER TABLE project_code_intel_snapshots
    ADD COLUMN IF NOT EXISTS collection text NOT NULL DEFAULT 'default';

ALTER TABLE project_code_intel_files
    ADD COLUMN IF NOT EXISTS collection text NOT NULL DEFAULT 'default';

ALTER TABLE project_code_intel_records
    ADD COLUMN IF NOT EXISTS collection text NOT NULL DEFAULT 'default';

ALTER TABLE project_code_intel_edges
    ADD COLUMN IF NOT EXISTS collection text NOT NULL DEFAULT 'default';

ALTER TABLE project_code_intel_parser_failures
    ADD COLUMN IF NOT EXISTS collection text NOT NULL DEFAULT 'default';

ALTER TABLE project_code_intel_static_runs
    ADD COLUMN IF NOT EXISTS collection text NOT NULL DEFAULT 'default';

ALTER TABLE project_code_intel_static_rules
    ADD COLUMN IF NOT EXISTS collection text NOT NULL DEFAULT 'default';

ALTER TABLE project_code_intel_static_findings
    ADD COLUMN IF NOT EXISTS collection text NOT NULL DEFAULT 'default';

ALTER TABLE project_code_intel_snapshots
    DROP CONSTRAINT IF EXISTS project_code_intel_snapshots_repo_commit_sha_tree_sha_key;

CREATE UNIQUE INDEX IF NOT EXISTS project_code_intel_snapshots_collection_repo_tree_idx
    ON project_code_intel_snapshots (collection, repo, commit_sha, tree_sha);

-- code-intel-schema-v3: records v2 migration. Runs once per database --
-- guarded on the presence of the old snapshot_id column, which the v2
-- CREATE TABLE above never creates. Newest-snapshot-wins, non-null
-- embedding preferred, matches the DISTINCT ON key the new UNIQUE
-- constraint uses so the copy can never violate it.
DO $records_v2_migration$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'project_code_intel_records'
          AND column_name = 'snapshot_id'
    ) THEN
        EXECUTE $ddl$
            CREATE TABLE project_code_intel_records_v2 (
                id bigserial PRIMARY KEY,
                collection text NOT NULL DEFAULT 'default',
                repo text NOT NULL,
                source_path text NOT NULL,
                file_sha256 text NOT NULL DEFAULT '',
                language text NOT NULL,
                file_role text NOT NULL,
                content_class text NOT NULL,
                record_type text NOT NULL,
                record_id text NOT NULL,
                parent_record_id text,
                title text NOT NULL,
                summary text NOT NULL,
                embedding_text text NOT NULL,
                display_content text NOT NULL,
                embedding_text_hash text NOT NULL,
                display_hash text NOT NULL,
                line_start integer,
                line_end integer,
                byte_start integer,
                byte_end integer,
                symbol text,
                symbol_kind text,
                confidence_kind text NOT NULL DEFAULT 'high_confidence_fact',
                confidence numeric,
                tool text,
                rule_id text,
                severity text,
                analyzer text,
                analyzer_version text,
                parser text,
                parser_version text,
                chunker_version text NOT NULL,
                metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
                embedding vector,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now(),
                search_document tsvector GENERATED ALWAYS AS (
                    setweight(to_tsvector('english', title), 'A') ||
                    setweight(to_tsvector('english', summary), 'B') ||
                    setweight(to_tsvector('english', embedding_text), 'C') ||
                    setweight(to_tsvector('simple', source_path), 'A') ||
                    setweight(to_tsvector('simple', coalesce(symbol, '')), 'A') ||
                    setweight(to_tsvector('simple', coalesce(rule_id, '')), 'A') ||
                    setweight(to_tsvector('simple', coalesce(record_type, '')), 'B')
                ) STORED,
                UNIQUE (collection, repo, source_path, file_sha256, record_type, record_id, embedding_text_hash)
            )
        $ddl$;

        EXECUTE $ddl$
            INSERT INTO project_code_intel_records_v2 (
                collection, repo, source_path, file_sha256, language, file_role,
                content_class, record_type, record_id, parent_record_id, title,
                summary, embedding_text, display_content, embedding_text_hash,
                display_hash, line_start, line_end, byte_start, byte_end, symbol,
                symbol_kind, confidence_kind, confidence, tool, rule_id, severity,
                analyzer, analyzer_version, parser, parser_version, chunker_version,
                metadata, embedding, created_at, updated_at
            )
            SELECT DISTINCT ON (
                r.collection, r.repo, r.source_path, coalesce(r.file_sha256, ''),
                r.record_type, r.record_id, r.embedding_text_hash
            )
                r.collection, r.repo, r.source_path, coalesce(r.file_sha256, ''), r.language, r.file_role,
                r.content_class, r.record_type, r.record_id, r.parent_record_id, r.title,
                r.summary, r.embedding_text, r.display_content, r.embedding_text_hash,
                r.display_hash, r.line_start, r.line_end, r.byte_start, r.byte_end, r.symbol,
                r.symbol_kind, r.confidence_kind, r.confidence, r.tool, r.rule_id, r.severity,
                r.analyzer, r.analyzer_version, r.parser, r.parser_version, r.chunker_version,
                r.metadata, r.embedding, r.created_at, r.updated_at
            FROM project_code_intel_records r
            JOIN project_code_intel_snapshots s ON s.id = r.snapshot_id
            ORDER BY
                r.collection, r.repo, r.source_path, coalesce(r.file_sha256, ''),
                r.record_type, r.record_id, r.embedding_text_hash,
                (r.embedding IS NULL) ASC, s.created_at DESC, r.id DESC
        $ddl$;

        EXECUTE 'DROP TABLE project_code_intel_records';
        EXECUTE 'ALTER TABLE project_code_intel_records_v2 RENAME TO project_code_intel_records';
    END IF;
END
$records_v2_migration$;

-- Extends the old (collection, repo, source_path) files index with
-- file_sha256, so reader joins from records to files (the new membership
-- join) can use one index. Old name dropped so a stale narrower index
-- doesn't linger; DROP IF EXISTS is a cheap catalog check once it is gone.
DROP INDEX IF EXISTS project_code_intel_files_collection_repo_path_idx;

CREATE INDEX IF NOT EXISTS project_code_intel_files_collection_repo_path_hash_idx
    ON project_code_intel_files (collection, repo, source_path, file_sha256);

ALTER TABLE project_code_intel_files
    ADD COLUMN IF NOT EXISTS is_untracked boolean NOT NULL DEFAULT false;

ALTER TABLE project_code_intel_files
    ADD COLUMN IF NOT EXISTS indexed_dirty boolean NOT NULL DEFAULT false;

CREATE INDEX IF NOT EXISTS project_code_intel_files_collection_class_idx
    ON project_code_intel_files (collection, repo, language, file_role, content_class);

CREATE INDEX IF NOT EXISTS project_code_intel_files_metadata_idx
    ON project_code_intel_files USING gin (metadata);

CREATE INDEX IF NOT EXISTS project_code_intel_records_search_idx
    ON project_code_intel_records USING gin (search_document);

CREATE INDEX IF NOT EXISTS project_code_intel_records_collection_type_idx
    ON project_code_intel_records (collection, repo, record_type, source_path);

-- records_snapshot_type_idx (snapshot_id, record_type) dropped in schema v3:
-- records no longer carry snapshot_id. Snapshot-scoped record_type lookups
-- go through project_code_intel_records_collection_type_idx joined against
-- files for the snapshot filter.

CREATE INDEX IF NOT EXISTS project_code_intel_records_rule_idx
    ON project_code_intel_records (rule_id)
    WHERE rule_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS project_code_intel_records_collection_symbol_idx
    ON project_code_intel_records (collection, repo, symbol)
    WHERE symbol IS NOT NULL;

CREATE INDEX IF NOT EXISTS project_code_intel_records_collection_file_line_idx
    ON project_code_intel_records (collection, repo, source_path, line_start, line_end);

CREATE INDEX IF NOT EXISTS project_code_intel_records_metadata_idx
    ON project_code_intel_records USING gin (metadata);

CREATE INDEX IF NOT EXISTS project_code_intel_edges_collection_lookup_idx
    ON project_code_intel_edges (collection, repo, edge_type, source_symbol, target_symbol);

CREATE INDEX IF NOT EXISTS project_code_intel_parser_failures_collection_idx
    ON project_code_intel_parser_failures (collection, repo, source_path);

CREATE INDEX IF NOT EXISTS project_code_intel_static_runs_collection_idx
    ON project_code_intel_static_runs (collection, repo, tool_name, created_at DESC);

CREATE INDEX IF NOT EXISTS project_code_intel_static_findings_collection_idx
    ON project_code_intel_static_findings (collection, repo, rule_id, primary_source_path);

CREATE INDEX IF NOT EXISTS project_code_intel_static_findings_snapshot_rule_idx
    ON project_code_intel_static_findings (snapshot_id, rule_id);

CREATE INDEX IF NOT EXISTS project_code_intel_static_findings_metadata_idx
    ON project_code_intel_static_findings USING gin (properties);

CREATE INDEX IF NOT EXISTS project_code_intel_static_locations_path_idx
    ON project_code_intel_static_locations (source_path, line_start, line_end);

CREATE INDEX IF NOT EXISTS project_code_intel_static_code_flows_path_idx
    ON project_code_intel_static_code_flows (source_path, line_start, line_end);

CREATE UNIQUE INDEX IF NOT EXISTS project_code_intel_edges_collection_unique_idx
    ON project_code_intel_edges (
        snapshot_id,
        collection,
        source_record_id,
        edge_type,
        coalesce(target_record_id, ''),
        coalesce(source_symbol, ''),
        coalesce(target_symbol, ''),
        coalesce(source_path, ''),
        coalesce(target_path, '')
    );

CREATE INDEX IF NOT EXISTS project_code_intel_index_runs_collection_idx
    ON project_code_intel_index_runs (collection, started_at DESC);

-- records_snapshot_symbol_def_idx (snapshot_id, symbol) dropped in schema
-- v3: resolve_edge_targets now joins symbol_definition records through
-- files for the snapshot filter, served by
-- project_code_intel_records_collection_symbol_idx.

-- Speeds up resolve_edge_targets: find unresolved edges by target_symbol within a snapshot.
CREATE INDEX IF NOT EXISTS project_code_intel_edges_snapshot_unresolved_idx
    ON project_code_intel_edges (snapshot_id, target_symbol)
    WHERE target_record_id IS NULL AND target_symbol IS NOT NULL;

DROP TRIGGER IF EXISTS project_code_intel_records_touch_updated_at
    ON project_code_intel_records;

CREATE OR REPLACE FUNCTION project_code_intel_touch_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

CREATE TRIGGER project_code_intel_records_touch_updated_at
BEFORE UPDATE ON project_code_intel_records
FOR EACH ROW
EXECUTE FUNCTION project_code_intel_touch_updated_at();
