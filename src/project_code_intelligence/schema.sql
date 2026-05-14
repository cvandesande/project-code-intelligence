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

CREATE TABLE IF NOT EXISTS project_code_intel_records (
    id bigserial PRIMARY KEY,
    snapshot_id bigint NOT NULL REFERENCES project_code_intel_snapshots(id)
        ON DELETE CASCADE,
    file_id bigint REFERENCES project_code_intel_files(id) ON DELETE CASCADE,
    collection text NOT NULL DEFAULT 'default',
    repo text NOT NULL,
    repo_role text NOT NULL,
    branch text,
    commit_sha text NOT NULL,
    tree_sha text NOT NULL,
    source_path text NOT NULL,
    file_sha256 text,
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
    UNIQUE (snapshot_id, record_type, record_id, embedding_text_hash)
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

CREATE INDEX IF NOT EXISTS project_code_intel_files_collection_repo_path_idx
    ON project_code_intel_files (collection, repo, source_path);

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

CREATE INDEX IF NOT EXISTS project_code_intel_records_snapshot_type_idx
    ON project_code_intel_records (snapshot_id, record_type);

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
