-- Optional role split for project-code-intelligence.
--
-- The default deployment runs pci-index (writer) and pci-mcp (reader) under the
-- same database user. This script creates a separate read-only role for the
-- MCP server so that an exploited MCP boundary cannot write to the database.
--
-- Usage:
--   psql "$PROJECT_CODE_INTELLIGENCE_DATABASE_URL" \
--     -v reader_password="'choose-a-strong-password'" \
--     -f scripts/setup_database_roles.sql
--
-- After running this once, set the following env vars for the MCP server only:
--   PROJECT_CODE_INTELLIGENCE_MCP_PGVECTOR_USER=codeintel_reader
--   PROJECT_CODE_INTELLIGENCE_MCP_PGVECTOR_PASS=<reader_password>
-- (or supply PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL with the reader DSN).
--
-- The writer role (existing user) remains unchanged.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'codeintel_reader') THEN
        EXECUTE format('CREATE ROLE codeintel_reader LOGIN PASSWORD %L', :reader_password);
    END IF;
END
$$;

-- Reader can connect to the database and read schema metadata.
GRANT CONNECT ON DATABASE codeintel TO codeintel_reader;
GRANT USAGE ON SCHEMA public TO codeintel_reader;

-- Read-only on every project_code_intel_* table that exists today.
GRANT SELECT ON ALL TABLES IN SCHEMA public TO codeintel_reader;

-- And on any future tables created by the writer.
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO codeintel_reader;

-- Explicitly revoke write privileges in case ALL TABLES granted them previously.
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA public FROM codeintel_reader;
REVOKE CREATE ON SCHEMA public FROM codeintel_reader;
