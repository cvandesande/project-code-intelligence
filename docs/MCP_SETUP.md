# MCP Setup

`project-code-intelligence` exposes a stdio MCP server through `pci-mcp`.
The server reads from Postgres/pgvector and does not need to run in Docker.

Use Docker Compose for local dependencies such as Postgres and optional
embedding servers. Point your MCP client at the installed `pci-mcp` command.

## Recommended Scope

Use one local database per indexed repository or workspace. Use:

- the inferred database for that repository or workspace.
- a collection inside that database for a workspace or project family.
- MCP tool `repo` filters for individual repositories inside that collection.

For normal use, you do not need to set a collection environment variable.
`pci-index` infers the database and collection from the paths you pass:

- one repo path: database scope and collection use the repo directory
- multiple repo paths: database scope and collection use the current working
  directory; pass repo subdirectories from that workspace

For a workspace with related repositories:

```sh
cd /path/to/workspace
pci-index repo-a repo-b repo-c
```

The MCP server also infers its collection from its configured working directory
when `PROJECT_CODE_INTELLIGENCE_COLLECTION` is unset. Set the MCP `cwd` to the
same repo or workspace directory you used when indexing. Agents then use repo
keys such as `repo-a` or `repo-b` in tool calls. Do not use absolute filesystem
paths as repo filters. Run `code_intel_status` without a repo filter to see
available collection and repo keys.

Use explicit collection configuration only when you want a collection name
different from the inferred directory name:

```sh
pci-index --collection workspace-name repo-a repo-b repo-c
```

## Security Model

Collections are an organization and safety feature, not a database security
boundary. The default local setup now gives each repository or workspace its
own inferred database, then scopes normal MCP tool calls to the configured or
inferred collection inside that database.

Do not rely on collection filters as the only protection between repositories
with different trust or sensitivity levels. If the same database credentials can
read every collection, then direct database access, a misconfigured MCP `cwd`, a
collection override, or a server bug could expose data from another collection.

Recommended deployment by sensitivity:

- Personal workstation with trusted assistants: one database with multiple
  collections is usually reasonable for closely related repositories, but the
  default inferred database per repository/workspace is easier to reset.
- Unrelated private repos with different sensitivity: prefer separate databases
  or separate database users.
- Team, shared, or untrusted-agent access: use stronger isolation such as
  separate databases, scoped database roles, or PostgreSQL row-level security.

Embeddings, SARIF findings, paths, symbols, and snippets can all reveal source
details. Treat database dumps and vector indexes as source-derived private data.
Remote embedding endpoints also receive source-derived text; use local
embeddings unless sending that text to the provider is acceptable.

## Database Configuration

The local Docker Compose Postgres service works without extra configuration.
Host tools infer a project database name from the repo/workspace path when no
database is explicitly configured.

For an external database, prefer one database URL. If the URL contains a
database path, that database is used exactly:

```sh
PROJECT_CODE_INTELLIGENCE_DATABASE_URL=postgresql://user:password@host:5432/database?sslmode=prefer
```

If the URL omits the database path, host, port, credentials, and query options
come from the URL, but the database name is inferred from the repo/workspace:

```sh
PROJECT_CODE_INTELLIGENCE_DATABASE_URL=postgresql://user:password@host:5432?sslmode=prefer
```

For first-use bootstrap, use PostgreSQL admin credentials once with
`pci-doctor --init-postgres`. Doctor creates/updates the cluster-level
`pci_index_admin` role, installs pgvector into `template1`, writes the
non-superuser `pci-index` credentials to
`${XDG_CONFIG_HOME:-~/.config}/project-code-intelligence/pci-index.env` with
`0600` permissions, and prints the same
`PROJECT_CODE_INTELLIGENCE_DATABASE_ADMIN_*` exports for copy/paste or secret
manager use. It does not create a project database. Then `pci-index` loads the
user config when those variables are unset and creates the inferred project
database and schema before indexing. Use `pci-index --init-db` when you want to
initialize the database/schema and exit without scanning:

```sh
PROJECT_CODE_INTELLIGENCE_DATABASE_URL=postgresql://host:5432?sslmode=prefer
PROJECT_CODE_INTELLIGENCE_POSTGRES_ADMIN_USER=postgres
PROJECT_CODE_INTELLIGENCE_POSTGRES_ADMIN_PASSWORD=admin-password
pci-doctor --init-postgres

pci-index --init-db .
pci-index .
```

Use `pci-doctor --init-postgres --no-write-config` when you want only printed
exports and no user config file.

When admin variables are set for an inferred database, generated scoped roles
override credentials embedded in `PROJECT_CODE_INTELLIGENCE_DATABASE_URL`.
Set `PROJECT_CODE_INTELLIGENCE_DATABASE_USER` and
`PROJECT_CODE_INTELLIGENCE_DATABASE_PASSWORD` only when you intentionally want
to force explicit runtime credentials.

For MCP clients, set the same base URL and scope path so `pci-mcp` infers the
same project database as `pci-index`:

```sh
PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL=postgresql://host:5432?sslmode=prefer
PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH=/path/to/repo-or-workspace
PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER=project_ro
PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD=password
```

When `PROJECT_CODE_INTELLIGENCE_DATABASE_ADMIN_USER` and
`PROJECT_CODE_INTELLIGENCE_DATABASE_ADMIN_PASSWORD` are available, `pci-index`
prints the project-specific `Export for pci-mcp (RO)` block after
`pci-index --init-db` and ordinary index runs.

To print copy/paste-ready project-scoped client configuration and the required
shell exports, use `--mcp-config`:

```sh
pci-index --init-db --mcp-config codex .
pci-index --init-db --mcp-config claude .
pci-index --init-db --mcp-config opencode .
```

Generated client config uses the generic server key
`project-code-intelligence`. That key is intentionally reused because the
config belongs to the indexed repo/workspace, not to global client config. Do
not paste generated snippets into a global MCP config. Generated client config
references environment variables instead of embedding credential values; the
export block printed after the config contains the project read-only
credentials. The collection is inferred from the project `cwd`; use
`PROJECT_CODE_INTELLIGENCE_COLLECTION` only as an explicit MCP runtime scope.
For index runs, prefer `pci-index --collection NAME`; an inherited collection
environment variable is ignored unless
`PROJECT_CODE_INTELLIGENCE_ALLOW_COLLECTION_OVERRIDE=1` is also set. Use
`--mcp-server-name NAME` only when you are deliberately creating a
non-project-scoped setup.

If those MCP-specific variables are unset, `pci-mcp` falls back to the generic
database variables.

The one-time bootstrap user usually needs local Postgres superuser privileges
because pgvector is not trusted. `pci-doctor --init-postgres` creates a
non-superuser `pci_index_admin` role with `CREATEDB` and `CREATEROLE`; new
project databases inherit pgvector from `template1`. If a project database was
created before `template1` had pgvector, reset that inferred database and index
again.

If admin variables are not set, `pci-index` uses the normal configured
credentials. Those credentials must already be able to create/use the inferred
database, and no separate RW/RO roles are generated.

If your MCP client or secret manager separates credentials, leave them out of
the URL and pass them separately:

```sh
PROJECT_CODE_INTELLIGENCE_DATABASE_URL=postgresql://host:5432?sslmode=prefer
PROJECT_CODE_INTELLIGENCE_DATABASE_USER=user
PROJECT_CODE_INTELLIGENCE_DATABASE_PASSWORD=password
```

Set `PGVECTOR_DB` only when you want to disable database-name inference and use
a fixed database name.

`pci-index --reset .` drops only the inferred PCI-managed database for that
repo/workspace scope. Use `pci-doctor --clean` for broad local cleanup.

Do not commit real database credentials or private export files. Generated
project-scoped snippets reference environment variables and are suitable to
share only when the repo path and server command are also acceptable to share.
The MCP export block contains read-only database credentials; load those values
from your shell, direnv, a private env file, or a system secret manager. The
`pci-doctor --init-postgres` user config contains non-superuser credentials for
`pci-index`; keep it private and at `0600`.

## Codex

`pci-index --mcp-config codex` emits config for the indexed repo's
`.codex/config.toml`.

Generate the snippet from the indexed repo/workspace:

```sh
pci-index --init-db --mcp-config codex .
```

The command prints the project target path before the TOML:

```text
Codex project-scoped MCP config
Write this snippet to: /home/you/src/project-code-intelligence/.codex/config.toml
This snippet is project-scoped and references environment variables for credentials.
Load the required environment variables below before starting the MCP client.
Do not paste this into a global MCP config; the server key is intentionally reused per project.
```

```toml
[mcp_servers.project-code-intelligence]
command = "/home/you/.local/bin/pci-mcp"
cwd = "/home/you/src/project-code-intelligence"
startup_timeout_sec = 20
tool_timeout_sec = 120
env_vars = [
  "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL",
  "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER",
  "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD",
  "PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH",
]
```

```sh
export PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL='postgresql://host:5432?sslmode=prefer'
export PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER=project_ro
export PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD=password
export PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH=/home/you/src/project-code-intelligence
```

For a single-repo local setup using the Compose database, you can omit the
`env_vars` list and export block if the MCP client launches `pci-mcp` from the
repo root and the default local database credentials are acceptable.

## Claude Code

`pci-index --mcp-config claude` emits config for the indexed repo's
project-scoped `.mcp.json`.

Generate the snippet from the indexed repo/workspace:

```sh
pci-index --init-db --mcp-config claude .
```

Claude Code treats `.mcp.json` as project scoped and prompts before using it in
a trusted workspace. The generated `.mcp.json` uses Claude's `${VAR}`
environment expansion, so credentials stay in the accompanying export block or
your secret manager.

```json
{
  "mcpServers": {
    "project-code-intelligence": {
      "type": "stdio",
      "command": "/home/you/.local/bin/pci-mcp",
      "args": [],
      "cwd": "/home/you/src/project-code-intelligence",
      "env": {
        "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL": "${PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL}",
        "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER": "${PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER}",
        "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD": "${PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD}",
        "PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH": "${PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH}"
      }
    }
  }
}
```

Use the export block printed after the JSON before launching Claude Code.

## OpenCode

`pci-index --mcp-config opencode` emits config for the indexed repo's
`opencode.json`. OpenCode config is JSON or JSONC and defines MCP servers under
`mcp`.

Generate the snippet from the indexed repo/workspace:

```sh
pci-index --init-db --mcp-config opencode .
```

The generated `opencode.json` uses OpenCode's `{env:VAR}` substitution, so
credentials stay in the accompanying export block or your secret manager.

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "project-code-intelligence": {
      "type": "local",
      "command": ["/home/you/.local/bin/pci-mcp"],
      "enabled": true,
      "cwd": "/home/you/src/project-code-intelligence",
      "environment": {
        "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL": "{env:PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL}",
        "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER": "{env:PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER}",
        "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD": "{env:PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD}",
        "PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH": "{env:PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH}"
      }
    }
  }
}
```

Use the export block printed after the JSON before launching OpenCode.

## Other MCP Clients

`pci-index --mcp-config` intentionally emits project-scoped config only for
Codex, Claude Code, and OpenCode. For other clients, use the `env` export block
with that client's project/workspace configuration if it has one.

## Agent Guidance

For repositories that use this MCP server heavily, copy
[`docs/examples/AGENTS.md`](examples/AGENTS.md) into the indexed repository.

The important behavior for assistants is:

1. Call `code_intel_status` first.
2. Use the collection and repo keys reported by `code_intel_status`.
3. For implementation questions, pass `file_role: source` when you want to
   exclude tests and docs rather than just rank source higher.
4. Use repo filters such as `openwrt`, `ask-cmm`, or `fci`, not absolute paths.
5. Treat the MCP index as a navigation aid and verify important behavior
   against the working tree.
