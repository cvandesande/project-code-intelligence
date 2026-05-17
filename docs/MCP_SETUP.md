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
`pci_index_admin` role, installs pgvector into `template1`, and prints the
`PROJECT_CODE_INTELLIGENCE_DATABASE_ADMIN_*` exports to keep for `pci-index`;
it does not create a project database. Then
`pci-index` creates the inferred project database and schema before indexing.
Use `pci-index --init-db` when you want to initialize the database/schema and
exit without scanning:

```sh
PROJECT_CODE_INTELLIGENCE_DATABASE_URL=postgresql://host:5432?sslmode=prefer
PROJECT_CODE_INTELLIGENCE_POSTGRES_ADMIN_USER=postgres
PROJECT_CODE_INTELLIGENCE_POSTGRES_ADMIN_PASSWORD=admin-password
pci-doctor --init-postgres

# Use the PROJECT_CODE_INTELLIGENCE_DATABASE_ADMIN_* exports printed above.
pci-index --init-db .
pci-index .
```

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

To print copy/paste-ready client configuration instead of shell exports, use
`--mcp-config`:

```sh
pci-index --init-db --mcp-config codex .
pci-index --init-db --mcp-config claude .
pci-index --init-db --mcp-config opencode .
```

Use `--mcp-server-name NAME` when you want a stable client-specific server key
instead of the default `pci-<collection>` name.

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

Do not commit real database credentials. Put private values in user-local MCP
configuration, a local ignored file, or your system secret manager.

## Codex

Codex stores MCP config in `~/.codex/config.toml`, or in a project-scoped
`.codex/config.toml` for trusted projects. The CLI and the VSCode Codex
extension share this configuration.

Prefer generating the snippet from the indexed repo/workspace:

```sh
pci-index --init-db --mcp-config codex .
```

```toml
[mcp_servers.project-code-intelligence]
command = "/home/you/.local/bin/pci-mcp"
cwd = "/home/you/src/project-code-intelligence"
startup_timeout_sec = 20
tool_timeout_sec = 120

[mcp_servers.project-code-intelligence.env]
PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL = "postgresql://host:5432?sslmode=prefer"
PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH = "/home/you/src/project-code-intelligence"
PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER = "project_ro"
PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD = "password"
```

For a single-repo local setup using the Compose database, the environment block
can be omitted.

## Claude Code

Claude Code supports local, user, and project MCP scopes. Project-scoped MCP
servers are stored in `.mcp.json`; user/local scoped servers are private to the
user. The CLI and the `anthropic.claude-code` VSCode extension share this
configuration.

Use project-scoped `.mcp.json` only for non-secret shared configuration. Keep
credentials in local/user configuration or environment variables.

Generate a private config snippet from the indexed repo/workspace:

```sh
pci-index --init-db --mcp-config claude .
```

```json
{
  "mcpServers": {
    "project-code-intelligence": {
      "type": "stdio",
      "command": "/home/you/.local/bin/pci-mcp",
      "args": [],
      "cwd": "/home/you/src/project-code-intelligence",
      "env": {
        "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL": "postgresql://host:5432?sslmode=prefer",
        "PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH": "/home/you/src/project-code-intelligence",
        "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER": "project_ro",
        "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD": "password"
      }
    }
  }
}
```

## Cline

The Cline VSCode extension (`saoudrizwan.claude-dev`) stores MCP server
configuration at a user-scoped JSON file. The simplest way to edit it is the
MCP Servers icon in the Cline panel, which opens the file. The schema uses a
top-level `mcpServers` key.

- macOS: `~/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- Linux: `~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- Windows: `%APPDATA%\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`

```json
{
  "mcpServers": {
    "project-code-intelligence": {
      "type": "stdio",
      "command": "/home/you/.local/bin/pci-mcp",
      "args": [],
      "cwd": "/home/you/src/project-code-intelligence",
      "env": {
        "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL": "postgresql://host:5432?sslmode=prefer",
        "PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH": "/home/you/src/project-code-intelligence",
        "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER": "project_ro",
        "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD": "password"
      }
    }
  }
}
```

The MCP Servers panel in Cline lists running servers and the tools each
exposes.

## GitHub Copilot Chat (VSCode)

Copilot Chat reads MCP server configuration from a user-scoped `mcp.json`, or
from a workspace `.vscode/mcp.json`. Add a server with the Command Palette
command `MCP: Add Server`, or edit the file directly. The schema uses a
top-level `servers` key (not `mcpServers`).

User-scoped file:

- macOS: `~/Library/Application Support/Code/User/mcp.json`
- Linux: `~/.config/Code/User/mcp.json`
- Windows: `%APPDATA%\Code\User\mcp.json`

```json
{
  "servers": {
    "project-code-intelligence": {
      "type": "stdio",
      "command": "/home/you/.local/bin/pci-mcp",
      "args": [],
      "cwd": "/home/you/src/project-code-intelligence",
      "env": {
        "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL": "postgresql://host:5432?sslmode=prefer",
        "PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH": "/home/you/src/project-code-intelligence",
        "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER": "project_ro",
        "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD": "password"
      }
    }
  },
  "inputs": []
}
```

After saving, click the inline `Start` action above the server entry in
`mcp.json`, or run `MCP: List Servers` from the Command Palette to view status.
The tools appear in the Tools picker when Copilot Chat is in agent mode.

For a workspace-scoped configuration that ships with the repo, place the same
content (without secret credentials) in `.vscode/mcp.json` at the workspace
root.

## OpenCode

OpenCode config is JSON or JSONC and defines MCP servers under `mcp`.

Generate a private config snippet from the indexed repo/workspace:

```sh
pci-index --init-db --mcp-config opencode .
```

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
        "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL": "postgresql://host:5432?sslmode=prefer",
        "PROJECT_CODE_INTELLIGENCE_DATABASE_SCOPE_PATH": "/home/you/src/project-code-intelligence",
        "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER": "project_ro",
        "PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD": "password"
      }
    }
  }
}
```

## Agent Guidance

For repositories that use this MCP server heavily, copy
[`docs/examples/AGENTS.md`](examples/AGENTS.md) into the indexed repository.

The important behavior for assistants is:

1. Call `code_intel_status` first.
2. Use the collection and repo keys reported by `code_intel_status`.
3. Use repo filters such as `openwrt`, `ask-cmm`, or `fci`, not absolute paths.
4. Treat the MCP index as a navigation aid and verify important behavior
   against the working tree.
