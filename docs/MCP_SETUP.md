# MCP Setup

`project-code-intelligence` exposes a stdio MCP server through `pci-mcp`.
The server reads from Postgres/pgvector and does not need to run in Docker.

Use Docker Compose for local dependencies such as Postgres and optional
embedding servers. Point your MCP client at the installed `pci-mcp` command.

## Recommended Scope

Use one database for one or more indexed repositories. Use:

- a collection for a workspace or project family.
- MCP tool `repo` filters for individual repositories inside that collection.

For normal use, you do not need to set a collection environment variable.
`pci-index` infers it from the paths you pass:

- one repo path: collection is the repo directory name
- multiple repo paths: collection is the common parent directory name

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

Use explicit collection configuration only when you want a name different from
the inferred directory name:

```sh
pci-index --collection workspace-name repo-a repo-b repo-c
```

## Security Model

Collections are an organization and safety feature, not a database security
boundary. The MCP server scopes normal tool calls to the configured or inferred
collection, which helps avoid accidental cross-repo results when several repos
share one database.

Do not rely on collection filters as the only protection between repositories
with different trust or sensitivity levels. If the same database credentials can
read every collection, then direct database access, a misconfigured MCP `cwd`, a
collection override, or a server bug could expose data from another collection.

Recommended deployment by sensitivity:

- Personal workstation with trusted assistants: one database with multiple
  collections is usually reasonable.
- Unrelated private repos with different sensitivity: prefer separate databases
  or separate database users.
- Team, shared, or untrusted-agent access: use stronger isolation such as
  separate databases, scoped database roles, or PostgreSQL row-level security.

Embeddings, SARIF findings, paths, symbols, and snippets can all reveal source
details. Treat database dumps and vector indexes as source-derived private data.
Remote embedding endpoints also receive source-derived text; use local
embeddings unless sending that text to the provider is acceptable.

## Database Configuration

The local Docker Compose database works without extra configuration. For an
external database, prefer one database URL:

```sh
PROJECT_CODE_INTELLIGENCE_DATABASE_URL=postgresql://user:password@host:5432/database?sslmode=prefer
```

If your MCP client or secret manager separates credentials, leave them out of
the URL and pass them separately:

```sh
PROJECT_CODE_INTELLIGENCE_DATABASE_URL=postgresql://host:5432/database?sslmode=prefer
PROJECT_CODE_INTELLIGENCE_DATABASE_USER=user
PROJECT_CODE_INTELLIGENCE_DATABASE_PASSWORD=password
```

Do not commit real database credentials. Put private values in user-local MCP
configuration, a local ignored file, or your system secret manager.

## Codex

Codex stores MCP config in `~/.codex/config.toml`, or in a project-scoped
`.codex/config.toml` for trusted projects. The CLI and IDE extension share this
configuration.

```toml
[mcp_servers.project-code-intelligence]
command = "/home/you/.local/bin/pci-mcp"
cwd = "/home/you/src/project-code-intelligence"
startup_timeout_sec = 20
tool_timeout_sec = 120

[mcp_servers.project-code-intelligence.env]
PROJECT_CODE_INTELLIGENCE_DATABASE_URL = "postgresql://host:5432/database?sslmode=prefer"
PROJECT_CODE_INTELLIGENCE_DATABASE_USER = "user"
PROJECT_CODE_INTELLIGENCE_DATABASE_PASSWORD = "password"
```

For a single-repo local setup using the Compose database, the environment block
can be omitted.

## Claude Code

Claude Code supports local, user, and project MCP scopes. Project-scoped MCP
servers are stored in `.mcp.json`; user/local scoped servers are private to the
user.

Use project-scoped `.mcp.json` only for non-secret shared configuration. Keep
credentials in local/user configuration or environment variables.

```json
{
  "mcpServers": {
    "project-code-intelligence": {
      "type": "stdio",
      "command": "/home/you/.local/bin/pci-mcp",
      "args": [],
      "cwd": "/home/you/src/project-code-intelligence",
      "env": {
        "PROJECT_CODE_INTELLIGENCE_DATABASE_URL": "postgresql://host:5432/database?sslmode=prefer",
        "PROJECT_CODE_INTELLIGENCE_DATABASE_USER": "user",
        "PROJECT_CODE_INTELLIGENCE_DATABASE_PASSWORD": "password"
      }
    }
  }
}
```

## OpenCode

OpenCode config is JSON or JSONC and defines MCP servers under `mcp`.

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
        "PROJECT_CODE_INTELLIGENCE_DATABASE_URL": "postgresql://host:5432/database?sslmode=prefer",
        "PROJECT_CODE_INTELLIGENCE_DATABASE_USER": "user",
        "PROJECT_CODE_INTELLIGENCE_DATABASE_PASSWORD": "password"
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
