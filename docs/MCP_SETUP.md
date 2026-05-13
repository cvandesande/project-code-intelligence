# MCP Setup

`project-code-intelligence` exposes a stdio MCP server through `pci-mcp`.
The server reads from Postgres/pgvector and does not need to run in Docker.

Use Docker Compose for local dependencies such as Postgres and optional
embedding servers. Point your MCP client at the installed `pci-mcp` command.

## Recommended Scope

Use one database for one or more indexed repositories. Use:

- `PROJECT_CODE_INTELLIGENCE_COLLECTION` for a workspace or project family.
- MCP tool `repo` filters for individual repositories inside that collection.

For a single repository, collection can usually stay unset. For a workspace with
related repositories, set a collection during indexing and in the MCP server:

```sh
cd /path/to/workspace
PROJECT_CODE_INTELLIGENCE_COLLECTION=workspace-name pci-index repo-a repo-b repo-c
```

The MCP server can then be hard-scoped to `workspace-name`, while agents use
repo keys such as `repo-a` or `repo-b` in tool calls. Do not use absolute
filesystem paths as repo filters. Run `code_intel_status` without a repo filter
to see available collection and repo keys.

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
startup_timeout_sec = 20
tool_timeout_sec = 120

[mcp_servers.project-code-intelligence.env]
PROJECT_CODE_INTELLIGENCE_DATABASE_URL = "postgresql://host:5432/database?sslmode=prefer"
PROJECT_CODE_INTELLIGENCE_DATABASE_USER = "user"
PROJECT_CODE_INTELLIGENCE_DATABASE_PASSWORD = "password"
PROJECT_CODE_INTELLIGENCE_COLLECTION = "workspace-name"
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
      "env": {
        "PROJECT_CODE_INTELLIGENCE_DATABASE_URL": "postgresql://host:5432/database?sslmode=prefer",
        "PROJECT_CODE_INTELLIGENCE_DATABASE_USER": "user",
        "PROJECT_CODE_INTELLIGENCE_DATABASE_PASSWORD": "password",
        "PROJECT_CODE_INTELLIGENCE_COLLECTION": "workspace-name"
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
      "environment": {
        "PROJECT_CODE_INTELLIGENCE_DATABASE_URL": "postgresql://host:5432/database?sslmode=prefer",
        "PROJECT_CODE_INTELLIGENCE_DATABASE_USER": "user",
        "PROJECT_CODE_INTELLIGENCE_DATABASE_PASSWORD": "password",
        "PROJECT_CODE_INTELLIGENCE_COLLECTION": "workspace-name"
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
