# MCP Setup

`project-code-intelligence` exposes a stdio MCP server through `pci mcp`.
The server reads from Postgres/pgvector and does not need to run in Docker.

Use Docker Compose for local dependencies such as Postgres and optional
embedding servers. Point your MCP client at the installed `pci` command with the `mcp` argument.

## Recommended Scope

Use one local database per indexed repository or workspace. Use:

- the inferred database for that repository or workspace.
- a collection inside that database for a workspace or project family.
- MCP tool `repo` filters for individual repositories inside that collection.

For normal use, you do not need to set a collection environment variable.
`pci index` infers the database and collection from the paths you pass:

- one repo path: database scope and collection use the repo directory
- multiple repo paths: database scope and collection use the current working
  directory; pass repo subdirectories from that workspace

For a workspace with related repositories:

```sh
cd /path/to/workspace
pci index repo-a repo-b repo-c
```

The MCP server also infers its collection from its configured working directory
when `PCI_COLLECTION` is unset. Set the MCP `cwd` to the
same repo or workspace directory you used when indexing. Agents then use repo
keys such as `repo-a` or `repo-b` in tool calls. Do not use absolute filesystem
paths as repo filters. Run `code_intel_status` without a repo filter to see
available collection and repo keys.

Use explicit collection configuration only when you want a collection name
different from the inferred directory name:

```sh
pci index --collection workspace-name repo-a repo-b repo-c
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
PCI_DATABASE_URL=postgresql://user:password@host:5432/database?sslmode=prefer
```

If the URL omits the database path, host, port, credentials, and query options
come from the URL, but the database name is inferred from the repo/workspace:

```sh
PCI_DATABASE_URL=postgresql://user:password@host:5432?sslmode=prefer
```

For first-use bootstrap, use PostgreSQL admin credentials once with
`pci doctor --init-postgres`. Doctor creates/updates the cluster-level
`pci_index_admin` role, installs pgvector into `template1`, writes the
non-superuser `pci index` credentials to
`${XDG_CONFIG_HOME:-~/.config}/project-code-intelligence/pci-index.env` with
`0600` permissions, and prints the same
`PCI_DATABASE_ADMIN_*` exports for copy/paste or secret
manager use. It does not create a project database. Then `pci index` loads the
user config when those variables are unset and creates the inferred project
database and schema before indexing. Use `pci index --init-db` when you want to
initialize the database/schema and exit without scanning:

```sh
PCI_DATABASE_URL=postgresql://host:5432?sslmode=prefer
PCI_POSTGRES_ADMIN_USER=postgres
PCI_POSTGRES_ADMIN_PASSWORD=admin-password
pci doctor --init-postgres

pci index --init-db .
pci index .
```

For the bundled local database, no Postgres admin bootstrap is needed:

```sh
pci doctor --start-db
pci doctor
```

When `pci doctor` cannot reach a database, fix the database first by setting
`PCI_DATABASE_URL`, running `pci doctor --start-db`, or running
`pci doctor --init-postgres` for a non-bundled Postgres cluster. Indexing only
makes sense after the database is reachable.

Use `pci doctor --init-postgres --no-write-config` when you want only printed
exports and no user config file.

When admin variables are set for an inferred database, generated scoped roles
override credentials embedded in `PCI_DATABASE_URL`.
Set `PCI_DATABASE_USER` and
`PCI_DATABASE_PASSWORD` only when you intentionally want
to force explicit runtime credentials.

For generic MCP clients or custom launchers that cannot set the server working
directory, set the same base URL and scope path so `pci mcp` infers the same
project database as `pci index`:

```sh
PCI_MCP_DATABASE_URL=postgresql://host:5432?sslmode=prefer
PCI_DATABASE_SCOPE_PATH=/path/to/repo-or-workspace
PCI_MCP_DATABASE_USER=project_ro
PCI_MCP_DATABASE_PASSWORD=password
```

To create private read-only MCP credentials and print a credential-free client
configuration, use `--mcp-config`:

```sh
pci index --init-db --mcp-config codex .
pci index --init-db --mcp-config claude .
pci index --init-db --mcp-config opencode .
pci index --init-db --mcp-config vscode .
pci index --init-db --mcp-config cline .
pci index --init-db --mcp-config zed .
```

Generated client config uses the generic server key
`project-code-intelligence`. That key is intentionally reused when the config
belongs to the indexed repo/workspace. Do not paste project-scoped snippets
into global MCP config. Every supported harness launches
`pci mcp --scope /absolute/project/path`. PCI resolves that scope to a
project-keyed credential file under the user's PCI config directory, stored
with mode `0600`; no database URL, username, or password is written to the
repository or harness config.

Install or update a generated config with `pci mcp install --target TARGET`.
Use `--uninstall` to remove only PCI's server entry while preserving unrelated
client settings. Supported targets are `codex`, `claude`, `opencode`, `pi`, `vscode`,
`copilot`, `cline`, and `zed`; Cline also needs `--config-path` for
its user-scoped settings file.

If those MCP-specific variables are unset, `pci mcp` falls back to the generic
database variables.

The one-time bootstrap user usually needs local Postgres superuser privileges
because pgvector is not trusted. `pci doctor --init-postgres` creates a
non-superuser `pci_index_admin` role with `CREATEDB` and `CREATEROLE`; new
project databases inherit pgvector from `template1`. If a project database was
created before `template1` had pgvector, reset that inferred database and index
again.

If admin variables are not set, `pci index` promotes the writer credentials to
the effective admin and still creates per-project RW/RO roles using
deterministic HMAC-derived passwords. The bundled local pgvector container
ships with a writer that has `CREATEROLE`, so this works without setup; for a
remote Postgres whose writer lacks `CREATEROLE` the role-creation SQL fails
with guidance to run `pci doctor --init-postgres`.

If your MCP client or secret manager separates credentials, leave them out of
the URL and pass them separately:

```sh
PCI_DATABASE_URL=postgresql://host:5432?sslmode=prefer
PCI_DATABASE_USER=user
PCI_DATABASE_PASSWORD=password
```

Set `PCI_PG_DB` only when you want to disable database-name inference and use
a fixed database name.

`pci index --reset .` drops only the inferred PCI-managed database for that
repo/workspace scope. Use `pci doctor --clean` for broad local runtime cleanup
while keeping the CLI installed. Use `make tool-uninstall` to run that cleanup
and then remove the installed `pci` binary.

Installed packages use bundled Compose assets. If you need to customize the
Compose file, copy `docker-compose.yml` and point PCI at that copy:

```sh
PCI_COMPOSE_FILE=/path/to/docker-compose.yml pci doctor --start-db
```

Do not edit files inside a Nix store path or an installed wheel. Re-running
`uv tool install --reinstall` or `make tool-install` replaces the installed
package, but `PCI_COMPOSE_FILE` keeps your Compose customization outside the
package lifecycle.

Do not commit real database credentials or private export files. Generated
project-scoped snippets for every supported harness avoid embedding credentials
and are suitable to share only when the repo path and server command are also
acceptable to share. PCI stores the per-project read-only values outside the
repository in its private user config directory with mode `0600`. The
`pci doctor --init-postgres` user config contains non-superuser credentials for
`pci index`; keep it private and at `0600`.

## Codex

`pci index --mcp-config codex` emits config for the indexed repo's
`.codex/config.toml`.

Generate the snippet from the indexed repo/workspace:

```sh
pci index --init-db --mcp-config codex .
```

The command prints the project target path before the TOML:

```text
Codex project-scoped MCP config
Write this snippet to: /home/you/src/project-code-intelligence/.codex/config.toml
This snippet contains no database credentials; pci mcp loads them from private user config.
Do not paste this into a global MCP config; the server key is intentionally reused per project.
```

```toml
[mcp_servers.project-code-intelligence]
command = "/home/you/.local/bin/pci"
args = ["mcp", "--scope", "/home/you/src/project-code-intelligence"]
cwd = "/home/you/src/project-code-intelligence"
startup_timeout_sec = 20
tool_timeout_sec = 120
```

Run `pci mcp install --target codex` to merge this entry automatically.

## Claude Code

`pci index --mcp-config claude` emits config for the indexed repo's
project-scoped `.mcp.json`.

Generate the snippet from the indexed repo/workspace:

```sh
pci index --init-db --mcp-config claude .
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
      "command": "/home/you/.local/bin/pci",
      "args": ["mcp"],
      "cwd": "/home/you/src/project-code-intelligence",
      "env": {
        "PCI_MCP_DATABASE_URL": "${PCI_MCP_DATABASE_URL}",
        "PCI_MCP_DATABASE_USER": "${PCI_MCP_DATABASE_USER}",
        "PCI_MCP_DATABASE_PASSWORD": "${PCI_MCP_DATABASE_PASSWORD}"
      }
    }
  }
}
```

Use the export block printed after the JSON before launching Claude Code.

## OpenCode

`pci index --mcp-config opencode` emits config for the indexed repo's
`opencode.json`. OpenCode config is JSON or JSONC and defines MCP servers under
`mcp`.

For contributors working inside this repo, the tracked `opencode.example.json`
already wires the local server; copy it to the gitignored `opencode.json` to
enable it (this repo ships an active local `opencode.json` for maintainers):

```sh
cp opencode.example.json opencode.json
```

Or generate the snippet from any indexed repo/workspace:

```sh
pci index --init-db --mcp-config opencode .
```

The generated `opencode.json` uses OpenCode's `{env:VAR}` substitution, so
credentials stay in the accompanying export block or your secret manager.

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "project-code-intelligence": {
      "type": "local",
      "command": ["/home/you/.local/bin/pci", "mcp"],
      "enabled": true,
      "cwd": "/home/you/src/project-code-intelligence",
      "environment": {
        "PCI_MCP_DATABASE_URL": "{env:PCI_MCP_DATABASE_URL}",
        "PCI_MCP_DATABASE_USER": "{env:PCI_MCP_DATABASE_USER}",
        "PCI_MCP_DATABASE_PASSWORD": "{env:PCI_MCP_DATABASE_PASSWORD}"
      }
    }
  }
}
```

Use the export block printed after the JSON before launching OpenCode.

## VS Code Copilot

`pci index --mcp-config vscode` emits config for the indexed repo's
`.vscode/mcp.json`. `--mcp-config copilot` is an alias for the same output.

Generate the snippet from the indexed repo/workspace:

```sh
pci index --init-db --mcp-config vscode .
```

VS Code stores MCP servers under top-level `servers` in `.vscode/mcp.json`.
The generated config uses VS Code's `${env:VAR}` substitution, so credentials
stay in the accompanying export block or your secret manager.

```json
{
  "servers": {
    "project-code-intelligence": {
      "type": "stdio",
      "command": "/home/you/.local/bin/pci",
      "args": ["mcp"],
      "env": {
        "PCI_MCP_DATABASE_URL": "${env:PCI_MCP_DATABASE_URL}",
        "PCI_MCP_DATABASE_USER": "${env:PCI_MCP_DATABASE_USER}",
        "PCI_MCP_DATABASE_PASSWORD": "${env:PCI_MCP_DATABASE_PASSWORD}",
        "PCI_COLLECTION": "${env:PCI_COLLECTION}",
        "PCI_DATABASE_SCOPE_PATH": "${env:PCI_DATABASE_SCOPE_PATH}"
      }
    }
  }
}
```

Use the export block printed after the JSON before launching VS Code. It
includes `PCI_COLLECTION` because VS Code's documented
stdio MCP config does not define a server `cwd` field.

## Zed

`pci index --mcp-config zed` emits JSON for Zed project settings. Add or merge
the generated snippet into `.zed/settings.json` in the indexed repo/workspace.

Generate the snippet from the indexed repo/workspace:

```sh
pci index --init-db --mcp-config zed .
```

The generated Zed snippet contains the read-only database values directly
because Zed does not document environment-variable interpolation for MCP `env`
values. Keep `.zed/settings.json` local and do not commit it. The snippet also
sets explicit collection and scope values because Zed's MCP server config does
not include a server `cwd`. After adding project settings, trust the worktree in
Zed so project settings can start MCP servers.

Write or merge this into `.zed/settings.json`:

```json
{
  "context_servers": {
    "project-code-intelligence": {
      "command": "/home/you/.local/bin/pci",
      "args": ["mcp"],
      "env": {
        "PCI_MCP_DATABASE_URL": "postgresql://host:5432/project_db?sslmode=prefer",
        "PCI_MCP_DATABASE_USER": "project_ro",
        "PCI_MCP_DATABASE_PASSWORD": "password",
        "PCI_COLLECTION": "project-code-intelligence",
        "PCI_DATABASE_SCOPE_PATH": "/home/you/src/project-code-intelligence"
      }
    }
  }
}
```

## Cline (VS Code)

`pci index --mcp-config cline` emits a JSON snippet for Cline's VS Code
extension MCP settings. Add or merge the generated entry under `mcpServers` in
Cline's MCP settings JSON:

1. Open the MCP Servers icon in the Cline panel.
2. Open the Configure tab.
3. Click Configure MCP Servers.

Generate the snippet from the indexed repo/workspace:

```sh
pci index --init-db --mcp-config cline .
```

Cline's VS Code MCP settings are user-scoped, not repository-scoped. The
generated JSON contains the read-only database values directly because Cline's
documented VS Code config shape does not describe VS Code-style environment
substitution for `env` values. Keep this settings file local and do not commit
it. Use `--mcp-server-name NAME` if you need distinct keys for multiple
projects.

```json
{
  "mcpServers": {
    "project-code-intelligence": {
      "command": "/home/you/.local/bin/pci",
      "args": ["mcp"],
      "env": {
        "PCI_MCP_DATABASE_URL": "postgresql://host:5432/project_db?sslmode=prefer",
        "PCI_MCP_DATABASE_USER": "project_ro",
        "PCI_MCP_DATABASE_PASSWORD": "password",
        "PCI_COLLECTION": "project-code-intelligence",
        "PCI_DATABASE_SCOPE_PATH": "/home/you/src/project-code-intelligence"
      },
      "autoApprove": [],
      "disabled": false
    }
  }
}
```

## Pi

Pi has no built-in MCP client, so PCI installs a project-local extension that
launches the stdio server and registers its discovered tools dynamically:

```sh
pci mcp install --target pi
pci hook install --target pi
```

The commands write separate managed extensions under `.pi/extensions/`.
Restart Pi and trust the project to load them. Use the corresponding
`--uninstall` command to remove only PCI's extension. The hook extension adds
the session banner and runs edit evidence for Pi's `edit` and `write` tools.

## Other MCP Clients

`pci index --mcp-config` intentionally emits project-scoped config only for
Codex, Claude Code, OpenCode, VS Code Copilot, and Zed, plus user-scoped Cline
config for the VS Code extension. For other clients, use the `env` export block
with that client's project/workspace configuration if it has one.

## Agent Guidance

For repositories that use this MCP server heavily, drop
[`docs/SYSTEM_PROMPT.md`](SYSTEM_PROMPT.md) into the connected agent's
system-prompt slot. The design notes behind it are in
[`docs/SYSTEM_PROMPT_RATIONALE.md`](SYSTEM_PROMPT_RATIONALE.md).

The important behavior for assistants is:

1. Call `code_intel_status` first.
2. Use the collection and repo keys reported by `code_intel_status`.
3. For implementation questions, pass `file_role: source` when you want to
   exclude tests and docs rather than just rank source higher.
4. Use repo filters such as `service-api`, `web-ui`, or `shared-lib`, not absolute paths.
5. Treat the MCP index as a navigation aid and verify important behavior
   against the working tree.
