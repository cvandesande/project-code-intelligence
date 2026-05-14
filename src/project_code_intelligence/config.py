"""Typed configuration loaded from environment variables.

Environment variables remain the main automation interface for CI, containers,
and MCP clients. This module keeps parsing and validation in one place so the
rest of the code works with explicit values instead of repeatedly reading
``os.environ`` deep in execution paths.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from project_code_intelligence.exceptions import ConfigError

Env = Mapping[str, str]
DEFAULT_PGVECTOR_HOST = "127.0.0.1"
DEFAULT_PGVECTOR_PORT = "5433"
DEFAULT_PGVECTOR_DB = "codeintel"
DEFAULT_PGVECTOR_USER = "codeintel"
DEFAULT_PGVECTOR_PASS = DEFAULT_PGVECTOR_USER
DEFAULT_DB_CONNECT_TIMEOUT_SECONDS = 10
DEFAULT_DB_KEEPALIVES_IDLE_SECONDS = 30
DEFAULT_DB_KEEPALIVES_INTERVAL_SECONDS = 10
DEFAULT_DB_KEEPALIVES_COUNT = 3
DEFAULT_FASTEMBED_MODEL = "jinaai/jina-embeddings-v2-base-code"
DEFAULT_FASTEMBED_HOST = "127.0.0.1"
DEFAULT_FASTEMBED_PORT = 18081
DEFAULT_LOCAL_EMBEDDING_ENDPOINT = f"http://{DEFAULT_FASTEMBED_HOST}:{DEFAULT_FASTEMBED_PORT}/v1/embeddings"
DEFAULT_FASTEMBED_EMBEDDING_ENDPOINT = DEFAULT_LOCAL_EMBEDDING_ENDPOINT
DEFAULT_EMBEDDING_ENDPOINT_MODEL = "local"
DEFAULT_LEMONADE_EMBEDDING_ENDPOINT = DEFAULT_LOCAL_EMBEDDING_ENDPOINT
DEFAULT_LEMONADE_EMBEDDING_MODEL = "embed-gemma-300m-FLM"
DEFAULT_LOCAL_EMBEDDING_ENDPOINT_MODEL = DEFAULT_EMBEDDING_ENDPOINT_MODEL
DEFAULT_GPU_EMBEDDING_MODEL = "Qwen3-Embedding-0.6B-Q8_0.gguf"
DEFAULT_LARGE_GPU_EMBEDDING_MODEL = "Qwen3-Embedding-4B-Q8_0.gguf"
DEFAULT_APPLE_METAL_MODEL = "nomic-embed-code.Q8_0.gguf"
DEFAULT_MLX_MODEL = "mlx-community/Qwen3-Embedding-0.6B-8bit"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_VOYAGE_EMBEDDING_MODEL = "voyage-3.5"


def _env(env: Env | None) -> Env:
    return os.environ if env is None else env


def env_text(name: str, default: str | None = None, *, env: Env | None = None) -> str | None:
    value = _env(env).get(name)
    if not value:
        return default
    return value


def env_bool(name: str, *, default: bool = False, env: Env | None = None) -> bool:
    value = env_text(name, env=env)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean: 1/0, true/false, yes/no, or on/off")


def env_int(
    name: str,
    default: int,
    *,
    env: Env | None = None,
    minimum: int | None = None,
) -> int:
    value = env_text(name, env=env)
    if value is None:
        parsed = default
    else:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ConfigError(f"{name} must be an integer") from exc
    if minimum is not None and parsed < minimum:
        raise ConfigError(f"{name} must be greater than or equal to {minimum}")
    return parsed


def env_float(
    name: str,
    default: float,
    *,
    env: Env | None = None,
    minimum: float | None = None,
) -> float:
    value = env_text(name, env=env)
    if value is None:
        parsed = default
    else:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ConfigError(f"{name} must be a number") from exc
    if minimum is not None and parsed < minimum:
        raise ConfigError(f"{name} must be greater than or equal to {minimum:g}")
    return parsed


def default_embedding_endpoint(env: Env | None = None, *, local_default: bool = False) -> str | None:
    configured = env_text("PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT", env=env)
    if configured:
        return configured
    model = env_text("PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT_MODEL", env=env)
    if model and model.strip().lower().endswith("-flm"):
        return DEFAULT_LEMONADE_EMBEDDING_ENDPOINT
    if local_default:
        return DEFAULT_FASTEMBED_EMBEDDING_ENDPOINT
    return None


def default_embedding_endpoint_model(env: Env | None = None, *, endpoint: str | None = None) -> str:
    configured = env_text("PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT_MODEL", env=env)
    if configured:
        return configured
    if endpoint is None:
        endpoint = env_text("PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT", env=env)
    if endpoint and endpoint.rstrip("/") == DEFAULT_LOCAL_EMBEDDING_ENDPOINT:
        return DEFAULT_LOCAL_EMBEDDING_ENDPOINT_MODEL
    return DEFAULT_EMBEDDING_ENDPOINT_MODEL


def endpoint_hostname(endpoint: str | None) -> str | None:
    if not endpoint:
        return None
    return urlsplit(endpoint).hostname


def mask_database_dsn(dsn: str) -> str:
    parts = urlsplit(dsn)
    if not parts.scheme or not parts.netloc:
        return "database URL=<configured>"
    user = f"{parts.username}@" if parts.username else ""
    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = f":{parts.port}" if parts.port else ""
    except ValueError:
        port = ""
    query_parts: list[str] = []
    for item in parts.query.split("&"):
        key = item.split("=", 1)[0]
        query_parts.append(f"{key}=<hidden>" if key.lower() == "password" else item)
    query = "&".join(query_parts)
    return urlunsplit((parts.scheme, f"{user}{host}{port}", parts.path, query, ""))


def embedding_api_key(endpoint: str | None = None, *, env: Env | None = None) -> str | None:
    configured = env_text("PROJECT_CODE_INTELLIGENCE_EMBEDDING_API_KEY", env=env)
    if configured:
        return configured
    hostname = (endpoint_hostname(endpoint) or "").lower()
    if hostname == "api.openai.com":
        return env_text("OPENAI_API_KEY", env=env)
    if hostname == "api.voyageai.com":
        return env_text("VOYAGE_API_KEY", env=env)
    return None


@dataclass(frozen=True)
class DatabaseSettings:
    dsn: str | None = None
    dsn_source: str = "PROJECT_CODE_INTELLIGENCE_DATABASE_URL"
    dsn_user: str | None = None
    dsn_password: str | None = None
    host: str = DEFAULT_PGVECTOR_HOST
    port: str = DEFAULT_PGVECTOR_PORT
    dbname: str | None = DEFAULT_PGVECTOR_DB
    user: str | None = DEFAULT_PGVECTOR_USER
    password: str | None = DEFAULT_PGVECTOR_PASS
    sslmode: str = "prefer"
    connect_timeout_seconds: int = DEFAULT_DB_CONNECT_TIMEOUT_SECONDS
    keepalives_idle_seconds: int = DEFAULT_DB_KEEPALIVES_IDLE_SECONDS
    keepalives_interval_seconds: int = DEFAULT_DB_KEEPALIVES_INTERVAL_SECONDS
    keepalives_count: int = DEFAULT_DB_KEEPALIVES_COUNT
    allow_writes: bool = False

    @classmethod
    def from_env(cls, env: Env | None = None, *, role: str = "writer") -> DatabaseSettings:
        # MCP role: prefer MCP-specific env vars (separate read-only credentials);
        # fall back to the writer's settings so existing single-role deployments keep working.
        mcp = role == "mcp"
        database_url = (env_text("PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_URL", env=env) if mcp else None) or env_text(
            "PROJECT_CODE_INTELLIGENCE_DATABASE_URL", env=env
        )
        legacy_dsn = env_text("PGVECTOR_DSN", env=env)
        dsn = database_url or legacy_dsn
        dsn_user = (env_text("PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_USER", env=env) if mcp else None) or env_text(
            "PROJECT_CODE_INTELLIGENCE_DATABASE_USER", env=env
        )
        dsn_password = (
            env_text("PROJECT_CODE_INTELLIGENCE_MCP_DATABASE_PASSWORD", env=env) if mcp else None
        ) or env_text("PROJECT_CODE_INTELLIGENCE_DATABASE_PASSWORD", env=env)
        user = (env_text("PROJECT_CODE_INTELLIGENCE_MCP_PGVECTOR_USER", env=env) if mcp else None) or env_text(
            "PGVECTOR_USER", DEFAULT_PGVECTOR_USER, env=env
        )
        password = (env_text("PROJECT_CODE_INTELLIGENCE_MCP_PGVECTOR_PASS", env=env) if mcp else None) or env_text(
            "PGVECTOR_PASS", DEFAULT_PGVECTOR_PASS, env=env
        )
        return cls(
            dsn=dsn,
            dsn_source="PROJECT_CODE_INTELLIGENCE_DATABASE_URL" if database_url else "PGVECTOR_DSN",
            dsn_user=dsn_user,
            dsn_password=dsn_password,
            host=env_text("PGVECTOR_HOST", DEFAULT_PGVECTOR_HOST, env=env) or DEFAULT_PGVECTOR_HOST,
            port=env_text("PGVECTOR_PORT", DEFAULT_PGVECTOR_PORT, env=env) or DEFAULT_PGVECTOR_PORT,
            dbname=env_text("PGVECTOR_DB", DEFAULT_PGVECTOR_DB, env=env),
            user=user,
            password=password,
            sslmode=env_text("PGVECTOR_SSLMODE", "prefer", env=env) or "prefer",
            connect_timeout_seconds=env_int(
                "PROJECT_CODE_INTELLIGENCE_DB_CONNECT_TIMEOUT_SECONDS",
                DEFAULT_DB_CONNECT_TIMEOUT_SECONDS,
                env=env,
                minimum=1,
            ),
            keepalives_idle_seconds=env_int(
                "PROJECT_CODE_INTELLIGENCE_DB_KEEPALIVES_IDLE_SECONDS",
                DEFAULT_DB_KEEPALIVES_IDLE_SECONDS,
                env=env,
                minimum=1,
            ),
            keepalives_interval_seconds=env_int(
                "PROJECT_CODE_INTELLIGENCE_DB_KEEPALIVES_INTERVAL_SECONDS",
                DEFAULT_DB_KEEPALIVES_INTERVAL_SECONDS,
                env=env,
                minimum=1,
            ),
            keepalives_count=env_int(
                "PROJECT_CODE_INTELLIGENCE_DB_KEEPALIVES_COUNT",
                DEFAULT_DB_KEEPALIVES_COUNT,
                env=env,
                minimum=1,
            ),
            allow_writes=env_bool("PROJECT_CODE_INTELLIGENCE_ALLOW_WRITES", default=False, env=env),
        )

    def missing_connection_names(self) -> list[str]:
        if self.dsn:
            return []
        return [
            name
            for name, value in (
                ("PGVECTOR_DB", self.dbname),
                ("PGVECTOR_USER", self.user),
                ("PGVECTOR_PASS", self.password),
            )
            if not value
        ]

    def connection_hint(self) -> str:
        if self.dsn:
            extras: list[str] = []
            if self.dsn_user:
                extras.append("PROJECT_CODE_INTELLIGENCE_DATABASE_USER=<set>")
            if self.dsn_password:
                extras.append("PROJECT_CODE_INTELLIGENCE_DATABASE_PASSWORD=<set>")
            suffix = " " + " ".join(extras) if extras else ""
            return f"{self.dsn_source}=<hidden>{suffix}"
        return (
            f"PGVECTOR_HOST={self.host} "
            f"PGVECTOR_PORT={self.port} "
            f"PGVECTOR_DB={self.dbname or '<unset>'} "
            f"PGVECTOR_USER={self.user or '<unset>'} "
            f"PGVECTOR_PASS={'<set>' if self.password else '<unset>'}"
        )

    def display_target(self) -> str:
        if self.dsn:
            return mask_database_dsn(self.dsn)
        host = self.host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        target = f"postgresql://{self.user or '<unset>'}@{host}:{self.port}/{self.dbname or '<unset>'}"
        return f"{target} sslmode={self.sslmode}"


@dataclass(frozen=True)
class IngestSettings:
    collection: str | None = None
    profile: str = "generic"
    repos: str | None = None
    mode: str = "incremental"
    explicit_sarif: str | None = None
    sarif_max_bytes: int = 50 * 1024 * 1024
    embedding_endpoint: str | None = None
    embedding_endpoint_model: str = "local"
    embedding_max_chars: int = 3000
    embedding_min_chars: int = 800
    preembed: bool = True
    preembedding_ahead_batches: int = 16
    runtime_heartbeat_seconds: int = 300
    token_chars_per_token: float = 4.0

    @classmethod
    def from_env(cls, env: Env | None = None) -> IngestSettings:
        embedding_endpoint = default_embedding_endpoint(env=env)
        return cls(
            collection=env_text("PROJECT_CODE_INTELLIGENCE_COLLECTION", env=env),
            profile=env_text("PROJECT_CODE_INTELLIGENCE_PROFILE", "generic", env=env) or "generic",
            repos=env_text("PROJECT_CODE_INTELLIGENCE_REPOS", env=env),
            mode=env_text("PROJECT_CODE_INTELLIGENCE_MODE", "incremental", env=env) or "incremental",
            explicit_sarif=env_text("PROJECT_CODE_INTELLIGENCE_SARIF", env=env),
            sarif_max_bytes=env_int(
                "PROJECT_CODE_INTELLIGENCE_SARIF_MAX_BYTES",
                50 * 1024 * 1024,
                env=env,
                minimum=0,
            ),
            embedding_endpoint=embedding_endpoint,
            embedding_endpoint_model=default_embedding_endpoint_model(env=env, endpoint=embedding_endpoint),
            embedding_max_chars=env_int(
                "PROJECT_CODE_INTELLIGENCE_EMBEDDING_MAX_CHARS",
                3000,
                env=env,
                minimum=1,
            ),
            embedding_min_chars=env_int(
                "PROJECT_CODE_INTELLIGENCE_EMBEDDING_MIN_CHARS",
                800,
                env=env,
                minimum=200,
            ),
            preembed=env_bool("PROJECT_CODE_INTELLIGENCE_PREEMBED", default=True, env=env),
            preembedding_ahead_batches=env_int(
                "PROJECT_CODE_INTELLIGENCE_PREEMBED_AHEAD_BATCHES",
                16,
                env=env,
                minimum=1,
            ),
            runtime_heartbeat_seconds=env_int(
                "PROJECT_CODE_INTELLIGENCE_RUNTIME_HEARTBEAT_SECONDS",
                300,
                env=env,
                minimum=0,
            ),
            token_chars_per_token=env_float(
                "PROJECT_CODE_INTELLIGENCE_TOKEN_CHARS_PER_TOKEN",
                4.0,
                env=env,
                minimum=1.0,
            ),
        )


def configured_collection(env: Env | None = None) -> str | None:
    return env_text("PROJECT_CODE_INTELLIGENCE_COLLECTION", env=env)


def collection_override_allowed(env: Env | None = None) -> bool:
    return env_bool("PROJECT_CODE_INTELLIGENCE_ALLOW_COLLECTION_OVERRIDE", default=False, env=env)
