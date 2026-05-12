from __future__ import annotations

def make_conninfo(
    conninfo: str = "",
    *,
    host: str | None = None,
    port: str | None = None,
    dbname: str | None = None,
    user: str | None = None,
    password: str | None = None,
    sslmode: str | None = None,
) -> str: ...
