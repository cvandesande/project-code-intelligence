"""Shared helpers for native environment diagnostics."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from project_code_intelligence import config, db
from project_code_intelligence.doctor.types import CheckResult, Status

if TYPE_CHECKING:
    from collections.abc import Sequence

BYTES_PER_KIB = 1024.0


def result(name: str, status: Status, message: str, detail: str | None = None) -> CheckResult:
    return CheckResult(name=name, status=status, message=message, detail=detail)


def row_object(row: db.DbRow, key: str) -> object:
    return row[key]


def row_text(row: db.DbRow, key: str) -> str:
    value = row_object(row, key)
    return "" if value is None else str(value)


def row_bool(row: db.DbRow, key: str) -> bool:
    value = row_object(row, key)
    return bool(value)


def bytes_from_text(text: str | None) -> int | None:
    if not text:
        return None
    try:
        value = int(text.strip())
    except ValueError:
        return None
    return value if value >= 0 else None


def human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    scaled = float(value)
    unit = units[0]
    for unit in units:
        if abs(scaled) < BYTES_PER_KIB or unit == units[-1]:
            break
        scaled /= BYTES_PER_KIB
    if unit == "B":
        return f"{value} B"
    return f"{scaled:.1f} {unit}"


def bool_from_env(name: str, *, default: bool, env: config.Env) -> tuple[bool, CheckResult | None]:
    try:
        return config.env_bool(name, default=default, env=env), None
    except ValueError as exc:
        return default, result("configuration", "fail", str(exc))


def table_exists(conn: db.DbConnection, table: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s) IS NOT NULL AS exists", [f"public.{table}"]).fetchone()
    if row is None:
        return False
    return row_bool(row, "exists")


def version_tuple(text: str) -> tuple[int, ...]:
    match = re.search(r"\d+(?:\.\d+)*", text)
    if not match:
        return ()
    return tuple(int(part) for part in match.group(0).split("."))


def version_at_least(value: Sequence[int], minimum: Sequence[int]) -> bool:
    width = max(len(value), len(minimum))
    padded_value = tuple(value) + (0,) * (width - len(value))
    padded_minimum = tuple(minimum) + (0,) * (width - len(minimum))
    return padded_value >= padded_minimum


def status_for_requirement(*, ok: bool, required: bool) -> Status:
    if ok:
        return "ok"
    return "fail" if required else "warn"
