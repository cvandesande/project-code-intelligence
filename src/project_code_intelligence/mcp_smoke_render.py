"""Rich rendering for the pci-mcp-smoke status response."""

from __future__ import annotations

import json
import operator
from typing import TYPE_CHECKING, cast

from rich.console import Group
from rich.table import Table
from rich.text import Text

from project_code_intelligence import console_ui
from project_code_intelligence.console_ui import as_dict as _as_dict
from project_code_intelligence.console_ui import as_list as _as_list

if TYPE_CHECKING:
    from rich.console import RenderableType

JsonDict = dict[str, object]


def _extract_payload_text(response: object) -> str | None:
    outer = _as_dict(response)
    if outer is None:
        return None
    result = _as_dict(outer.get("result"))
    if result is None:
        return None
    content = _as_list(result.get("content"))
    if not content:
        return None
    first = _as_dict(content[0])
    if first is None:
        return None
    text = first.get("text")
    return text if isinstance(text, str) else None


def _extract_payload(response: object) -> JsonDict | None:
    text = _extract_payload_text(response)
    if text is None:
        return None
    try:
        payload = cast("object", json.loads(text))
    except json.JSONDecodeError:
        return None
    return _as_dict(payload)


def _snapshot_for(payload: JsonDict, repo: str) -> JsonDict | None:
    snapshots = _as_list(payload.get("snapshots"))
    if snapshots is None:
        return None
    for snap in snapshots:
        snap_dict = _as_dict(snap)
        if snap_dict is not None and snap_dict.get("repo") == repo:
            return snap_dict
    return None


def _scalar_for(items: object, repo: str, key: str) -> int:
    item_list = _as_list(items)
    if item_list is None:
        return 0
    for item in item_list:
        entry = _as_dict(item)
        if entry is not None and entry.get("repo") == repo:
            value = entry.get(key)
            if isinstance(value, (int, float)):
                return int(value)
    return 0


def _records_by_type_for(payload: JsonDict, repo: str) -> list[tuple[str, int]]:
    items = _as_list(payload.get("records_by_type"))
    if items is None:
        return []
    result: list[tuple[str, int]] = []
    for item in items:
        entry = _as_dict(item)
        if entry is None or entry.get("repo") != repo:
            continue
        record_type = entry.get("record_type")
        count = entry.get("count")
        if isinstance(record_type, str) and isinstance(count, (int, float)):
            result.append((record_type, int(count)))
    result.sort(key=operator.itemgetter(1), reverse=True)
    return result


def _records_by_type_block(pairs: list[tuple[str, int]]) -> Group | None:
    if not pairs:
        return None
    sub = Table.grid(padding=(0, 1))
    sub.add_column(min_width=20, no_wrap=True)
    sub.add_column(overflow="fold")
    for record_type, count in pairs:
        sub.add_row(Text(f"  {record_type}", style="dim"), Text(console_ui.format_count(count)))
    return Group(Text("Records by type", style="bold"), sub)


def _resolve_status(payload: JsonDict, snapshot: JsonDict | None) -> tuple[console_ui.PillKind, str]:
    if not payload.get("schema_present"):
        return "fail", "SCHEMA MISSING"
    if snapshot is None:
        return "warn", "NO DATA"
    return "ok", "HEALTHY"


def _snapshot_line(snapshot: JsonDict) -> str:
    parts: list[str] = []
    snapshot_id = snapshot.get("id")
    if isinstance(snapshot_id, int):
        parts.append(f"#{snapshot_id}")
    commit_sha = snapshot.get("commit_sha")
    if isinstance(commit_sha, str):
        parts.append(console_ui.short_sha(commit_sha))
    branch = snapshot.get("branch")
    if isinstance(branch, str) and branch:
        parts.append(branch)
    if snapshot.get("dirty"):
        parts.append("dirty")
    return " · ".join(parts) if parts else "—"


def _add_schema_row(rows: Table, payload: JsonDict) -> None:
    schema_versions = _as_list(payload.get("schema_versions"))
    if schema_versions:
        console_ui.add_row(rows, "Schema", ", ".join(str(v) for v in schema_versions))
        return
    console_ui.add_row(rows, "Schema", "present" if payload.get("schema_present") else "missing")


def _add_files_row(rows: Table, payload: JsonDict, repo: str) -> None:
    files_count = _scalar_for(payload.get("files"), repo, "files")
    skipped_files = _scalar_for(payload.get("files"), repo, "skipped_files")
    if not (files_count or skipped_files):
        return
    detail = console_ui.format_count(files_count)
    if skipped_files:
        detail += f" · {console_ui.format_count(skipped_files)} skipped"
    console_ui.add_row(rows, "Files", detail)


def _add_records_row(rows: Table, payload: JsonDict, repo: str) -> None:
    records_count = _scalar_for(payload.get("records"), repo, "records")
    embedded_count = _scalar_for(payload.get("records"), repo, "embedded_records")
    if not (records_count or embedded_count):
        return
    detail = console_ui.format_count(records_count)
    if embedded_count and records_count > 0:
        pct = round(100 * embedded_count / records_count)
        detail += f" · {console_ui.format_count(embedded_count)} embedded ({pct}%)"
    console_ui.add_row(rows, "Records", detail)


def _add_edges_row(rows: Table, payload: JsonDict, repo: str) -> None:
    edges_count = _scalar_for(payload.get("edges"), repo, "edges")
    if edges_count:
        console_ui.add_row(rows, "Edges", console_ui.format_count(edges_count))


def _add_static_findings_row(rows: Table, payload: JsonDict, repo: str) -> None:
    static_findings = _as_list(payload.get("static_findings"))
    if static_findings is None:
        return
    finding_count = sum(
        1 for item in static_findings if (entry := _as_dict(item)) is not None and entry.get("repo") == repo
    )
    if finding_count:
        console_ui.add_row(rows, "Static findings", console_ui.format_count(finding_count))


def _status_rows(payload: JsonDict, repo: str, snapshot: JsonDict | None) -> Table:
    rows = console_ui.section_grid()
    _add_schema_row(rows, payload)
    if snapshot is not None:
        console_ui.add_row(rows, "Snapshot", _snapshot_line(snapshot))
    _add_files_row(rows, payload, repo)
    _add_records_row(rows, payload, repo)
    _add_edges_row(rows, payload, repo)
    _add_static_findings_row(rows, payload, repo)
    return rows


_PROBE_RESULT_KEYS: dict[str, str] = {
    "list_code_intel_files": "files",
    "search_code_intel_text": "results",
    "search_code_intel_semantic": "results",
    "related_code_intel": "edges",
}


def _probes_block(probes: list[dict[str, object]] | None) -> Group | None:
    if not probes:
        return None
    sub = Table.grid(padding=(0, 1))
    sub.add_column(width=1, no_wrap=True)
    sub.add_column(min_width=32, no_wrap=True)
    sub.add_column(overflow="fold")
    for probe in probes:
        tool = str(probe.get("tool", "?"))
        repo = probe.get("repo")
        label = f"{repo}: {tool}" if isinstance(repo, str) else tool
        status = str(probe.get("status", "?"))
        glyph_kind: console_ui.PillKind = "ok" if status == "ok" else "fail"
        glyph = Text(console_ui.STATUS_GLYPHS[glyph_kind], style={"ok": "green", "fail": "red"}[glyph_kind])
        detail = _probe_detail(probe, tool)
        sub.add_row(glyph, Text(label, style="dim"), Text(detail))
    return Group(Text("MCP tool probes", style="bold"), sub)


def _probe_detail(probe: dict[str, object], tool: str) -> str:
    if probe.get("status") != "ok":
        error = probe.get("error")
        if isinstance(error, str):
            return error
        match probe.get("response"):
            case {"error": str() as inner}:
                return inner
            case _:
                pass
        return "failed"
    payload = _extract_payload(probe.get("response"))
    if payload is None:
        return "no payload"
    result_key = _PROBE_RESULT_KEYS.get(tool)
    if result_key is None:
        return "ok"
    items = _as_list(payload.get(result_key))
    count = len(items) if items is not None else 0
    return f"{count} item(s)"


def render_status(response: object, *, repo: str, probes: list[dict[str, object]] | None = None) -> None:
    console = console_ui.build_console()
    payload = _extract_payload(response)
    if payload is None:
        console.print(console_ui.main_panel(Text("Could not parse MCP response payload.", style="red")))
        return

    snapshot = _snapshot_for(payload, repo)
    kind, label = _resolve_status(payload, snapshot)
    body: list[RenderableType] = [
        console_ui.header_row(f"pci-mcp {repo}", kind, label),
        Text(),
        _status_rows(payload, repo, snapshot),
    ]
    by_type = _records_by_type_block(_records_by_type_for(payload, repo))
    if by_type is not None:
        body.extend((Text(), by_type))
    probes_block = _probes_block(probes)
    if probes_block is not None:
        body.extend((Text(), probes_block))
    console.print(console_ui.main_panel(Group(*body)))


def render_error(response: object) -> None:
    console = console_ui.build_console()
    message: str = "MCP server returned an error"
    match response:
        case {"error": {"message": str() as outer_message}}:
            message = outer_message
        case _:
            pass
    payload = _extract_payload(response)
    if payload is not None and isinstance(payload.get("error"), str):
        message = str(payload["error"])
    body = Group(console_ui.header_row("pci-mcp", "fail", "ERROR"), Text(), Text(message, style="red"))
    console.print(console_ui.main_panel(body))
