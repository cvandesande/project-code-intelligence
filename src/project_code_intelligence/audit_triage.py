"""Persistent, repository-local dispositions for ``pci audit`` candidates."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from project_code_intelligence.common import sha256_text
from project_code_intelligence.sarif.parse import json_object

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from project_code_intelligence.analyze import MotifGroup

TriageStatus = Literal["open", "dismissed"]
DEFAULT_TRIAGE_PATH = Path(".pci/audit-triage.json")
# Motif groups cannot realistically approach this bound; using it avoids
# treating the old `--limit 999` convention as proof of completeness.
FULL_TRIAGE_LIMIT = 2**31 - 1
TRIAGE_FILE_VERSION = 2
_SHARED_FILE_MODE = 0o644
_MIN_RECONCILE_OVERLAP = 2


@dataclass(frozen=True)
class Candidate:
    scope: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class TriageEntry:
    status: TriageStatus
    reason: str | None
    members: tuple[str, ...]
    scope: str | None = None
    updated_at: str | None = None


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _package_version() -> str:
    try:
        return version("project-code-intelligence")
    except PackageNotFoundError:
        return "unknown"


def candidate_members(group: MotifGroup, repo: str) -> tuple[str, ...]:
    return tuple(sorted(f"{member.symbol} {member.source_path.removeprefix(repo + '/')}" for member in group.members))


def candidate_id(group: MotifGroup, collection: str, repo: str) -> str:
    identity = f"{collection}/{repo}\n" + "\n".join(candidate_members(group, repo))
    return "redundancy-" + sha256_text(identity)[:12]


def load_triage(path: Path) -> dict[str, TriageEntry]:
    if not path.exists():
        return {}
    try:
        decoded = cast("object", json.loads(path.read_text(encoding="utf-8")))
        payload = json_object(decoded)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read audit triage file {path}: {exc}") from exc
    file_version = payload.get("version")
    if file_version not in {1, TRIAGE_FILE_VERSION}:
        raise ValueError(f"unsupported audit triage file version in {path}")
    raw_candidates = json_object(payload.get("candidates"))
    entries: dict[str, TriageEntry] = {}
    for key, raw_entry in raw_candidates.items():
        entry = json_object(raw_entry)
        status = entry.get("status")
        reason = entry.get("reason")
        members = entry.get("members")
        scope = entry.get("scope")
        updated_at = entry.get("updated_at")
        if status not in {"open", "dismissed"} or (reason is not None and not isinstance(reason, str)):
            raise ValueError(f"invalid audit triage entry {key!r} in {path}")
        if not isinstance(members, list) or not all(isinstance(member, str) for member in members):
            raise ValueError(f"invalid audit triage members for {key!r} in {path}")
        if (scope is not None and not isinstance(scope, str)) or (
            updated_at is not None and not isinstance(updated_at, str)
        ):
            raise ValueError(f"invalid audit triage provenance for {key!r} in {path}")
        entries[key] = TriageEntry(
            status=cast("TriageStatus", status),
            reason=reason,
            members=tuple(cast("list[str]", members)),
            scope=scope,
            updated_at=updated_at,
        )
    return entries


@contextmanager
def triage_lock(path: Path) -> Generator[None]:
    """Serialize read-modify-write cycles without leaving a lock file in the repo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(directory_fd, fcntl.LOCK_UN)
        os.close(directory_fd)


def write_triage(path: Path, entries: dict[str, TriageEntry]) -> None:
    payload = {
        "version": TRIAGE_FILE_VERSION,
        "updated_at": _now(),
        "generator": {"name": "pci audit", "version": _package_version()},
        "candidates": {
            key: {
                "status": entry.status,
                "reason": entry.reason,
                "members": list(entry.members),
                "scope": entry.scope,
                "updated_at": entry.updated_at,
            }
            for key, entry in sorted(entries.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            _ = stream.write("\n")
        Path(temporary).chmod(_SHARED_FILE_MODE)
        _ = Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def current_candidates(groups: Sequence[MotifGroup], collection: str, repo: str) -> dict[str, Candidate]:
    scope = f"{collection}/{repo}"
    return {
        candidate_id(group, collection, repo): Candidate(scope=scope, members=candidate_members(group, repo))
        for group in groups
    }


def make_entry(candidate: Candidate, status: TriageStatus, reason: str | None) -> TriageEntry:
    return TriageEntry(
        status=status,
        reason=reason,
        members=candidate.members,
        scope=candidate.scope,
        updated_at=_now(),
    )


def _overlap(left: Sequence[str], right: Sequence[str]) -> int:
    return len(set(left) & set(right))


def reconcile(current: dict[str, Candidate], entries: dict[str, TriageEntry]) -> dict[str, TriageEntry]:
    """Move dispositions across unique group expansions/contractions of at least two members."""
    reconciled = dict(entries)
    unmatched_current = {key: value for key, value in current.items() if key not in entries}
    unmatched_saved = {key: value for key, value in entries.items() if key not in current}
    possible: dict[str, list[str]] = {}
    reverse: dict[str, list[str]] = {}
    for current_id, candidate in unmatched_current.items():
        for saved_id, entry in unmatched_saved.items():
            if (
                entry.scope not in {None, candidate.scope}
                or _overlap(candidate.members, entry.members) < _MIN_RECONCILE_OVERLAP
            ):
                continue
            possible.setdefault(current_id, []).append(saved_id)
            reverse.setdefault(saved_id, []).append(current_id)
    for current_id, saved_ids in possible.items():
        if len(saved_ids) != 1 or len(reverse[saved_ids[0]]) != 1:
            continue
        saved_id = saved_ids[0]
        candidate = current[current_id]
        entry = reconciled.pop(saved_id)
        reconciled[current_id] = replace(entry, members=candidate.members, scope=candidate.scope, updated_at=_now())
    return reconciled


def summary(
    current: dict[str, Candidate], entries: dict[str, TriageEntry], *, infer_fixed: bool = False
) -> dict[str, list[tuple[str, TriageEntry]]]:
    result: dict[str, list[tuple[str, TriageEntry]]] = {"open": [], "dismissed": [], "fixed": []}
    for candidate_id_value, candidate in current.items():
        entry = entries.get(
            candidate_id_value,
            TriageEntry(status="open", reason=None, members=candidate.members, scope=candidate.scope),
        )
        result[entry.status].append((candidate_id_value, entry))
    if infer_fixed:
        for candidate_id_value, entry in entries.items():
            if candidate_id_value not in current:
                result["fixed"].append((candidate_id_value, entry))
    for items in result.values():
        items.sort(key=itemgetter(0))
    return result
