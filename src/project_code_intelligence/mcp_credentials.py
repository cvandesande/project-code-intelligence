"""Private, project-scoped credentials used only by the MCP runtime."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import TYPE_CHECKING, cast

from project_code_intelligence import config

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_ENV_NAMES = (
    "PCI_MCP_DATABASE_URL",
    "PCI_MCP_DATABASE_USER",
    "PCI_MCP_DATABASE_PASSWORD",
    "PCI_COLLECTION",
    "PCI_DATABASE_SCOPE_PATH",
)


def credential_path(scope: Path) -> Path:
    root = config.user_config_dir()
    if root is None:
        raise config.ConfigError("Cannot store MCP credentials because XDG_CONFIG_HOME and HOME are unset.")
    identity = hashlib.sha256(str(scope.resolve()).encode()).hexdigest()[:16]
    return root / "mcp" / f"{identity}.json"


def write(scope: Path, values: Mapping[str, str]) -> Path:
    path = credential_path(scope)
    payload = {name: values[name] for name in _ENV_NAMES}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp_path = type(path)(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            _ = handle.write(json.dumps(payload, indent=2) + "\n")
        tmp_path.chmod(0o600)
        _ = tmp_path.replace(path)
        path.chmod(0o600)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def load(scope: Path) -> Path:
    path = credential_path(scope)
    if not path.is_file():
        raise config.ConfigError(
            f"No private MCP credentials exist for {scope.resolve()}. "
            "Run `pci index . --mcp-config <client>` from that project."
        )
    loaded = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(loaded, dict):
        raise config.ConfigError(f"Invalid MCP credential file: {path}")
    values = cast("dict[object, object]", loaded)
    for name in _ENV_NAMES:
        value = values.get(name)
        if not isinstance(value, str) or not value:
            raise config.ConfigError(f"Missing {name} in MCP credential file: {path}")
        os.environ[name] = value
    return path
