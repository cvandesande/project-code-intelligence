#!/usr/bin/env python3
"""End-to-end smoke test against a running pgvector database."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import NoReturn, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))

from project_code_intelligence import process  # noqa: E402


class SmokeNamespace(argparse.Namespace):
    collection: str
    keep_fixture: bool


def fail(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def write_stdout(message: str) -> None:
    _ = sys.stdout.write(message + "\n")


def write_stderr(message: str) -> None:
    _ = sys.stderr.write(message + "\n")


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    timeout: int = 120,
) -> process.CompletedProcess[str]:
    proc = process.run(
        command,
        process.RunOptions(
            cwd=cwd,
            env=env,
            input_text=input_text,
            capture_output=True,
            timeout=timeout,
            check=False,
        ),
    )
    if proc.returncode != 0:
        if proc.stdout:
            _ = sys.stdout.write(proc.stdout)
        if proc.stderr:
            _ = sys.stderr.write(proc.stderr)
        fail(f"{command[0]} exited with {proc.returncode}")
    return proc


def write_fixture_repo(path: Path) -> None:
    _ = run(["git", "init", "-q"], cwd=path)
    _ = run(["git", "config", "user.email", "smoke@example.invalid"], cwd=path)
    _ = run(["git", "config", "user.name", "Project Code Intelligence Smoke"], cwd=path)
    _ = (path / "README.md").write_text(
        "# Smoke Fixture\n\nTiny repository for integration testing.\n",
        encoding="utf-8",
    )
    _ = (path / "demo.py").write_text(
        "\n".join([
            "def add(left: int, right: int) -> int:",
            "    return left + right",
            "",
            "",
            "class Greeter:",
            "    def hello(self, name: str) -> str:",
            "        return f'hello {name}'",
            "",
        ]),
        encoding="utf-8",
    )
    _ = (path / "pyproject.toml").write_text(
        '[project]\nname = "smoke-fixture"\nversion = "0.0.0"\n',
        encoding="utf-8",
    )
    _ = run(["git", "add", "."], cwd=path)
    _ = run(["git", "commit", "-q", "-m", "Initial smoke fixture"], cwd=path)


def smoke_env(collection: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PROJECT_CODE_INTELLIGENCE_COLLECTION"] = collection
    env["PROJECT_CODE_INTELLIGENCE_MODE"] = "full"
    env["PROJECT_CODE_INTELLIGENCE_PROFILE"] = "generic"
    env["PROJECT_CODE_INTELLIGENCE_REPOS"] = "."
    return env


def parse_mcp_tool_response(stdout: str, tool_name: str) -> dict[str, object]:
    try:
        rpc_response_value = cast("object", json.loads(stdout))
        if not isinstance(rpc_response_value, dict):
            fail("MCP smoke response was not an object")
        rpc_response = cast("dict[str, object]", rpc_response_value)
        if "error" in rpc_response:
            fail(f"{tool_name} returned an MCP error: {rpc_response['error']}")
        result_value = rpc_response["result"]
        if not isinstance(result_value, dict):
            fail("MCP result was not an object")
        result = cast("dict[str, object]", result_value)
        content_value = result["content"]
        if not isinstance(content_value, list) or not content_value or not isinstance(content_value[0], dict):
            fail("MCP result content was not a non-empty list")
        content_item = cast("dict[str, object]", content_value[0])
        text = content_item["text"]
        if not isinstance(text, str):
            fail(f"{tool_name} result text was not a string")
        tool_value = cast("object", json.loads(text))
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        fail(f"unexpected {tool_name} response: {exc}")
    if not isinstance(tool_value, dict):
        fail(f"{tool_name} response was not an object")
    tool_object = cast("dict[object, object]", tool_value)
    return {str(key): value for key, value in tool_object.items()}


def call_mcp_tool(tool_name: str, arguments: dict[str, object], *, cwd: Path, env: dict[str, str]) -> dict[str, object]:
    request: dict[str, object] = {
        "jsonrpc": "2.0",
        "id": tool_name,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    proc = run(
        [sys.executable, "-m", "project_code_intelligence.server"],
        cwd=cwd,
        env=env,
        input_text=json.dumps(request) + "\n",
        timeout=30,
    )
    return parse_mcp_tool_response(proc.stdout, tool_name)


def result_mentions(results: object, text: str) -> bool:
    if not isinstance(results, list):
        return False
    rows = cast("list[object]", results)
    needle = text.lower()
    for row in rows:
        if isinstance(row, dict):
            row_values = cast("dict[object, object]", row).values()
            haystack = " ".join(str(value) for value in row_values).lower()
            if needle in haystack:
                return True
    return False


def remove_tree(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


def run_ingest_checks(fixture_dir: Path, env: dict[str, str]) -> None:
    dry_run = run(
        [sys.executable, "-m", "project_code_intelligence.cli", "--no-embed", "--dry-run"],
        cwd=fixture_dir,
        env=env,
    )
    if '"files"' not in dry_run.stdout:
        fail("dry-run output did not look like an ingest report")
    ingest = run([sys.executable, "-m", "project_code_intelligence.cli", "--no-embed"], cwd=fixture_dir, env=env)
    if '"snapshot_ids"' not in ingest.stdout:
        fail("ingest output did not include snapshot IDs")
    incremental_env = dict(env)
    incremental_env["PROJECT_CODE_INTELLIGENCE_MODE"] = "incremental"
    incremental = run(
        [sys.executable, "-m", "project_code_intelligence.cli", "--no-embed"],
        cwd=fixture_dir,
        env=incremental_env,
    )
    if '"mode": "incremental"' not in incremental.stdout:
        fail("incremental ingest did not run in incremental mode")


def run_mcp_checks(fixture_dir: Path, env: dict[str, str]) -> tuple[int, int]:
    status_proc = run([sys.executable, "-m", "project_code_intelligence.cli", "mcp-smoke"], cwd=fixture_dir, env=env)
    status = parse_mcp_tool_response(status_proc.stdout, "code_intel_status")
    if status.get("schema_present") is not True:
        fail("MCP status did not report an initialized schema")
    snapshots = status.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        fail("MCP status did not report the smoke snapshot")
    search = call_mcp_tool("search_code_intel_text", {"query": "Greeter", "limit": 5}, cwd=fixture_dir, env=env)
    search_results = search.get("results")
    if not isinstance(search_results, list):
        fail("MCP text search did not return the fixture symbol")
    search_results_list = cast("list[object]", search_results)
    if not result_mentions(search_results_list, "Greeter"):
        fail("MCP text search did not return the fixture symbol")
    return len(cast("list[object]", snapshots)), len(search_results_list)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--collection",
        default=f"integration-smoke-{os.getpid()}",
        help="Collection name to write and query.",
    )
    _ = parser.add_argument(
        "--keep-fixture", action="store_true", help="Leave the temporary fixture repository on disk."
    )
    args = parser.parse_args(namespace=SmokeNamespace())

    fixture_dir = Path(tempfile.mkdtemp(prefix="pci-smoke-"))
    try:
        write_fixture_repo(fixture_dir)
        env = smoke_env(str(args.collection))
        run_ingest_checks(fixture_dir, env)
        snapshots_count, search_results_count = run_mcp_checks(fixture_dir, env)
        write_stdout(
            json.dumps({
                "collection": args.collection,
                "fixture": str(fixture_dir),
                "snapshots": snapshots_count,
                "search_results": search_results_count,
            })
        )
    finally:
        if args.keep_fixture:
            write_stderr(f"kept fixture repo: {fixture_dir}")
        else:
            remove_tree(fixture_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
