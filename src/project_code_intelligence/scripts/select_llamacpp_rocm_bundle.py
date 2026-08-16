#!/usr/bin/env python3
"""Select the Lemonade llama.cpp ROCm release bundle for the local AMD GPU."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import urllib.error
from pathlib import Path
from typing import NoReturn, TypedDict, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project_code_intelligence import http_client, process
from project_code_intelligence.rocm_bundles import (
    DEFAULT_LLAMACPP_ROCM_REPO,
    RocmBundleError,
    bundle_for_gfx_target,
    gfx_target_from_pci_ids,
    gfx_targets_from_text,
    llama_rocm_asset_name,
    llama_rocm_download_url,
    normalize_gfx_target,
    normalize_github_repo,
    normalize_rocm_bundle,
)


class GitHubRelease(TypedDict):
    tag_name: str
    assets: list[dict[str, str]]


class Selection(TypedDict):
    gfx_target: str
    bundle: str
    release: str
    asset: str
    url: str
    source: str


class RocmNamespace(argparse.Namespace):
    repo: str
    release: str
    format: str


def fail(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def write_stdout(message: str) -> None:
    _ = sys.stdout.write(message + "\n")


def command_output(command: list[str]) -> str:
    try:
        proc = process.run(
            command,
            process.RunOptions(
                check=False,
                stdout=process.PIPE,
                stderr=process.STDOUT,
                timeout=10,
            ),
        )
    except (OSError, process.TimeoutExpired):
        return ""
    return proc.stdout


def read_sysfs_value(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def sysfs_gfx_targets(sysfs_root: Path = Path("/sys/class/drm")) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    for device_path in sorted(sysfs_root.glob("card*/device")):
        target = gfx_target_from_pci_ids(
            read_sysfs_value(device_path / "vendor"),
            read_sysfs_value(device_path / "device"),
        )
        if target and target not in seen:
            seen.add(target)
            targets.append(target)
    return targets


def detected_gfx_targets() -> tuple[list[str], str]:
    override = os.environ.get("PCI_AMD_GFX")
    if override:
        return [normalize_gfx_target(override)], "PCI_AMD_GFX"

    sysfs_targets = sysfs_gfx_targets()
    if sysfs_targets:
        return sysfs_targets, "/sys/class/drm"

    for command in (["rocminfo"], ["rocm-smi"], ["amd-smi", "static"], ["lspci", "-nn"]):
        output = command_output(command)
        targets = gfx_targets_from_text(output)
        if targets:
            return targets, command[0]
    return [], "auto-detect"


def selected_bundle() -> tuple[str, str, str]:
    bundle_override = os.environ.get("PCI_LLAMA_CPP_ROCM_BUNDLE")
    if bundle_override:
        bundle = normalize_rocm_bundle(bundle_override)
        return "", bundle, "PCI_LLAMA_CPP_ROCM_BUNDLE"

    targets, source = detected_gfx_targets()
    if not targets:
        fail("could not detect an AMD gfx target; install rocminfo or set PCI_AMD_GFX=gfxNNNN")

    bundles = {bundle_for_gfx_target(target) for target in targets}
    if len(bundles) != 1:
        fail(
            "detected multiple AMD gfx bundle families " + ", ".join(sorted(bundles)) + "; set PCI_AMD_GFX or "
            "PCI_LLAMA_CPP_ROCM_BUNDLE explicitly"
        )
    return targets[0], bundles.pop(), source


def latest_release(repo: str) -> GitHubRelease:
    normalized_repo = normalize_github_repo(repo)
    request = http_client.request(
        f"https://api.github.com/repos/{normalized_repo}/releases/latest",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "project-code-intelligence"},
        method="GET",
    )
    try:
        response_body = http_client.read_text(request, timeout=30)
        data_value = cast("object", json.loads(response_body))
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        fail(f"could not resolve latest {repo} release: {exc}")
    if not isinstance(data_value, dict):
        fail(f"unexpected latest release response from {repo}")
    data = cast("dict[str, object]", data_value)
    tag_name = data.get("tag_name")
    assets_value = data.get("assets")
    if not isinstance(tag_name, str) or not isinstance(assets_value, list):
        fail(f"latest release response from {normalized_repo} did not include tag_name and assets")

    typed_assets: list[dict[str, str]] = []
    assets = cast("list[object]", assets_value)
    for item in assets:
        if not isinstance(item, dict):
            continue
        asset_item = cast("dict[str, object]", item)
        name = asset_item.get("name")
        url = asset_item.get("browser_download_url")
        if isinstance(name, str) and isinstance(url, str):
            typed_assets.append({"name": name, "browser_download_url": url})
    return {"tag_name": tag_name, "assets": typed_assets}


def select_release_asset(repo: str, release: str, bundle: str) -> tuple[str, str, str]:
    if release != "latest":
        asset = llama_rocm_asset_name(release, bundle)
        return release, asset, llama_rocm_download_url(repo, release, bundle)

    metadata = latest_release(repo)
    suffix = f"ubuntu-rocm-{bundle}-x64.zip"
    for asset in metadata["assets"]:
        if asset["name"].endswith(suffix):
            return metadata["tag_name"], asset["name"], asset["browser_download_url"]
    fail(f"latest {repo} release did not include a Linux ROCm {bundle} asset")


def select(args: RocmNamespace) -> Selection:
    repo = args.repo
    release = args.release
    gfx_target, bundle, source = selected_bundle()
    release_tag, asset, url = select_release_asset(repo, release, bundle)
    return {
        "gfx_target": gfx_target,
        "bundle": bundle,
        "release": release_tag,
        "asset": asset,
        "url": url,
        "source": source,
    }


def print_env(selection: Selection) -> None:
    values = {
        "PCI_AMD_GFX": selection["gfx_target"],
        "PCI_LLAMA_CPP_ROCM_BUNDLE": selection["bundle"],
        "PCI_LLAMA_CPP_ROCM_RELEASE": selection["release"],
        "PCI_LLAMA_CPP_ROCM_ASSET": selection["asset"],
        "PCI_LLAMA_CPP_ROCM_URL": selection["url"],
    }
    for name, value in values.items():
        if value:
            write_stdout(f"export {name}={shlex.quote(value)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--repo",
        default=os.environ.get("PCI_LLAMA_CPP_ROCM_REPO", DEFAULT_LLAMACPP_ROCM_REPO),
        help="GitHub repository in owner/name form.",
    )
    _ = parser.add_argument(
        "--release",
        default=os.environ.get("PCI_LLAMA_CPP_ROCM_RELEASE", "latest"),
        help="Release tag, or 'latest' to query GitHub's latest release.",
    )
    _ = parser.add_argument(
        "--format",
        choices=("env", "url", "asset", "bundle", "gfx", "json"),
        default="env",
        help="Output format.",
    )
    args = parser.parse_args(namespace=RocmNamespace())
    try:
        selection = select(args)
    except RocmBundleError as exc:
        fail(str(exc))

    output_format = args.format
    if output_format == "env":
        print_env(selection)
    elif output_format == "url":
        write_stdout(selection["url"])
    elif output_format == "asset":
        write_stdout(selection["asset"])
    elif output_format == "bundle":
        write_stdout(selection["bundle"])
    elif output_format == "gfx":
        write_stdout(selection["gfx_target"])
    else:
        write_stdout(json.dumps(selection, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
