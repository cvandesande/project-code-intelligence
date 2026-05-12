"""ROCm gfx target mapping for Lemonade llama.cpp release bundles."""

from __future__ import annotations

import re

DEFAULT_LLAMACPP_ROCM_REPO = "lemonade-sdk/llamacpp-rocm"
SUPPORTED_ROCM_BUNDLES = ("gfx103X", "gfx110X", "gfx1150", "gfx1151", "gfx120X")
KNOWN_AMD_PCI_ID_GFX_TARGETS = {
    "1586": "gfx1151",
}
KNOWN_AMD_GPU_NAME_GFX_TARGETS = (
    ("strix halo", "gfx1151"),
    ("radeon 8060s", "gfx1151"),
    ("radeon 8050s", "gfx1151"),
    ("strix point", "gfx1150"),
    ("radeon 890m", "gfx1150"),
    ("radeon 880m", "gfx1150"),
)

_GFX_TARGET_RE = re.compile(r"\bgfx[0-9A-Za-z]+\b")
_PCI_ID_RE = re.compile(r"\b(?:0x)?([0-9A-Fa-f]{4})\b")
_GITHUB_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_RELEASE_TAG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class RocmBundleError(ValueError):
    """Raised when an AMD gfx target cannot be mapped to a release bundle."""


def normalize_gfx_target(value: str) -> str:
    match = _GFX_TARGET_RE.search(value.strip())
    if not match:
        raise RocmBundleError(f"could not find an AMD gfx target in {value!r}")
    return match.group(0).lower()


def gfx_targets_from_text(text: str) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    for match in _GFX_TARGET_RE.finditer(text):
        target = match.group(0).lower()
        if target not in seen:
            seen.add(target)
            targets.append(target)
    for match in _PCI_ID_RE.finditer(text):
        target = KNOWN_AMD_PCI_ID_GFX_TARGETS.get(match.group(1).lower())
        if target and target not in seen:
            seen.add(target)
            targets.append(target)
    normalized_text = text.lower()
    for name, target in KNOWN_AMD_GPU_NAME_GFX_TARGETS:
        if name in normalized_text and target not in seen:
            seen.add(target)
            targets.append(target)
    return targets


def gfx_target_from_pci_ids(vendor_id: str, device_id: str) -> str | None:
    vendor_match = _PCI_ID_RE.fullmatch(vendor_id.strip())
    device_match = _PCI_ID_RE.fullmatch(device_id.strip())
    if not vendor_match or not device_match:
        return None
    if vendor_match.group(1).lower() != "1002":
        return None
    return KNOWN_AMD_PCI_ID_GFX_TARGETS.get(device_match.group(1).lower())


def bundle_for_gfx_target(value: str) -> str:
    target = normalize_gfx_target(value)
    if target == "gfx1150":
        return "gfx1150"
    if target == "gfx1151":
        return "gfx1151"
    if target.startswith("gfx103"):
        return "gfx103X"
    if target.startswith("gfx110"):
        return "gfx110X"
    if target.startswith("gfx120"):
        return "gfx120X"
    raise RocmBundleError(
        f"unsupported AMD gfx target {target!r}; supported release bundles: " + ", ".join(SUPPORTED_ROCM_BUNDLES)
    )


def normalize_rocm_bundle(value: str) -> str:
    normalized = value.strip()
    for bundle in SUPPORTED_ROCM_BUNDLES:
        if normalized.lower() == bundle.lower():
            return bundle
    if normalized.lower().startswith("gfx"):
        return bundle_for_gfx_target(normalized)
    raise RocmBundleError(
        f"unsupported ROCm bundle {value!r}; supported release bundles: " + ", ".join(SUPPORTED_ROCM_BUNDLES)
    )


def llama_rocm_asset_name(release_tag: str, bundle: str) -> str:
    release = normalize_release_tag(release_tag)
    if release.startswith("llama-"):
        release = release.removeprefix("llama-")
    return f"llama-{release}-ubuntu-rocm-{normalize_rocm_bundle(bundle)}-x64.zip"


def normalize_release_tag(value: str) -> str:
    normalized = value.strip()
    if not normalized or not _RELEASE_TAG_RE.fullmatch(normalized):
        raise RocmBundleError("release tag must contain only letters, numbers, dot, underscore, or dash")
    return normalized


def normalize_github_repo(value: str) -> str:
    normalized = value.strip().strip("/")
    if not _GITHUB_REPO_RE.fullmatch(normalized):
        raise RocmBundleError("repository must be in owner/name form")
    return normalized


def llama_rocm_download_url(repo: str, release_tag: str, bundle: str) -> str:
    repository = normalize_github_repo(repo)
    release = normalize_release_tag(release_tag)
    asset = llama_rocm_asset_name(release_tag, bundle)
    return f"https://github.com/{repository}/releases/download/{release}/{asset}"
