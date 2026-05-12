"""Local hardware detection for native environment diagnostics."""

from __future__ import annotations

import os
import platform
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, cast

from project_code_intelligence import config, process
from project_code_intelligence.doctor.common import (
    bytes_from_text,
    human_bytes,
    result,
    status_for_requirement,
    version_at_least,
    version_tuple,
)
from project_code_intelligence.doctor.types import CheckResult, GpuInfo, Status

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

AMD_NPU_MIN_KERNEL = (7, 0)
AMD_NPU_MIN_FIRMWARE = (1, 1, 0, 0)
NVIDIA_SMI_MIN_CSV_COLUMNS = 3
GIB = 1024 * 1024 * 1024
GPU_QWEN3_DEFAULT_MODEL = config.DEFAULT_GPU_EMBEDDING_MODEL
GPU_QWEN3_LARGE_MODEL = config.DEFAULT_LARGE_GPU_EMBEDDING_MODEL


def npu_embedding_required(env: config.Env) -> bool:
    model = config.default_embedding_endpoint_model(env=env).strip().lower()
    return model.endswith("-flm")


def cpuinfo_text(path: Path | None = None) -> str:
    path = Path("/proc/cpuinfo") if path is None else path
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def cpu_suggests_supported_amd_npu(cpu_text: str) -> bool:
    normalized = " ".join(cpu_text.lower().split())
    return bool(re.search(r"ryzen ai.*((\b[34]\d{2}\b)|(\bz2\b))", normalized))


def read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def parse_uevent(text: str | None) -> dict[str, str]:
    values: dict[str, str] = {}
    if not text:
        return values
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def pci_id_parts(value: str | None) -> tuple[str | None, str | None]:
    if not value or ":" not in value:
        return None, None
    vendor_id, device_id = value.split(":", 1)
    return vendor_id.lower(), device_id.lower()


def vendor_name(vendor_id: str | None) -> str:
    normalized = (vendor_id or "").lower()
    if normalized in {"0x1002", "1002"}:
        return "AMD"
    if normalized in {"0x10de", "10de"}:
        return "NVIDIA"
    if normalized in {"0x8086", "8086"}:
        return "Intel"
    if normalized in {"0x106b", "106b"}:
        return "Apple"
    return "Unknown"


def driver_name(device_path: Path, uevent: Mapping[str, str]) -> str | None:
    driver = uevent.get("DRIVER")
    if driver:
        return driver
    driver_path = device_path / "driver"
    try:
        return driver_path.resolve().name
    except OSError:
        return None


def linux_drm_gpus(root: Path | None = None) -> list[GpuInfo]:
    root = Path("/sys/class/drm") if root is None else root
    if not root.exists():
        return []
    gpus: list[GpuInfo] = []
    for card in sorted(root.glob("card[0-9]*")):
        if not re.fullmatch(r"card\d+", card.name):
            continue
        device_path = card / "device"
        if not device_path.exists():
            continue
        uevent = parse_uevent(read_text_file(device_path / "uevent"))
        vendor_id = read_text_file(device_path / "vendor")
        device_id = read_text_file(device_path / "device")
        pci_vendor_id, pci_device_id = pci_id_parts(uevent.get("PCI_ID"))
        vendor_id = vendor_id or pci_vendor_id
        device_id = device_id or pci_device_id
        vendor = vendor_name(vendor_id)
        name = f"{vendor} GPU"
        if vendor_id or device_id:
            name += f" {vendor_id or 'unknown'}:{device_id or 'unknown'}"
        gpus.append(
            GpuInfo(
                name=name,
                vendor=vendor,
                vendor_id=vendor_id,
                device_id=device_id,
                driver=driver_name(device_path, uevent),
                card=card.name,
                pci_slot=uevent.get("PCI_SLOT_NAME"),
                vram_bytes=bytes_from_text(read_text_file(device_path / "mem_info_vram_total")),
                visible_vram_bytes=bytes_from_text(read_text_file(device_path / "mem_info_vis_vram_total")),
                shared_bytes=bytes_from_text(read_text_file(device_path / "mem_info_gtt_total")),
                source="drm-sysfs",
            )
        )
    return gpus


def parse_nvidia_smi_csv(output: str) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for index, line in enumerate(output.splitlines()):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < NVIDIA_SMI_MIN_CSV_COLUMNS:
            continue
        name, memory_mib, driver = parts[:NVIDIA_SMI_MIN_CSV_COLUMNS]
        try:
            memory_bytes = int(memory_mib) * 1024 * 1024
        except ValueError:
            memory_bytes = None
        gpus.append(
            GpuInfo(
                name=name or "NVIDIA GPU",
                vendor="NVIDIA",
                driver=f"nvidia {driver}" if driver else "nvidia",
                card=f"nvidia{index}",
                vram_bytes=memory_bytes,
                visible_vram_bytes=memory_bytes,
                source="nvidia-smi",
            )
        )
    return gpus


def nvidia_smi_gpus() -> list[GpuInfo]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        proc = process.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            process.RunOptions(
                capture_output=True,
                timeout=5,
                check=False,
            ),
        )
    except (OSError, process.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return parse_nvidia_smi_csv(proc.stdout)


def total_system_memory_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return None
    return pages * page_size


def discover_gpus() -> list[GpuInfo]:
    system = platform.system()
    if system == "Darwin" and platform.machine().lower() == "arm64":
        return [
            GpuInfo(
                name="Apple Silicon GPU",
                vendor="Apple",
                driver="Metal",
                shared_bytes=total_system_memory_bytes(),
                source="platform",
            )
        ]
    if system != "Linux":
        return []
    nvidia_gpus = nvidia_smi_gpus()
    drm_gpus = linux_drm_gpus()
    if not nvidia_gpus:
        return drm_gpus
    return [*nvidia_gpus, *[gpu for gpu in drm_gpus if gpu.vendor != "NVIDIA"]]


def amd_npu_firmware_versions() -> list[str]:
    paths = {
        *Path("/sys/bus/pci/drivers/amdxdna").glob("*/fw_version"),
        *Path("/sys/class/accel").glob("accel*/device/fw_version"),
    }
    versions: set[str] = set()
    for path in sorted(paths):
        value = read_text_file(path)
        if value:
            versions.add(value)
    return sorted(versions, key=version_tuple)


def _check_darwin_npu() -> list[CheckResult]:
    """Check Apple Neural Engine availability on macOS arm64."""
    if platform.machine().lower() != "arm64":
        return [result("npu", "skip", "No supported Apple Neural Engine backend is available on this Mac.")]
    if not coremltools_available():
        return [
            result(
                "npu",
                "skip",
                "Apple Neural Engine requires coremltools; using Metal GPU path.",
                "pip install coremltools transformers to enable the Core ML backend (ANE + GPU + CPU).",
            )
        ]
    cores = detect_ane_cores()
    detail = (
        "Core ML with compute_units=ALL distributes work across ANE, GPU, and CPU. "
        "Run pci-coreml-server --diagnose for per-operation device assignments."
    )
    if cores is not None and cores > 0:
        message = f"Apple Neural Engine detected: {cores} cores, available via Core ML."
    else:
        message = "Apple Neural Engine is available via Core ML (coremltools installed)."
    return [result("npu", "ok", message, detail)]


def check_npu_support(env: config.Env) -> list[CheckResult]:
    required = npu_embedding_required(env)
    system = platform.system()
    if system == "Darwin":
        return _check_darwin_npu()
    if system != "Linux":
        return [result("npu", "skip", f"NPU checks are not implemented for {system}.")]

    results: list[CheckResult] = []
    device_paths = sorted(Path("/dev/accel").glob("accel*"))
    device_names = ", ".join(str(path) for path in device_paths)
    supported_cpu = cpu_suggests_supported_amd_npu(cpuinfo_text())
    if not device_paths:
        if supported_cpu:
            results.append(
                result(
                    "npu",
                    status_for_requirement(ok=False, required=required),
                    "AMD Ryzen AI XDNA 2 CPU appears present, but no /dev/accel/accel* NPU device was found.",
                    "Install Linux kernel 7.0+ with amdxdna or amdxdna-dkms, update firmware, then retry.",
                )
            )
        else:
            results.append(result("npu", "skip", "No supported local NPU device was detected."))
        return results

    accessible_paths = [path for path in device_paths if os.access(path, os.R_OK | os.W_OK)]
    results.append(
        result(
            "npu",
            "ok" if accessible_paths else status_for_requirement(ok=False, required=required),
            f"AMD NPU device detected: {device_names}",
            None if accessible_paths else "The current user needs read/write access, usually through the render group.",
        )
    )

    release = platform.release()
    kernel_ok = version_at_least(version_tuple(release), AMD_NPU_MIN_KERNEL)
    driver_present = Path("/sys/module/amdxdna").exists() or Path("/sys/bus/pci/drivers/amdxdna").exists()
    if kernel_ok:
        results.append(result("npu-kernel", "ok", f"Linux kernel {release} meets the kernel 7.0+ requirement."))
    elif driver_present:
        results.append(
            result(
                "npu-kernel",
                "warn",
                f"Linux kernel {release} is below 7.0, but amdxdna appears present.",
                "This may be an amdxdna-dkms/backport setup. Validate with flm validate.",
            )
        )
    else:
        results.append(
            result(
                "npu-kernel",
                status_for_requirement(ok=False, required=required),
                f"Linux kernel {release} is below 7.0 and amdxdna was not detected.",
                "AMD FLM NPU support needs kernel 7.0+ with amdxdna, or amdxdna-dkms.",
            )
        )

    results.append(
        result(
            "npu-driver",
            "ok" if driver_present else status_for_requirement(ok=False, required=required),
            "amdxdna driver detected." if driver_present else "amdxdna driver was not detected.",
            None if driver_present else "Install kernel 7.0+ with amdxdna or install amdxdna-dkms.",
        )
    )

    firmware_versions = amd_npu_firmware_versions()
    if firmware_versions:
        oldest = min(firmware_versions, key=version_tuple)
        firmware_ok = all(
            version_at_least(version_tuple(version), AMD_NPU_MIN_FIRMWARE) for version in firmware_versions
        )
        results.append(
            result(
                "npu-firmware",
                "ok" if firmware_ok else status_for_requirement(ok=False, required=required),
                f"AMD NPU firmware version(s): {', '.join(firmware_versions)}",
                None
                if firmware_ok
                else (
                    f"Firmware must be at least {'.'.join(str(part) for part in AMD_NPU_MIN_FIRMWARE)}; "
                    f"oldest detected is {oldest}."
                ),
            )
        )
    else:
        results.append(
            result(
                "npu-firmware",
                status_for_requirement(ok=False, required=required),
                "AMD NPU firmware version was not found in sysfs.",
                "Expected /sys/bus/pci/drivers/amdxdna/*/fw_version. Validate with flm validate.",
            )
        )
    return results


def gpu_memory_summary(gpu: GpuInfo) -> str:
    parts: list[str] = []
    if gpu.vram_bytes is not None:
        parts.append(f"VRAM={human_bytes(gpu.vram_bytes)}")
    if gpu.visible_vram_bytes is not None and gpu.visible_vram_bytes != gpu.vram_bytes:
        parts.append(f"visible VRAM={human_bytes(gpu.visible_vram_bytes)}")
    if gpu.shared_bytes is not None:
        parts.append(f"shared/unified={human_bytes(gpu.shared_bytes)}")
    return "; ".join(parts) if parts else "memory=unknown"


def gpu_runtime_detail(gpu: GpuInfo) -> str | None:
    values = [
        value
        for value in (
            f"card={gpu.card}" if gpu.card else None,
            f"driver={gpu.driver}" if gpu.driver else None,
            f"pci={gpu.pci_slot}" if gpu.pci_slot else None,
            f"source={gpu.source}",
        )
        if value
    ]
    return "; ".join(values) if values else None


def gpu_effective_memory_bytes(gpu: GpuInfo) -> int | None:
    candidates = [value for value in (gpu.vram_bytes, gpu.shared_bytes) if value is not None]
    return max(candidates) if candidates else None


def max_gpu_memory_bytes(gpus: Sequence[GpuInfo]) -> int | None:
    values = [value for gpu in gpus if (value := gpu_effective_memory_bytes(gpu)) is not None]
    return max(values) if values else None


def has_gpu_vendor(gpus: Sequence[GpuInfo], vendor: str) -> bool:
    return any(gpu.vendor == vendor for gpu in gpus)


def has_ready_npu(results: Sequence[CheckResult]) -> bool:
    required = {"npu", "npu-driver", "npu-firmware", "npu-kernel"}
    statuses = {item.name: item.status for item in results if item.name in required}
    return required.issubset(statuses) and all(status == "ok" for status in statuses.values())


def check_gpu_support(gpus: Sequence[GpuInfo]) -> list[CheckResult]:
    if not gpus:
        return [result("gpu", "skip", "No local GPU was detected.")]

    results: list[CheckResult] = []
    for index, gpu in enumerate(gpus):
        results.append(
            result(
                f"gpu-{index}",
                "ok",
                f"{gpu.name}: {gpu_memory_summary(gpu)}",
                gpu_runtime_detail(gpu),
            )
        )

    if has_gpu_vendor(gpus, "AMD"):
        kfd_accessible = Path("/dev/kfd").exists() and os.access(Path("/dev/kfd"), os.R_OK | os.W_OK)
        dri_accessible = any(
            os.access(path, os.R_OK | os.W_OK)
            for path in [*Path("/dev/dri").glob("renderD*"), *Path("/dev/dri").glob("card*")]
        )
        results.append(
            result(
                "gpu-runtime-amd",
                "ok" if kfd_accessible and dri_accessible else "warn",
                "AMD GPU runtime devices are accessible."
                if kfd_accessible and dri_accessible
                else "AMD GPU detected, but ROCm container device access may be incomplete.",
                None
                if kfd_accessible and dri_accessible
                else "Check /dev/kfd and /dev/dri permissions before using the amdgpu Compose profile.",
            )
        )

    if has_gpu_vendor(gpus, "NVIDIA"):
        nvidia_smi = shutil.which("nvidia-smi")
        nvidia_device_present = Path("/dev/nvidiactl").exists() or any(Path("/dev").glob("nvidia[0-9]*"))
        results.append(
            result(
                "gpu-runtime-nvidia",
                "ok" if nvidia_smi and nvidia_device_present else "warn",
                "NVIDIA runtime tools and devices are present."
                if nvidia_smi and nvidia_device_present
                else "NVIDIA GPU detected, but nvidia-smi or /dev/nvidia* was not found.",
                None
                if nvidia_smi and nvidia_device_present
                else "Install the NVIDIA driver and NVIDIA Container Toolkit before using the nvidia Compose profile.",
            )
        )

    if has_gpu_vendor(gpus, "Intel"):
        results.append(
            result(
                "gpu-runtime-intel",
                "skip",
                "Intel GPU detected, but this project does not currently ship an Intel GPU embedding profile.",
            )
        )
    return results


def check_platform(env: config.Env) -> list[CheckResult]:
    results = [
        result(
            "platform",
            "ok",
            f"Python {platform.python_version()} on {platform.system()} {platform.release()} ({platform.machine()})",
        )
    ]
    llama_model = config.env_text("PROJECT_CODE_INTELLIGENCE_LLAMA_MODEL", env=env)
    llama_configured = bool(
        llama_model
        or config.env_text("PROJECT_CODE_INTELLIGENCE_LLAMA_SERVER", env=env)
        or config.env_text("PROJECT_CODE_INTELLIGENCE_LLAMA_CPP_DIR", env=env)
    )
    if platform.system() != "Darwin":
        return results

    apple_backend = (config.env_text("PROJECT_CODE_INTELLIGENCE_APPLE_BACKEND", env=env) or "").strip().lower()
    coreml_available = coremltools_available()
    if apple_backend == "coreml" or (apple_backend != "metal" and coreml_available):
        results.extend(_check_coreml(env))
    elif llama_configured:
        results.extend(_check_apple_metal(env, llama_model=llama_model))
    elif coreml_available:
        results.extend(_check_coreml(env))
    else:
        results.append(
            result(
                "apple-metal",
                "skip",
                "Apple Metal llama.cpp is not configured.",
                "Set PROJECT_CODE_INTELLIGENCE_LLAMA_MODEL for Metal, "
                "or pip install coremltools transformers for Core ML (ANE + GPU).",
            )
        )
    return results


def coremltools_available() -> bool:
    try:
        __import__("coremltools")
    except ImportError:
        return False
    return True


def detect_ane_cores() -> int | None:
    """Return the Neural Engine core count via coremltools, or None if unavailable."""
    try:
        compute_device = cast("object", __import__("coremltools.models.compute_device", fromlist=["MLComputeDevice"]))
    except (ImportError, Exception):  # noqa: BLE001 - coremltools native lib may fail
        return None
    device_cls = cast("object | None", getattr(compute_device, "MLComputeDevice", None))
    get_all: Callable[[], list[object]] | None = (
        cast("Callable[[], list[object]] | None", getattr(device_cls, "get_all_compute_devices", None))
        if device_cls is not None
        else None
    )
    ane_cls = cast("type[object] | None", getattr(compute_device, "MLNeuralEngineComputeDevice", None))
    if get_all is None or ane_cls is None:
        return None
    try:
        devices = get_all()
    except Exception:  # noqa: BLE001 - native framework may fail
        return None
    for device in devices:
        if isinstance(device, ane_cls):
            core_count = cast("object | None", getattr(device, "total_core_count", None))
            return core_count if isinstance(core_count, int) else 0
    return None


def _check_coreml(env: config.Env) -> list[CheckResult]:
    results: list[CheckResult] = []
    cores = detect_ane_cores()
    if cores is not None and cores > 0:
        results.append(
            result(
                "apple-coreml",
                "ok",
                f"coremltools is available; Neural Engine has {cores} cores; Core ML backend uses ANE + GPU + CPU.",
                "Run pci-coreml-server --diagnose for per-operation device assignments.",
            )
        )
    else:
        results.append(
            result(
                "apple-coreml",
                "ok",
                "coremltools is available; Core ML backend uses ANE + GPU + CPU.",
                "Run pci-coreml-server --diagnose for per-operation device assignments.",
            )
        )
    model_name = config.env_text("PROJECT_CODE_INTELLIGENCE_COREML_MODEL", config.DEFAULT_COREML_MODEL, env=env)
    results.append(
        result(
            "apple-coreml-model",
            "ok",
            f"Core ML embedding model: {model_name}.",
            "Set PROJECT_CODE_INTELLIGENCE_COREML_MODEL to override.",
        )
    )
    return results


def _check_apple_metal(env: config.Env, *, llama_model: str | None) -> list[CheckResult]:
    results: list[CheckResult] = []
    llama_server = config.env_text("PROJECT_CODE_INTELLIGENCE_LLAMA_SERVER", "llama-server", env=env) or "llama-server"
    resolved_llama_server = shutil.which(llama_server) if Path(llama_server).name == llama_server else llama_server
    if resolved_llama_server and Path(resolved_llama_server).exists():
        results.append(result("apple-metal", "ok", f"llama-server found at {resolved_llama_server}"))
    else:
        results.append(
            result(
                "apple-metal",
                "warn",
                "llama-server was not found; install llama.cpp on the macOS host for Apple GPU embeddings.",
                "Recommended install path: brew install llama.cpp. "
                "Or pip install coremltools transformers for Core ML (ANE + GPU).",
            )
        )

    if llama_model:
        model_path = Path(llama_model)
        status: Status = "ok" if model_path.is_file() else "fail"
        message = (
            f"llama.cpp embedding model exists: {llama_model}"
            if model_path.is_file()
            else f"llama.cpp embedding model was not found: {llama_model}"
        )
        results.append(result("apple-metal-model", status, message))
    else:
        results.append(
            result(
                "apple-metal-model",
                "warn",
                "PROJECT_CODE_INTELLIGENCE_LLAMA_MODEL is not set.",
                "Set it to a GGUF embedding model such as Qwen3-Embedding-0.6B-Q8_0.gguf.",
            )
        )
    return results
