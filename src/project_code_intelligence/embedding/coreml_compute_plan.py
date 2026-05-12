"""Core ML compute plan inspection for device assignment analysis."""

from __future__ import annotations

import sys
from importlib import import_module
from typing import cast


def _compile_for_plan(model_path: str) -> str | None:
    """Compile a .mlpackage to .mlmodelc for compute plan inspection."""
    if model_path.endswith(".mlmodelc"):
        return model_path
    try:
        ct = import_module("coremltools")
    except ImportError:
        return None
    utils_mod = cast("object", getattr(ct, "utils", None))
    compile_fn = getattr(utils_mod, "compile_model", None) if utils_mod else None
    if compile_fn is None:
        return None
    try:
        return cast("str", compile_fn(model_path))
    except Exception:  # noqa: BLE001 - coremltools compile can raise arbitrary errors
        return None


def _classify_device(
    device: object,
    *,
    ane_cls: type[object] | None,
    gpu_cls: type[object] | None,
    cpu_cls: type[object] | None,
) -> str:
    """Return a device label for a preferred_compute_device instance."""
    if ane_cls is not None and isinstance(device, ane_cls):
        return "ANE"
    if gpu_cls is not None and isinstance(device, gpu_cls):
        return "GPU"
    if cpu_cls is not None and isinstance(device, cpu_cls):
        return "CPU"
    return "unknown"


def _plan_operations(plan: object) -> list[object]:
    """Extract the flat list of ML Program operations from a compute plan."""
    model_structure = cast("object | None", getattr(plan, "model_structure", None))
    program = cast("object | None", getattr(model_structure, "program", None)) if model_structure is not None else None
    functions_raw = cast("object | None", getattr(program, "functions", None)) if program is not None else None
    if functions_raw is None:
        return []
    ops: list[object] = []
    for function in cast("dict[str, object]", functions_raw).values():
        block = cast("object | None", getattr(function, "block", None))
        operations_raw = cast("object | None", getattr(block, "operations", None)) if block is not None else None
        if operations_raw is not None:
            ops.extend(cast("list[object]", operations_raw))
    return ops


def _walk_plan_operations(
    plan: object,
    *,
    ane_cls: type[object] | None,
    gpu_cls: type[object] | None,
    cpu_cls: type[object] | None,
) -> tuple[dict[str, int], float, int]:
    """Walk ML Program operations and count device assignments.

    Returns (counts_by_device, ane_weight, total_ops).
    """
    ops = _plan_operations(plan)
    if not ops:
        return {}, 0.0, 0

    get_usage = getattr(plan, "get_compute_device_usage_for_mlprogram_operation", None)
    get_cost = getattr(plan, "get_estimated_cost_for_mlprogram_operation", None)
    counts: dict[str, int] = {"ANE": 0, "GPU": 0, "CPU": 0, "unknown": 0}
    ane_weight = 0.0

    for op in ops:
        if get_usage is None:
            counts["unknown"] += 1
            continue
        usage = cast("object | None", get_usage(op))
        if usage is None:
            counts["unknown"] += 1
            continue
        device = cast("object | None", getattr(usage, "preferred_compute_device", None))
        if device is None:
            counts["unknown"] += 1
            continue
        label = _classify_device(device, ane_cls=ane_cls, gpu_cls=gpu_cls, cpu_cls=cpu_cls)
        counts[label] = counts.get(label, 0) + 1
        if label == "ANE" and get_cost is not None:
            cost = cast("object | None", get_cost(op))
            weight = cast("object | None", getattr(cost, "weight", None)) if cost is not None else None
            if isinstance(weight, float):
                ane_weight += weight
    return counts, ane_weight, len(ops)


def _load_compute_plan(model_path: str) -> object | None:
    """Compile model and load its MLComputePlan, or return None on failure."""
    try:
        compute_plan_mod = import_module("coremltools.models.compute_plan")
    except ImportError:
        _ = sys.stderr.write("  MLComputePlan requires coremltools 8.0+; skipping compute plan analysis.\n")
        return None

    ct = import_module("coremltools")
    compute_unit_cls = cast("object", getattr(ct, "ComputeUnit", None))
    compute_all = cast("object", getattr(compute_unit_cls, "ALL", None)) if compute_unit_cls is not None else None
    if compute_all is None:
        _ = sys.stderr.write("  ComputeUnit.ALL not available.\n")
        return None

    plan_cls = cast("object", getattr(compute_plan_mod, "MLComputePlan", None))
    load_fn = getattr(plan_cls, "load_from_path", None) if plan_cls is not None else None
    if load_fn is None:
        _ = sys.stderr.write("  MLComputePlan.load_from_path not available.\n")
        return None

    compiled_path = _compile_for_plan(model_path)
    if compiled_path is None:
        _ = sys.stderr.write("  Could not compile model for compute plan inspection.\n")
        return None

    try:
        return cast("object", load_fn(compiled_path, compute_units=compute_all))
    except Exception as exc:  # noqa: BLE001 - coremltools internals can raise arbitrary errors
        _ = sys.stderr.write(f"  Could not load compute plan: {exc}\n")
        return None


def format_compute_plan(model_path: str) -> str | None:
    """Load a Core ML compute plan and return a device assignment summary string."""
    plan = _load_compute_plan(model_path)
    if plan is None:
        return None

    try:
        compute_device_mod = import_module("coremltools.models.compute_device")
    except ImportError:
        return None

    ane_cls = cast("type[object] | None", getattr(compute_device_mod, "MLNeuralEngineComputeDevice", None))
    gpu_cls = cast("type[object] | None", getattr(compute_device_mod, "MLGPUComputeDevice", None))
    cpu_cls = cast("type[object] | None", getattr(compute_device_mod, "MLCPUComputeDevice", None))

    counts, ane_weight, total = _walk_plan_operations(plan, ane_cls=ane_cls, gpu_cls=gpu_cls, cpu_cls=cpu_cls)

    if total == 0:
        return None

    lines: list[str] = [f"\n  Compute plan ({total} operations):"]
    for name in ("ANE", "GPU", "CPU"):
        count = counts.get(name, 0)
        if count > 0:
            lines.append(f"    {name}: {count} ops ({100 * count // total}%)")
    unknown = counts.get("unknown", 0)
    if unknown > 0:
        lines.append(f"    Unassigned: {unknown} ops")

    if counts.get("ANE", 0) > 0:
        ane_count = counts["ANE"]
        lines.append(f"\n  Neural Engine will handle {ane_count}/{total} operations.")
        if ane_weight > 0:
            lines.append(f"  Estimated ANE workload share: {ane_weight:.0%}")
    else:
        lines.extend([
            "\n  No operations scheduled for Neural Engine.",
            "  This model may not support ANE acceleration on this hardware.",
        ])
    lines.append("")
    return "\n".join(lines)


def print_compute_plan(model_path: str) -> None:
    """Load a Core ML compute plan and print device assignment summary to stderr."""
    text = format_compute_plan(model_path)
    if text:
        _ = sys.stderr.write(text)
