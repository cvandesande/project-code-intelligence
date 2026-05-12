"""Optional Linux power and energy sensor helpers for benchmark tooling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class EnergySensor:
    label: str
    path: Path
    max_energy_range_uj: int | None


@dataclass(frozen=True)
class PowerSensor:
    label: str
    path: Path


PowerSensorSource = EnergySensor | PowerSensor


@dataclass(frozen=True)
class PowerMeasurement:
    label: str
    source: str
    source_type: str
    elapsed_seconds: float
    energy_joules: float | None
    average_watts: float | None
    samples: int
    note: str | None = None


def read_int(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").strip())


def read_label(path: Path) -> str:
    for parent in path.parents:
        name_path = parent / "name"
        if name_path.is_file():
            name = name_path.read_text(encoding="utf-8", errors="replace").strip()
            if name:
                return f"{name}:{path.name}"
        if parent == parent.parent:
            break
    return f"{path.parent.name}:{path.name}"


def energy_delta_uj(start_uj: int, end_uj: int, max_energy_range_uj: int | None) -> int:
    delta = end_uj - start_uj
    if delta >= 0:
        return delta
    if max_energy_range_uj is None or max_energy_range_uj <= 0:
        return 0
    return (max_energy_range_uj - start_uj) + end_uj


def energy_sensor(path: Path) -> EnergySensor:
    max_path = path.parent / "max_energy_range_uj"
    max_energy_range_uj = read_int(max_path) if max_path.is_file() else None
    return EnergySensor(label=read_label(path), path=path, max_energy_range_uj=max_energy_range_uj)


def power_sensor(path: Path) -> PowerSensor:
    return PowerSensor(label=read_label(path), path=path)


def parse_power_source(value: str) -> list[PowerSensorSource]:
    path = Path(value)
    if not path.exists():
        raise ValueError(f"power source path does not exist: {path}")
    if path.is_dir():
        energy_path = path / "energy_uj"
        if energy_path.is_file():
            return [energy_sensor(energy_path)]
        power_paths = sorted([*path.glob("power*_average"), *path.glob("power*_input")])
        if power_paths:
            return [power_sensor(power_path) for power_path in power_paths]
        raise ValueError(f"power source directory has no supported sensors: {path}")
    if path.name == "energy_uj":
        return [energy_sensor(path)]
    if path.name.startswith("power") and path.name.endswith(("_input", "_average")):
        return [power_sensor(path)]
    raise ValueError(f"unsupported power source path: {path}")


def unique_sources(sensors: Iterable[PowerSensorSource]) -> list[PowerSensorSource]:
    seen: set[Path] = set()
    unique: list[PowerSensorSource] = []
    for sensor in sensors:
        if sensor.path in seen:
            continue
        seen.add(sensor.path)
        unique.append(sensor)
    return unique


def discover_power_sources(sys_root: Path = Path("/sys")) -> list[PowerSensorSource]:
    sensors: list[PowerSensorSource] = []
    powercap_root = sys_root / "class" / "powercap"
    if powercap_root.is_dir():
        sensors.extend(energy_sensor(path) for path in sorted(powercap_root.rglob("energy_uj")))

    hwmon_root = sys_root / "class" / "hwmon"
    if hwmon_root.is_dir():
        power_paths: list[Path] = []
        for hwmon in sorted(hwmon_root.glob("hwmon*")):
            averaged = sorted(hwmon.glob("power*_average"))
            averaged_channels = {path.name.removesuffix("_average") for path in averaged}
            power_paths.extend(averaged)
            power_paths.extend(
                path
                for path in sorted(hwmon.glob("power*_input"))
                if path.name.removesuffix("_input") not in averaged_channels
            )
        sensors.extend(power_sensor(path) for path in power_paths)

    return unique_sources(sensors)


class PowerMonitor:
    def __init__(self, sensors: list[PowerSensorSource], sample_interval_seconds: float) -> None:
        self._sensors = sensors
        self._sample_interval_seconds = sample_interval_seconds
        self._started_at = 0.0
        self._start_energy_uj: dict[Path, int] = {}
        self._power_samples_uw: dict[Path, list[int]] = {
            sensor.path: [] for sensor in sensors if isinstance(sensor, PowerSensor)
        }
        self._notes: dict[Path, str] = {}
        self._stop = Event()
        self._lock = Lock()
        self._thread: Thread | None = None

    def start(self) -> None:
        self._started_at = time.perf_counter()
        for sensor in self._sensors:
            if isinstance(sensor, EnergySensor):
                try:
                    self._start_energy_uj[sensor.path] = read_int(sensor.path)
                except (OSError, ValueError) as exc:
                    self._notes[sensor.path] = str(exc)
        if self._power_samples_uw:
            self._sample_power_once()
            self._thread = Thread(target=self._sample_power_until_stopped, daemon=True)
            self._thread.start()

    def stop(self) -> list[PowerMeasurement]:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(1.0, self._sample_interval_seconds * 2.0))
        elapsed_seconds = max(time.perf_counter() - self._started_at, 0.0)
        return [self._measurement(sensor, elapsed_seconds) for sensor in self._sensors]

    def _sample_power_until_stopped(self) -> None:
        while not self._stop.wait(self._sample_interval_seconds):
            self._sample_power_once()

    def _sample_power_once(self) -> None:
        with self._lock:
            for sensor in self._sensors:
                if not isinstance(sensor, PowerSensor):
                    continue
                try:
                    self._power_samples_uw[sensor.path].append(read_int(sensor.path))
                except (OSError, ValueError) as exc:
                    _ = self._notes.setdefault(sensor.path, str(exc))

    def _measurement(self, sensor: PowerSensorSource, elapsed_seconds: float) -> PowerMeasurement:
        note = self._notes.get(sensor.path)
        if isinstance(sensor, EnergySensor):
            start_uj = self._start_energy_uj.get(sensor.path)
            try:
                end_uj = read_int(sensor.path)
            except (OSError, ValueError) as exc:
                return PowerMeasurement(
                    label=sensor.label,
                    source=str(sensor.path),
                    source_type="energy_uj",
                    elapsed_seconds=elapsed_seconds,
                    energy_joules=None,
                    average_watts=None,
                    samples=0,
                    note=str(exc),
                )
            if start_uj is None:
                return PowerMeasurement(
                    label=sensor.label,
                    source=str(sensor.path),
                    source_type="energy_uj",
                    elapsed_seconds=elapsed_seconds,
                    energy_joules=None,
                    average_watts=None,
                    samples=0,
                    note=note or "initial energy reading was unavailable",
                )
            energy_joules = energy_delta_uj(start_uj, end_uj, sensor.max_energy_range_uj) / 1_000_000.0
            average_watts = energy_joules / elapsed_seconds if elapsed_seconds > 0 else None
            return PowerMeasurement(
                label=sensor.label,
                source=str(sensor.path),
                source_type="energy_uj",
                elapsed_seconds=elapsed_seconds,
                energy_joules=energy_joules,
                average_watts=average_watts,
                samples=2,
                note=note,
            )

        samples = self._power_samples_uw.get(sensor.path, [])
        if not samples:
            return PowerMeasurement(
                label=sensor.label,
                source=str(sensor.path),
                source_type="power_uw",
                elapsed_seconds=elapsed_seconds,
                energy_joules=None,
                average_watts=None,
                samples=0,
                note=note or "no power samples were recorded",
            )
        average_watts = (sum(samples) / len(samples)) / 1_000_000.0
        return PowerMeasurement(
            label=sensor.label,
            source=str(sensor.path),
            source_type="power_uw",
            elapsed_seconds=elapsed_seconds,
            energy_joules=average_watts * elapsed_seconds,
            average_watts=average_watts,
            samples=len(samples),
            note=note,
        )
