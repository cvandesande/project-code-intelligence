from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from project_code_intelligence.power import (
    EnergySensor,
    PowerMonitor,
    PowerSensor,
    discover_power_sources,
    energy_delta_uj,
    parse_power_source,
    read_label,
)


class PowerTests(unittest.TestCase):
    def test_energy_delta_handles_monotonic_and_wrapped_counters(self) -> None:
        self.assertEqual(energy_delta_uj(100, 175, None), 75)
        self.assertEqual(energy_delta_uj(900, 25, 1000), 125)
        self.assertEqual(energy_delta_uj(900, 25, None), 0)

    def test_read_label_uses_nearest_name_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sensor_dir = Path(tmp) / "class" / "hwmon" / "hwmon0"
            sensor_dir.mkdir(parents=True)
            _ = (sensor_dir / "name").write_text("amdgpu\n", encoding="utf-8")
            power_path = sensor_dir / "power1_average"
            _ = power_path.write_text("123000000\n", encoding="utf-8")

            self.assertEqual(read_label(power_path), "amdgpu:power1_average")

    def test_discover_power_sources_finds_powercap_and_hwmon_sensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sys_root = Path(tmp)
            powercap = sys_root / "class" / "powercap" / "intel-rapl:0"
            powercap.mkdir(parents=True)
            _ = (powercap / "name").write_text("package-0\n", encoding="utf-8")
            _ = (powercap / "energy_uj").write_text("100\n", encoding="utf-8")
            _ = (powercap / "max_energy_range_uj").write_text("1000000\n", encoding="utf-8")

            hwmon = sys_root / "class" / "hwmon" / "hwmon0"
            hwmon.mkdir(parents=True)
            _ = (hwmon / "name").write_text("amdgpu\n", encoding="utf-8")
            _ = (hwmon / "power1_average").write_text("12000000\n", encoding="utf-8")
            _ = (hwmon / "power1_input").write_text("13000000\n", encoding="utf-8")
            _ = (hwmon / "power2_input").write_text("5000000\n", encoding="utf-8")

            sensors = discover_power_sources(sys_root)

            self.assertEqual(
                [sensor.label for sensor in sensors],
                ["package-0:energy_uj", "amdgpu:power1_average", "amdgpu:power2_input"],
            )
            self.assertIsInstance(sensors[0], EnergySensor)
            self.assertIsInstance(sensors[1], PowerSensor)
            self.assertIsInstance(sensors[2], PowerSensor)

    def test_parse_power_source_accepts_directories_and_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sensor_dir = Path(tmp)
            _ = (sensor_dir / "name").write_text("demo\n", encoding="utf-8")
            _ = (sensor_dir / "energy_uj").write_text("100\n", encoding="utf-8")

            parsed = parse_power_source(str(sensor_dir))

            self.assertEqual(len(parsed), 1)
            self.assertIsInstance(parsed[0], EnergySensor)
            self.assertEqual(parsed[0].label, "demo:energy_uj")

    def test_parse_power_source_rejects_missing_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            _ = parse_power_source(str(Path(tmp) / "missing" / "energy_uj"))

    def test_power_monitor_reports_energy_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sensor_dir = Path(tmp)
            _ = (sensor_dir / "name").write_text("demo\n", encoding="utf-8")
            energy_path = sensor_dir / "energy_uj"
            _ = energy_path.write_text("1000000\n", encoding="utf-8")
            sensor = EnergySensor(label="demo:energy_uj", path=energy_path, max_energy_range_uj=None)

            monitor = PowerMonitor([sensor], sample_interval_seconds=0.01)
            monitor.start()
            _ = energy_path.write_text("1500000\n", encoding="utf-8")
            measurements = monitor.stop()

            self.assertEqual(len(measurements), 1)
            self.assertEqual(measurements[0].energy_joules, 0.5)
            self.assertIsNotNone(measurements[0].average_watts)

    def test_power_monitor_reports_sampled_power(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sensor_dir = Path(tmp)
            power_path = sensor_dir / "power1_average"
            _ = power_path.write_text("2000000\n", encoding="utf-8")
            sensor = PowerSensor(label="demo:power1_average", path=power_path)

            monitor = PowerMonitor([sensor], sample_interval_seconds=0.01)
            monitor.start()
            time.sleep(0.02)
            measurements = monitor.stop()

            self.assertEqual(len(measurements), 1)
            self.assertEqual(measurements[0].average_watts, 2.0)
            self.assertGreaterEqual(measurements[0].samples, 1)


if __name__ == "__main__":
    _ = unittest.main()
