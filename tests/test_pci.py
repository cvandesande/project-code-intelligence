import io
import unittest
from tempfile import TemporaryDirectory
from unittest import mock

from project_code_intelligence import pci


class PciDispatchTests(unittest.TestCase):
    def test_no_command_prints_usage_and_fails(self) -> None:
        with mock.patch.object(pci.sys, "stderr", io.StringIO()) as err:
            self.assertEqual(pci.main([]), 2)
        self.assertIn("usage: pci", err.getvalue())

    def test_unknown_command_fails(self) -> None:
        with mock.patch.object(pci.sys, "stderr", io.StringIO()):
            self.assertEqual(pci.main(["frobnicate"]), 2)

    def test_help_succeeds(self) -> None:
        with mock.patch.object(pci.sys, "stdout", io.StringIO()) as out:
            self.assertEqual(pci.main(["--help"]), 0)
        self.assertIn("hook", out.getvalue())

    def test_dispatch_reaches_hook_install_and_target_flag_works(self) -> None:
        with TemporaryDirectory() as tmp:
            args = ["hook", "install", "--target", "claude", "--project", tmp, "--dry-run", "--color", "never"]
            code = pci.main(args)
        self.assertEqual(code, 0)

    def test_services_verbs_map_to_doctor_flags(self) -> None:
        self.assertEqual(
            pci.resolve("services", ["start"]), (("project_code_intelligence.doctor", "main", ["--start"]), [])
        )
        self.assertEqual(pci.resolve("services", []), (("project_code_intelligence.doctor", "main", []), []))
        self.assertIsNone(pci.resolve("services", ["reboot"]))

    def test_status_resolves_to_the_status_cli(self) -> None:
        self.assertEqual(
            pci.resolve("status", ["--json"]),
            (("project_code_intelligence.status_cli", "main", []), ["--json"]),
        )

    def test_legacy_agent_spelling_still_accepted(self) -> None:
        with TemporaryDirectory() as tmp:
            code = pci.main(["hook", "install", "--agent", "claude", "--project", tmp, "--dry-run", "--color", "never"])
        self.assertEqual(code, 0)
