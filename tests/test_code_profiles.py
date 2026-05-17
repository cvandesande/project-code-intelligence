from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

from project_code_intelligence.code_profiles.example import ExampleProfile, string_list

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject


def base_classification() -> JsonObject:
    return {
        "file_role": "source",
        "content_class": "source",
        "is_source": True,
        "is_test": False,
    }


class ExampleProfileTests(unittest.TestCase):
    def test_string_list_coerces_list_items_and_rejects_other_values(self) -> None:
        self.assertEqual(string_list(["alpha", 2, True]), ["alpha", "2", "True"])
        self.assertEqual(string_list(("alpha", "beta")), [])
        self.assertEqual(string_list("alpha"), [])

    def test_classify_file_marks_services_deployments_and_architecture_docs(self) -> None:
        profile = ExampleProfile()
        original = base_classification()

        service = profile.classify_file("services/api/app.py", "python", original)
        deployment = profile.classify_file("deploy/prod/app.yaml", "yaml", original)
        infra = profile.classify_file("infra/db/config.toml", "toml", original)
        architecture = profile.classify_file("docs/architecture/overview.md", "markdown", original)
        ordinary = profile.classify_file("src/main.py", "python", original)

        self.assertEqual(service["file_role"], "service")
        self.assertEqual(deployment["file_role"], "deployment")
        self.assertEqual(infra["file_role"], "deployment")
        self.assertEqual(architecture["file_role"], "architecture-doc")
        self.assertEqual(ordinary["file_role"], "source")
        self.assertEqual(original["file_role"], "source")

    def test_file_metadata_extracts_service_and_deployment_scope(self) -> None:
        profile = ExampleProfile()

        self.assertEqual(
            profile.file_metadata("services/api/runtime/server.py", "python", base_classification()),
            {"service": "api", "service_area": "runtime"},
        )
        self.assertEqual(
            profile.file_metadata("deploy/prod/app.yaml", "yaml", base_classification()),
            {"deployment_area": "prod"},
        )
        self.assertEqual(
            profile.file_metadata("infra", "yaml", base_classification()),
            {"deployment_area": "infra"},
        )
        self.assertEqual(profile.file_metadata("src/main.py", "python", base_classification()), {})

    def test_extra_records_extracts_deployment_objects_from_yaml(self) -> None:
        profile = ExampleProfile()

        records, edges = profile.extra_records(
            "deploy/prod/app.yaml",
            "deploy/prod/app.yaml",
            "yaml",
            "kind: Service\nname: api\nspec: {}\n---\nname: worker\n",
        )

        self.assertEqual(edges, [])
        self.assertEqual([record.get("symbol") for record in records], ["api", "worker"])
        self.assertEqual(
            [record.get("record_id") for record in records],
            [
                "deploy/prod/app.yaml::deployment_object::api::000002",
                "deploy/prod/app.yaml::deployment_object::worker::000005",
            ],
        )
        self.assertEqual(records[0].get("metadata"), {"deployment_object": "api"})
        self.assertEqual(records[0].get("confidence_kind"), "heuristic_candidate")

        self.assertEqual(profile.extra_records("src/app.yaml", "src/app.yaml", "yaml", "name: api"), ([], []))
        self.assertEqual(profile.extra_records("deploy/app.json", "deploy/app.json", "json", "name: api"), ([], []))

    def test_security_context_adds_profile_specific_service_and_deployment_markers(self) -> None:
        profile = ExampleProfile()

        service = profile.security_context("services/api/app.py", "python", "service", "source")
        deployment = profile.security_context("deploy/prod/app.yaml", "yaml", "deployment", "config")

        self.assertEqual(service["security_contexts"], ["service_code", "source_code"])
        self.assertEqual(deployment["boundary_candidates"], ["config_input", "deployment_boundary"])

    def test_embedding_metadata_keys_include_profile_specific_fields(self) -> None:
        keys = ExampleProfile().embedding_metadata_keys()

        self.assertIn("symbols_defined", keys)
        self.assertIn("service", keys)
        self.assertIn("service_area", keys)
        self.assertIn("deployment_area", keys)
        self.assertIn("deployment_object", keys)


if __name__ == "__main__":
    _ = unittest.main()
