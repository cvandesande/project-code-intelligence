from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from project_code_intelligence import config
from project_code_intelligence.db import DatabaseRole, PostgresBootstrapResult
from project_code_intelligence.doctor import (
    CheckResult,
    GpuInfo,
    check_embedding_endpoint,
    check_embedding_options,
    check_gpu_support,
    color_text,
    cpu_suggests_supported_amd_npu,
    format_postgres_bootstrap_result,
    format_result,
    format_summary,
    gpu_memory_summary,
    human_bytes,
    parse_nvidia_smi_csv,
    remote_provider_precheck,
    should_use_color,
    status_for_requirement,
    summary_status,
    version_at_least,
    version_tuple,
)
from project_code_intelligence.doctor import cli as doctor_cli
from project_code_intelligence.doctor.database import check_database
from project_code_intelligence.embedding.bench import EmbeddingRequestResult


def successful_requester(
    endpoint: str,
    model: str,
    texts: list[str],
    timeout: float,
) -> EmbeddingRequestResult:
    _ = endpoint
    _ = texts
    _ = timeout
    return EmbeddingRequestResult(seconds=0.01, dimensions=3, response_model=model, response_bytes=128)


def require_check_result(item: CheckResult | None) -> CheckResult:
    if item is None:
        raise AssertionError("expected diagnostic result")
    return item


class _FakeCursor:
    def __init__(self, row: dict[str, object] | None) -> None:
        self._row = row

    def fetchone(self) -> dict[str, object] | None:
        return self._row


class _FakeMaintenanceConnection:
    def __enter__(self) -> _FakeMaintenanceConnection:
        return self

    def __exit__(self, _exc_type: object, exc: object, traceback: object) -> None:
        return None

    @staticmethod
    def execute(query: object, params: object | None = None) -> _FakeCursor:
        _ = query
        _ = params
        return _FakeCursor({
            "database_name": "postgres",
            "user_name": "postgres",
            "version": "PostgreSQL 17",
        })


class DoctorTests(unittest.TestCase):
    def test_human_bytes_formats_binary_units(self) -> None:
        self.assertEqual(human_bytes(None), "unknown")
        self.assertEqual(human_bytes(512), "512 B")
        self.assertEqual(human_bytes(1024 * 1024), "1.0 MiB")
        self.assertEqual(human_bytes(8 * 1024 * 1024 * 1024), "8.0 GiB")

    def test_gpu_memory_summary_reports_vram_and_shared_memory(self) -> None:
        summary = gpu_memory_summary(
            GpuInfo(
                name="AMD GPU",
                vendor="AMD",
                vram_bytes=512 * 1024 * 1024,
                shared_bytes=64 * 1024 * 1024 * 1024,
            )
        )

        self.assertIn("VRAM=512.0 MiB", summary)
        self.assertIn("shared/unified=64.0 GiB", summary)

    def test_parse_nvidia_smi_csv(self) -> None:
        gpus = parse_nvidia_smi_csv("NVIDIA GeForce RTX 4070, 12282, 555.55\n")

        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].vendor, "NVIDIA")
        self.assertEqual(gpus[0].vram_bytes, 12282 * 1024 * 1024)

    def test_embedding_options_include_gpu_and_large_model_candidate(self) -> None:
        results = check_embedding_options(
            env={},
            gpus=[GpuInfo(name="Apple Silicon GPU", vendor="Apple", shared_bytes=18 * 1024 * 1024 * 1024)],
            npu_results=[],
        )
        names = [item.name for item in results]

        self.assertIn("option-cpu", names)
        self.assertIn("option-gpu-apple", names)
        # Large GGUF model option is for AMD/NVIDIA GPU profiles only.
        self.assertNotIn("option-gpu-large-model", names)
        self.assertIn("option-remote", names)

    def test_version_tuple_parses_kernel_and_firmware_versions(self) -> None:
        self.assertEqual(version_tuple("7.0.4+deb13-amd64"), (7, 0, 4))
        self.assertEqual(version_tuple("NPU FW Version: 1.1.2.65"), (1, 1, 2, 65))

    def test_version_at_least_pads_short_versions(self) -> None:
        self.assertTrue(version_at_least((7, 0), (7, 0, 0)))
        self.assertTrue(version_at_least((1, 1, 2, 65), (1, 1, 0, 0)))
        self.assertFalse(version_at_least((6, 19, 13), (7, 0)))
        self.assertFalse(version_at_least((1, 0, 9, 0), (1, 1, 0, 0)))

    def test_npu_requirement_status_only_fails_when_required(self) -> None:
        self.assertEqual(status_for_requirement(ok=False, required=False), "warn")
        self.assertEqual(status_for_requirement(ok=False, required=True), "fail")
        self.assertEqual(status_for_requirement(ok=True, required=True), "ok")

    def test_cpu_detection_matches_supported_ryzen_ai_names(self) -> None:
        self.assertTrue(cpu_suggests_supported_amd_npu("AMD Ryzen AI Max+ 395 w/ Radeon 8060S"))
        self.assertTrue(cpu_suggests_supported_amd_npu("AMD Ryzen AI 9 HX 370"))
        self.assertTrue(cpu_suggests_supported_amd_npu("AMD Ryzen AI Z2 Extreme"))
        self.assertFalse(cpu_suggests_supported_amd_npu("AMD Ryzen 7 8845HS"))

    def test_nvidia_runtime_warns_with_container_toolkit_link_when_ctk_missing(self) -> None:
        def which(command: str) -> str | None:
            return "/usr/bin/nvidia-smi" if command == "nvidia-smi" else None

        with (
            patch("project_code_intelligence.doctor.hardware.shutil.which", side_effect=which),
            patch("project_code_intelligence.doctor.hardware.Path.exists", return_value=True),
            patch("project_code_intelligence.doctor.hardware.docker_has_nvidia_runtime", return_value=(False, None)),
        ):
            results = check_gpu_support([GpuInfo(name="NVIDIA L4", vendor="NVIDIA")])

        item = next(result for result in results if result.name == "gpu-runtime-nvidia")
        self.assertEqual(item.status, "warn")
        self.assertIn("nvidia-ctk", item.message)
        self.assertIn("NVIDIA Container Toolkit", item.message)
        self.assertIn("nvidia-ctk runtime configure", item.detail or "")
        self.assertIn("https://docs.nvidia.com/datacenter/cloud-native/container-toolkit", item.detail or "")

    def test_nvidia_runtime_warns_when_docker_lacks_nvidia_runtime(self) -> None:
        def which(command: str) -> str | None:
            if command in {"nvidia-smi", "nvidia-ctk"}:
                return f"/usr/bin/{command}"
            return None

        with (
            patch("project_code_intelligence.doctor.hardware.shutil.which", side_effect=which),
            patch("project_code_intelligence.doctor.hardware.Path.exists", return_value=True),
            patch(
                "project_code_intelligence.doctor.hardware.docker_has_nvidia_runtime",
                return_value=(False, "Docker did not report an nvidia runtime."),
            ),
        ):
            results = check_gpu_support([GpuInfo(name="NVIDIA L4", vendor="NVIDIA")])

        item = next(result for result in results if result.name == "gpu-runtime-nvidia")
        self.assertEqual(item.status, "warn")
        self.assertIn("Docker is not configured", item.message)
        self.assertIn("Docker did not report an nvidia runtime", item.detail or "")
        self.assertIn("https://docs.nvidia.com/datacenter/cloud-native/container-toolkit", item.detail or "")

    def test_color_output_can_be_forced_or_disabled(self) -> None:
        self.assertTrue(should_use_color("always", stdout_isatty=False, env={}))
        self.assertFalse(should_use_color("never", stdout_isatty=True, env={}))
        self.assertFalse(should_use_color("auto", stdout_isatty=True, env={"NO_COLOR": "1"}))
        self.assertTrue(should_use_color("auto", stdout_isatty=False, env={"FORCE_COLOR": "1"}))

    def test_format_result_colorizes_status_when_enabled(self) -> None:
        plain = format_result(CheckResult("demo", "ok", "ready"), color=False)
        colored = format_result(CheckResult("demo", "ok", "ready"), color=True)

        self.assertEqual(plain, "[ok] demo: ready")
        self.assertIn(color_text("ok", "\033[32m", enabled=True), colored)

    def test_summary_status_prioritizes_failures_then_warnings(self) -> None:
        self.assertEqual(summary_status([CheckResult("demo", "ok", "ready")]), ("ok", "ready"))
        self.assertEqual(summary_status([CheckResult("demo", "warn", "note")]), ("warn", "usable with notes"))
        self.assertEqual(
            summary_status([CheckResult("demo", "warn", "note"), CheckResult("db", "fail", "down")]),
            ("fail", "needs attention"),
        )

    def test_format_summary_is_concise_and_keeps_verbose_details_out(self) -> None:
        with patch("project_code_intelligence.process.container_engine_name", return_value="docker"):
            output = format_summary(
                [
                    CheckResult("platform", "ok", "Python 3.13 on Linux"),
                    CheckResult("gpu-0", "ok", "AMD GPU: shared/unified=64.0 GiB", "card=card0; driver=amdgpu"),
                    CheckResult("gpu-runtime-amd", "ok", "AMD GPU runtime devices are accessible."),
                    CheckResult("npu", "ok", "AMD NPU device detected: /dev/accel/accel0"),
                    CheckResult(
                        "database",
                        "ok",
                        "connected to codeintel as codeintel at "
                        "postgresql://codeintel@127.0.0.1:5433/codeintel sslmode=prefer",
                        "PostgreSQL 17",
                    ),
                    CheckResult(
                        "embedding-config",
                        "ok",
                        "endpoint=http://127.0.0.1:18081/v1/embeddings model=local",
                    ),
                    CheckResult(
                        "embedding-endpoint",
                        "ok",
                        "response model=embed-gemma-300m-FLM; dimensions=768; latency=0.01s",
                    ),
                    CheckResult("option-cpu", "ok", "CPU embeddings: FastEmbed default demo."),
                    CheckResult("option-npu", "ok", "AMD NPU embeddings: Lemonade FLM default demo."),
                    CheckResult("option-gpu-amd", "ok", "AMD GPU embeddings: llama.cpp ROCm default demo.gguf."),
                ],
                color=False,
            )

        self.assertIn("✓ READY", output)
        self.assertIn("Postgres", output)
        self.assertIn("127.0.0.1:5433", output)
        self.assertNotIn("codeintel @ 127.0.0.1:5433", output)
        self.assertNotIn("postgresql://codeintel@", output)
        self.assertNotIn("codeintel:codeintel", output)
        self.assertIn("Active path", output)
        self.assertIn("NPU", output)
        self.assertIn("http://127.0.0.1:18081/v1/embeddings", output)
        self.assertNotIn("response model=embed-gemma-300m-FLM; dimensions=", output)
        self.assertNotIn("Available embedding paths", output)
        self.assertNotIn("Switch embedding runtime", output)
        self.assertNotIn("cpu: docker compose --profile cpu up -d --build fastembed", output)
        self.assertNotIn("npu: docker compose --profile npu up -d lemonade-npu", output)
        self.assertNotIn("PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT=", output)
        self.assertNotIn("PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT_MODEL=", output)
        self.assertNotIn("amdgpu: docker compose --profile amdgpu up -d --build llama-rocm", output)
        self.assertNotIn("nvidia: docker compose --profile nvidia", output)
        self.assertNotIn("card=card0", output)

    def test_format_summary_suggests_database_initialization_for_db_issues(self) -> None:
        with patch("project_code_intelligence.process.container_engine_name", return_value="docker"):
            output = format_summary(
                [
                    CheckResult("platform", "ok", "Python 3.13 on Linux"),
                    CheckResult("database", "fail", "Could not connect to PostgreSQL/pgvector."),
                ],
                color=False,
            )

        self.assertIn("Start a local database", output)
        self.assertIn("docker compose up -d pgvector", output)
        self.assertIn("Prepare Postgres roles", output)
        self.assertIn("pci-doctor --init-postgres", output)
        self.assertIn("Index a repo and bootstrap its inferred database", output)
        self.assertNotIn("Prepare inferred DB roles", output)
        self.assertNotIn("pci-doctor --init-db", output)

    def test_format_summary_keeps_gpu_memory_summary_compact(self) -> None:
        output = format_summary(
            [
                CheckResult("platform", "ok", "Python 3.13.5 on Linux 7.0.4+deb13-amd64 (x86_64)"),
                CheckResult(
                    "gpu-0",
                    "ok",
                    "AMD GPU 0x1002:0x1586: VRAM=512.0 MiB; shared/unified=62.5 GiB",
                    "card=card0; driver=amdgpu",
                ),
                CheckResult(
                    "npu",
                    "ok",
                    "AMD NPU 0x1022:0x17f0",
                    "device=accel0; path=/dev/accel/accel0; driver=amdxdna",
                ),
                CheckResult(
                    "database",
                    "ok",
                    "connected to code-intel as app at postgresql://app@db.example.invalid:30432/code-intel",
                ),
                CheckResult(
                    "embedding-endpoint",
                    "ok",
                    "response model=Qwen3-Embedding-0.6B-Q8_0; dimensions=1024; latency=0.014s",
                ),
            ],
            color=False,
        )

        self.assertIn("AMD 1002:1586 · amdgpu · VRAM 512 MiB · shared 62.5 GiB", output)
        self.assertIn("AMD NPU 1022:17F0 · amdxdna · accel0", output)
        self.assertIn("Postgres", output)
        self.assertIn("db.example.invalid:30432", output)
        self.assertNotIn("code-intel @ db.example.invalid:30432", output)
        self.assertNotIn("shared/unified=62.5", output)
        self.assertNotIn("card=card0", output)
        self.assertNotIn("\n│               GiB", output)

    def test_format_summary_lists_remote_only_after_endpoint_validation(self) -> None:
        output = format_summary(
            [
                CheckResult("platform", "ok", "Python 3.13 on Linux"),
                CheckResult("gpu", "skip", "No local GPU was detected."),
                CheckResult("npu", "skip", "No supported local NPU device was detected."),
                CheckResult("database", "ok", "connected to codeintel as codeintel"),
                CheckResult(
                    "embedding-config",
                    "ok",
                    "endpoint=https://api.openai.com/v1/embeddings model=text-embedding-3-small",
                ),
                CheckResult("embedding-endpoint", "ok", "response model=text-embedding-3-small"),
                CheckResult("option-cpu", "ok", "CPU embeddings: FastEmbed default demo."),
                CheckResult("option-remote", "ok", "Remote OpenAI-compatible embeddings."),
            ],
            color=False,
        )

        self.assertIn("Active path", output)
        self.assertIn("Remote", output)
        self.assertIn("https://api.openai.com/v1/embeddings", output)
        self.assertNotIn("Available embedding paths", output)
        self.assertNotIn("remote: no embedding container needed", output)

    def test_format_summary_colorizes_headings_and_startup_profiles(self) -> None:
        with patch("project_code_intelligence.process.container_engine_name", return_value="docker"):
            output = format_summary(
                [
                    CheckResult("platform", "ok", "Python 3.13 on Linux"),
                    CheckResult("gpu", "skip", "No local GPU was detected."),
                    CheckResult("npu", "skip", "No supported local NPU device was detected."),
                    CheckResult("database", "ok", "connected to codeintel as codeintel"),
                    CheckResult("embedding-endpoint", "warn", "no endpoint configured"),
                    CheckResult("option-cpu", "ok", "CPU embeddings: FastEmbed default demo."),
                ],
                color=True,
            )

        self.assertIn("\033[1m", output)
        self.assertIn("\033[2m", output)
        self.assertIn("System", output)
        self.assertIn("Start CPU embeddings", output)
        self.assertIn("docker compose --profile cpu up -d --build fastembed", output)

    def test_format_summary_uses_user_facing_issue_labels(self) -> None:
        output = format_summary(
            [
                CheckResult("platform", "ok", "Python 3.13 on Linux"),
                CheckResult("gpu", "skip", "No local GPU was detected."),
                CheckResult("npu", "ok", "AMD NPU device detected: /dev/accel/accel0"),
                CheckResult("npu-kernel", "warn", "Linux kernel 6.19 is below 7.0, but amdxdna appears present."),
                CheckResult("npu-firmware", "fail", "AMD NPU firmware version(s): 1.0.0.0"),
                CheckResult("database", "skip", "database check skipped"),
                CheckResult("embedding", "skip", "embedding check skipped"),
            ],
            color=False,
        )

        self.assertIn("NPU kernel", output)
        self.assertIn("Linux kernel 6.19 is below 7.0", output)
        self.assertIn("NPU firmware", output)
        self.assertIn("AMD NPU firmware version", output)
        self.assertNotIn("npu-kernel", output)
        self.assertNotIn("npu-firmware", output)


class DoctorDatabaseTests(unittest.TestCase):
    def test_check_database_uses_maintenance_database_for_inferred_settings(self) -> None:
        admin_credential = "-".join(("admin", "fixture"))
        settings = config.DatabaseSettings(
            dsn="postgresql://db.example.invalid/pci_test?sslmode=prefer",
            dbname="pci_test",
            admin_user="postgres",
            admin_password=admin_credential,
            database_inferred=True,
        )
        with (
            patch("project_code_intelligence.doctor.database.config.DatabaseSettings.from_env", return_value=settings),
            patch("project_code_intelligence.doctor.database.db.connect", return_value=_FakeMaintenanceConnection()),
            patch("project_code_intelligence.process.container_engine_name", return_value="docker"),
        ):
            results = check_database()

        by_name = {item.name: item for item in results}
        self.assertEqual(by_name["database"].status, "ok")
        self.assertNotIn("project-database", by_name)

        output = format_summary(results, color=False)

        self.assertIn("✓ Postgres", output)
        self.assertNotIn("Project DB", output)
        self.assertNotIn("pci-doctor --init-db", output)
        self.assertNotIn("docker compose up -d pgvector", output)

    def test_format_postgres_bootstrap_result_prints_index_admin_exports(self) -> None:
        credential = " ".join(("secret", "value"))
        output = format_postgres_bootstrap_result(
            PostgresBootstrapResult(
                postgres_url="postgresql://db.example.invalid:5432?sslmode=prefer",
                index_role=DatabaseRole(
                    name="pci_index_admin",
                    password=credential,
                    created=True,
                    database_url="postgresql://db.example.invalid:5432?sslmode=prefer",
                ),
                vector_template_ready=True,
                vector_template_created=True,
            ),
            color=False,
        )

        self.assertIn("project-code-intelligence postgres roles", output)
        self.assertIn("pci_index_admin created", output)
        self.assertIn("CREATEDB", output)
        self.assertIn("CREATEROLE", output)
        self.assertIn("pgvector", output)
        self.assertIn("created in template1", output)
        self.assertIn(
            "export PROJECT_CODE_INTELLIGENCE_DATABASE_URL='postgresql://db.example.invalid:5432?sslmode=prefer'",
            output,
        )
        self.assertIn("export PROJECT_CODE_INTELLIGENCE_DATABASE_ADMIN_USER=pci_index_admin", output)
        self.assertIn("export PROJECT_CODE_INTELLIGENCE_DATABASE_ADMIN_PASSWORD='secret value'", output)
        self.assertNotIn("Project DB", output)
        self.assertNotIn("pci-doctor --init-db", output)

    def test_format_postgres_bootstrap_result_prints_ready_for_existing_index_admin(self) -> None:
        credential = " ".join(("secret", "value"))
        output = format_postgres_bootstrap_result(
            PostgresBootstrapResult(
                postgres_url="postgresql://db.example.invalid:5432?sslmode=prefer",
                index_role=DatabaseRole(
                    name="pci_index_admin",
                    password=credential,
                    created=False,
                    database_url="postgresql://db.example.invalid:5432?sslmode=prefer",
                ),
                vector_template_ready=True,
                vector_template_created=False,
            ),
            color=False,
        )

        self.assertIn("pci_index_admin ready", output)
        self.assertIn("ready in template1", output)

    def test_init_postgres_reads_postgres_admin_environment(self) -> None:
        captured_settings: list[config.DatabaseSettings] = []
        postgres_credential = "-".join(("postgres", "fixture"))
        index_credential = "-".join(("index", "fixture"))

        def fake_bootstrap(settings: config.DatabaseSettings) -> PostgresBootstrapResult:
            captured_settings.append(settings)
            return PostgresBootstrapResult(
                postgres_url="postgresql://db.example.invalid:5432?sslmode=prefer",
                index_role=DatabaseRole(
                    name="pci_index_admin",
                    password=index_credential,
                    created=True,
                    database_url="postgresql://db.example.invalid:5432?sslmode=prefer",
                ),
                vector_template_ready=True,
                vector_template_created=False,
            )

        with (
            patch.dict(
                os.environ,
                {
                    "PROJECT_CODE_INTELLIGENCE_DATABASE_URL": "postgresql://db.example.invalid:5432?sslmode=prefer",
                    "PROJECT_CODE_INTELLIGENCE_POSTGRES_ADMIN_USER": "postgres",
                    "PROJECT_CODE_INTELLIGENCE_POSTGRES_ADMIN_PASSWORD": postgres_credential,
                    "PROJECT_CODE_INTELLIGENCE_DATABASE_ADMIN_USER": "pci_index_admin",
                    "PROJECT_CODE_INTELLIGENCE_DATABASE_ADMIN_PASSWORD": index_credential,
                },
                clear=True,
            ),
            patch("project_code_intelligence.doctor.cli.db.bootstrap_postgres_roles", side_effect=fake_bootstrap),
            patch("project_code_intelligence.doctor.cli.write_stdout"),
        ):
            status = doctor_cli.init_postgres_roles(doctor_cli.DoctorArgs(color="never"))

        self.assertEqual(status, 0)
        self.assertEqual(captured_settings[0].admin_user, "postgres")
        self.assertEqual(captured_settings[0].admin_password, postgres_credential)


class DoctorEndpointTests(unittest.TestCase):
    def test_openai_endpoint_uses_openai_api_key(self) -> None:
        self.assertEqual(
            config.embedding_api_key(
                "https://api.openai.com/v1/embeddings",
                env={"OPENAI_API_KEY": "openai-key"},
            ),
            "openai-key",
        )

    def test_voyage_endpoint_uses_voyage_api_key(self) -> None:
        self.assertEqual(
            config.embedding_api_key(
                "https://api.voyageai.com/v1/embeddings",
                env={"VOYAGE_API_KEY": "voyage-key"},
            ),
            "voyage-key",
        )

    def test_generic_embedding_api_key_takes_precedence(self) -> None:
        self.assertEqual(
            config.embedding_api_key(
                "https://api.openai.com/v1/embeddings",
                env={
                    "PROJECT_CODE_INTELLIGENCE_EMBEDDING_API_KEY": "generic-key",
                    "OPENAI_API_KEY": "openai-key",
                },
            ),
            "generic-key",
        )

    def test_anthropic_endpoint_is_not_treated_as_embeddings_provider(self) -> None:
        item = remote_provider_precheck(
            "https://api.anthropic.com/v1/messages",
            "claude-sonnet-4-5",
            required=True,
            env={},
        )

        item = require_check_result(item)
        self.assertEqual(item.status, "fail")
        self.assertIn("not a first-party embeddings endpoint", item.message)

    def test_remote_endpoint_needs_explicit_model(self) -> None:
        item = remote_provider_precheck(
            "https://api.openai.com/v1/embeddings",
            "local",
            required=True,
            env={"OPENAI_API_KEY": "openai-key"},
        )

        item = require_check_result(item)
        self.assertEqual(item.status, "fail")
        self.assertIn(config.DEFAULT_OPENAI_EMBEDDING_MODEL, item.detail or "")

    def test_configured_remote_endpoint_can_pass_preflight(self) -> None:
        results = check_embedding_endpoint(
            env={
                "PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT": "https://api.openai.com/v1/embeddings",
                "PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT_MODEL": "text-embedding-3-small",
                "PROJECT_CODE_INTELLIGENCE_ALLOW_REMOTE_EMBEDDING": "1",
                "OPENAI_API_KEY": "openai-key",
            },
            mode="auto",
            timeout=1.0,
            requester=successful_requester,
        )

        self.assertEqual(results[-1].status, "ok")
        self.assertEqual(results[-1].name, "embedding-endpoint")

    def test_local_endpoint_resolves_runtime_model_before_preflight(self) -> None:
        with patch(
            "project_code_intelligence.doctor.embeddings.resolve_embedding_endpoint_model",
            return_value="embed-gemma-300m-FLM",
        ):
            results = check_embedding_endpoint(
                env={},
                mode="auto",
                timeout=1.0,
                requester=successful_requester,
            )

        config_result = next(item for item in results if item.name == "embedding-config")
        self.assertIn("model=embed-gemma-300m-FLM", config_result.message)
        self.assertEqual(results[-1].status, "ok")


class DoctorAppleTests(unittest.TestCase):
    def test_embedding_options_shows_mlx_for_apple(self) -> None:
        results = check_embedding_options(
            env={},
            gpus=[GpuInfo(name="Apple Silicon GPU", vendor="Apple", shared_bytes=16 * 1024 * 1024 * 1024)],
            npu_results=[],
        )
        apple = next(r for r in results if r.name == "option-gpu-apple")

        self.assertEqual(apple.status, "ok")
        self.assertIn("MLX", apple.message)

    def test_format_summary_shows_apple_metal_startup_command(self) -> None:
        output = format_summary(
            [
                CheckResult("platform", "ok", "Python 3.13 on Darwin 25.4.0 (arm64)"),
                CheckResult(
                    "npu", "skip", "Apple Neural Engine is not used; embeddings run via pci-apple-embed-server (MPS)."
                ),
                CheckResult("database", "ok", "connected to codeintel as codeintel"),
                CheckResult("embedding-endpoint", "warn", "no endpoint configured"),
                CheckResult(
                    "option-gpu-apple",
                    "ok",
                    "Apple GPU embeddings: native MLX default mlx-community/Qwen3-Embedding-0.6B-8bit.",
                ),
            ],
            color=False,
        )

        self.assertIn("Start Apple native embeddings", output)
        self.assertIn("pci-apple-embed-server", output)


if __name__ == "__main__":
    _ = unittest.main()
