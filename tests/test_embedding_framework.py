from __future__ import annotations

import unittest
from unittest.mock import patch

from project_code_intelligence import ingest_code_intel
from project_code_intelligence.embedding.framework import active_embedding_profile, option_label


class EmbeddingFrameworkTests(unittest.TestCase):
    def test_active_profile_uses_doctor_gpu_labels(self) -> None:
        profile = active_embedding_profile(
            endpoint="http://127.0.0.1:18081/v1/embeddings",
            response_model="Qwen3-Embedding-0.6B-Q8_0.gguf",
            endpoint_ok=True,
            option_ok=lambda name: name == "option-gpu-amd",
        )

        self.assertEqual(profile.profile, "amdgpu")
        self.assertEqual(profile.label, "AMD ROCm")

    def test_active_profile_keeps_remote_label_for_remote_endpoint(self) -> None:
        profile = active_embedding_profile(
            endpoint="https://api.openai.com/v1/embeddings",
            response_model="text-embedding-3-small",
            endpoint_ok=True,
            option_ok=lambda _name: False,
        )

        self.assertEqual(profile.profile, "remote")
        self.assertEqual(profile.label, "Remote")

    def test_option_label_is_shared_with_doctor_output(self) -> None:
        self.assertEqual(option_label("option-gpu-nvidia"), "NVIDIA CUDA")

    def test_pci_index_framework_uses_shared_profile_selector(self) -> None:
        def option_ok(name: str) -> bool:
            return name == "option-gpu-amd"

        with (
            patch(
                "project_code_intelligence.ingest_code_intel.resolve_embedding_endpoint_framework", return_value=None
            ),
            patch(
                "project_code_intelligence.ingest_code_intel.index_embedding_option_ok",
                side_effect=option_ok,
            ),
        ):
            label = ingest_code_intel.resolve_index_embedding_framework(
                "http://127.0.0.1:18081/v1/embeddings",
                "Qwen3-Embedding-0.6B-Q8_0.gguf",
            )

        self.assertEqual(label, "AMD ROCm")


if __name__ == "__main__":
    _ = unittest.main()
