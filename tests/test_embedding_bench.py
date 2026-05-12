from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, cast

from project_code_intelligence.embedding_bench import (
    BenchmarkResult,
    batch_for_run,
    generated_texts,
    line_chunks,
    parse_embedding_response,
    percentile,
    repository_texts,
    result_json,
    text_stats,
)
from project_code_intelligence.power import PowerMeasurement

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject

TEST_TEXT_CHARS = 120


class EmbeddingBenchTests(unittest.TestCase):
    def test_generated_texts_are_code_like_and_sized(self) -> None:
        texts = generated_texts(3, TEST_TEXT_CHARS)

        self.assertEqual(len(texts), 3)
        self.assertTrue(all(len(text) == TEST_TEXT_CHARS for text in texts))
        self.assertTrue(any("embedding_benchmark_sample=0" in text for text in texts))

    def test_parse_embedding_response_returns_dimensions_and_model(self) -> None:
        raw_response = json.dumps({
            "model": "demo-model",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
            ],
        })

        dimensions, model = parse_embedding_response(raw_response, expected_count=2)

        self.assertEqual(dimensions, 3)
        self.assertEqual(model, "demo-model")

    def test_parse_embedding_response_rejects_inconsistent_dimensions(self) -> None:
        raw_response = json.dumps({
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                {"object": "embedding", "index": 1, "embedding": [0.3]},
            ],
        })

        with self.assertRaises(ValueError):
            _ = parse_embedding_response(raw_response, expected_count=2)

    def test_percentile_uses_nearest_rank(self) -> None:
        values = [0.4, 0.1, 0.3, 0.2]

        self.assertEqual(percentile(values, 50), 0.2)
        self.assertEqual(percentile(values, 95), 0.4)

    def test_line_chunks_respect_character_limit(self) -> None:
        text = "".join(f"line {index:03d} with benchmark content\n" for index in range(20))

        chunks = line_chunks(text, target_chars=TEST_TEXT_CHARS, overlap_lines=1)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= TEST_TEXT_CHARS for chunk in chunks))
        self.assertTrue(any("line 000" in chunk for chunk in chunks))

    def test_line_chunks_trim_oversized_overlap(self) -> None:
        text = "\n".join(["a" * 90, "b" * 90, "c" * 90, "d" * 90])

        chunks = line_chunks(text, target_chars=TEST_TEXT_CHARS, overlap_lines=4)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= TEST_TEXT_CHARS for chunk in chunks))

    def test_repository_texts_load_supported_files(self) -> None:
        with self.subTest("repository chunks"), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _ = (root / "module.py").write_text("def demo():\n    return 1\n", encoding="utf-8")
            _ = (root / ".git").mkdir()
            _ = (root / ".git" / "ignored.py").write_text("secret\n", encoding="utf-8")

            texts = repository_texts(root, target_chars=TEST_TEXT_CHARS, max_texts=10)

        self.assertEqual(texts, ["def demo():\n    return 1"])

    def test_batch_for_run_cycles_repository_texts(self) -> None:
        batch = batch_for_run(["a", "bb", "ccc"], batch_size=4, run_index=1)

        self.assertEqual(batch, ["bb", "ccc", "a", "bb"])

    def test_text_stats_summarize_lengths(self) -> None:
        stats = text_stats(["a", "bb", "ccc", "dddd"])

        self.assertEqual(stats.count, 4)
        self.assertEqual(stats.min_chars, 1)
        self.assertEqual(stats.p50_chars, 2)
        self.assertEqual(stats.p95_chars, 4)
        self.assertEqual(stats.max_chars, 4)

    def test_result_json_includes_power_measurements(self) -> None:
        result = BenchmarkResult(
            endpoint="http://127.0.0.1:18081/v1/embeddings",
            model="local",
            response_model="local",
            input_source="synthetic",
            input_stats=text_stats(["x" * 80, "y" * 80]),
            min_duration_seconds=0.0,
            batch_size=2,
            runs=2,
            warmup=1,
            text_chars=80,
            total_texts=4,
            total_chars=320,
            total_seconds=2.0,
            request_seconds=[1.0, 1.0],
            vector_dimensions=384,
            response_bytes=100,
            power_measurements=[
                PowerMeasurement(
                    label="amdgpu:power1_average",
                    source="/sys/class/hwmon/hwmon0/power1_average",
                    source_type="power_uw",
                    elapsed_seconds=2.0,
                    energy_joules=20.0,
                    average_watts=10.0,
                    samples=3,
                )
            ],
        )

        payload = result_json(result)
        power = cast("list[JsonObject]", payload["power"])

        self.assertIsInstance(power, list)
        self.assertEqual(power[0]["average_watts"], 10.0)
        self.assertEqual(power[0]["joules_per_text"], 5.0)
        self.assertEqual(power[0]["joules_per_kchar"], 62.5)
        self.assertEqual(power[0]["texts_per_joule"], 0.2)
        self.assertEqual(power[0]["kchars_per_joule"], 0.016)


if __name__ == "__main__":
    _ = unittest.main()
