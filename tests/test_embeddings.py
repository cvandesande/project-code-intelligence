from __future__ import annotations

import json
import unittest
from typing import TYPE_CHECKING
from unittest.mock import patch

from project_code_intelligence import db
from project_code_intelligence.embeddings import (
    EmbeddingBackend,
    EmbeddingEndpointUnavailableError,
    EmbeddingRunConfig,
    embed_items_with_retry,
    embedding_input_text,
    endpoint_host_is_loopback,
    parse_embedding_items,
    smaller_embedding_max_chars,
    validate_embedding_endpoint,
    vector_literals_from_items,
)

if TYPE_CHECKING:
    from project_code_intelligence.models import JsonObject


def retry_values(item: str) -> JsonObject:
    return {"item": item}


class EmbeddingContractTests(unittest.TestCase):
    def test_embedding_input_text_truncates_with_audit_marker(self) -> None:
        self.assertEqual(embedding_input_text("short", 10), "short")

        truncated = embedding_input_text("abcdef" * 20, 90)

        self.assertLessEqual(len(truncated), 90)
        self.assertIn("[embedding input truncated from 120 chars", truncated)

    def test_smaller_embedding_max_chars_respects_retry_floor(self) -> None:
        with patch("project_code_intelligence.embeddings.embedding_retry_min_chars", return_value=800):
            self.assertEqual(smaller_embedding_max_chars(3000), 1500)
            self.assertEqual(smaller_embedding_max_chars(1000), 800)
            self.assertIsNone(smaller_embedding_max_chars(800))

    def test_endpoint_host_policy_handles_loopback_and_remote_hosts(self) -> None:
        self.assertTrue(endpoint_host_is_loopback("localhost"))
        self.assertTrue(endpoint_host_is_loopback("127.0.0.1"))
        self.assertTrue(endpoint_host_is_loopback("::1"))
        self.assertFalse(endpoint_host_is_loopback("embedding.example.invalid"))

        _ = validate_embedding_endpoint("http://[::1]:18081/v1/embeddings", env={})
        with self.assertRaises(ValueError):
            _ = validate_embedding_endpoint("http:///v1/embeddings", env={})
        with self.assertRaises(ValueError):
            _ = validate_embedding_endpoint("https://embedding.example.invalid/v1/embeddings", env={})
        _ = validate_embedding_endpoint(
            "https://embedding.example.invalid/v1/embeddings",
            env={"PROJECT_CODE_INTELLIGENCE_ALLOW_REMOTE_EMBEDDING": "1"},
        )

    def test_parse_embedding_items_requires_openai_compatible_response_shape(self) -> None:
        raw_response = json.dumps({
            "model": "demo",
            "data": [
                {"index": 1, "embedding": [0.3, 0.4]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ],
            "usage": {"total_tokens": 10},
        })

        items, response = parse_embedding_items("http://127.0.0.1:18081/v1/embeddings", raw_response, 2)

        self.assertEqual(response["model"], "demo")
        self.assertEqual(
            vector_literals_from_items("http://127.0.0.1:18081/v1/embeddings", items),
            [
                "[0.1,0.2]",
                "[0.3,0.4]",
            ],
        )

    def test_parse_embedding_items_rejects_bad_response_shapes(self) -> None:
        endpoint = "http://127.0.0.1:18081/v1/embeddings"

        with self.assertRaises(EmbeddingEndpointUnavailableError):
            _ = parse_embedding_items(endpoint, "[]", 1)
        with self.assertRaises(EmbeddingEndpointUnavailableError):
            _ = parse_embedding_items(endpoint, json.dumps({"data": [{"embedding": [1.0]}]}), 2)
        with self.assertRaises(EmbeddingEndpointUnavailableError):
            _ = parse_embedding_items(endpoint, json.dumps({"data": [1]}), 1)
        with self.assertRaises(EmbeddingEndpointUnavailableError):
            _ = vector_literals_from_items(endpoint, [{"index": 0}])

    def test_embed_items_with_retry_splits_recoverable_batch_errors(self) -> None:
        run_config = EmbeddingRunConfig(
            backend=EmbeddingBackend(endpoint=None, endpoint_model="local", use_llama_cli=True),
            max_chars=1200,
        )
        skipped: list[tuple[str, str, int]] = []

        def skip_item(item: str, reason: BaseException, max_chars: int) -> None:
            skipped.append((item, str(reason), max_chars))

        with patch(
            "project_code_intelligence.embeddings.embed_texts_once",
            side_effect=[
                EmbeddingEndpointUnavailableError("batch too large", recoverable_batch=True),
                [db.vector_literal([1.0])],
                [db.vector_literal([2.0]), db.vector_literal([3.0])],
            ],
        ):
            embedded, skipped_count = embed_items_with_retry(
                ["a", "b", "c"],
                run_config=run_config,
                text_for=lambda item: item,
                skip_item=skip_item,
                retry_event_values=retry_values,
            )

        self.assertEqual(embedded, [("a", "[1]"), ("b", "[2]"), ("c", "[3]")])
        self.assertEqual(skipped_count, 0)
        self.assertEqual(skipped, [])

    def test_embed_items_with_retry_marks_single_unrecoverable_item_skipped(self) -> None:
        run_config = EmbeddingRunConfig(
            backend=EmbeddingBackend(endpoint=None, endpoint_model="local", use_llama_cli=True),
            max_chars=800,
        )
        skipped: list[tuple[str, str, int]] = []

        def skip_item(item: str, reason: BaseException, max_chars: int) -> None:
            skipped.append((item, str(reason), max_chars))

        with patch(
            "project_code_intelligence.embeddings.embed_texts_once",
            side_effect=EmbeddingEndpointUnavailableError("context exceeded", recoverable_batch=True),
        ):
            embedded, skipped_count = embed_items_with_retry(
                ["a"],
                run_config=run_config,
                text_for=lambda item: item,
                skip_item=skip_item,
                retry_event_values=retry_values,
            )

        self.assertEqual(embedded, [])
        self.assertEqual(skipped_count, 1)
        self.assertEqual(skipped, [("a", "context exceeded", 800)])


if __name__ == "__main__":
    _ = unittest.main()
