from __future__ import annotations

import json
import os
import queue
import unittest
import urllib.error
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

from typing_extensions import override

import project_code_intelligence.embedding.preembedding as _preembedding
import project_code_intelligence.embedding.utils as _embedding_utils
from project_code_intelligence import db, process
from project_code_intelligence.embedding.endpoint import clear_embedding_endpoint_framework_cache
from project_code_intelligence.embeddings import (
    EmbeddingBackend,
    EmbeddingEndpointUnavailableError,
    EmbeddingRunConfig,
    embed_items_with_retry,
    embed_record_batch,
    embed_with_endpoint,
    embedding_contract,
    embedding_contract_from_metadata,
    embedding_input_text,
    embedding_metadata,
    endpoint_host_is_loopback,
    parse_embedding_items,
    require_compatible_embedding_contract,
    resolve_embedding_endpoint_framework,
    resolve_embedding_endpoint_model,
    smaller_embedding_max_chars,
    validate_embedding_endpoint,
    vector_literal_dimensions,
    vector_literals_from_items,
)
from project_code_intelligence.models import IntelRecord, Snapshot
from project_code_intelligence.runtime import PreEmbeddingResult, PreEmbeddingState
from project_code_intelligence.storage import RecordInsertContext

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from project_code_intelligence.models import JsonObject


def retry_values(item: str) -> JsonObject:
    return {"item": item}


def make_record(
    record_id: str,
    *,
    record_type: str = "code_chunk",
    embedding: str | None = None,
    metadata: JsonObject | None = None,
) -> IntelRecord:
    return IntelRecord(
        collection="test",
        source_path=f"src/{record_id}.py",
        language="python",
        file_role="source",
        content_class="source",
        record_type=record_type,
        record_id=record_id,
        title=record_id,
        summary="python chunk",
        embedding_text=f"type: {record_type}\ncontent:\n{record_id}",
        display_content=record_id,
        metadata={} if metadata is None else metadata,
        embedding=embedding,
    )


def insert_context() -> RecordInsertContext:
    snapshot = Snapshot(
        collection="test",
        repo="demo",
        repo_role="project",
        branch="main",
        commit_sha="abc123",
        tree_sha="tree123",
        dirty=False,
    )
    return RecordInsertContext(
        conn=cast("db.DbConnection", object()),
        snapshot=snapshot,
        snapshot_id=7,
        file_ids={},
        file_hashes={},
    )


class FakeThread:
    def __init__(self, target: object, *, kwargs: dict[str, object], daemon: bool) -> None:
        self.target = target
        self.kwargs = kwargs
        self.daemon = daemon
        self.started = False

    def start(self) -> None:
        self.started = True

    @staticmethod
    def is_alive() -> bool:
        return False


class LlamaEmbeddingUtilsTests(unittest.TestCase):
    def test_llama_build_command_requires_model_or_default_and_preserves_extra_args(self) -> None:
        prompt = Path("prompt.txt")

        with (
            patch.dict(
                os.environ,
                {"PROJECT_CODE_INTELLIGENCE_LLAMA_EMBEDDING_BIN": "/opt/llama/bin/embed"},
                clear=True,
            ),
            self.assertRaisesRegex(RuntimeError, "PROJECT_CODE_INTELLIGENCE_LLAMA_MODEL"),
        ):
            _ = _embedding_utils.llama.build_command(prompt)

        with patch.dict(
            os.environ,
            {
                "PROJECT_CODE_INTELLIGENCE_LLAMA_EMBEDDING_BIN": "/opt/llama/bin/embed",
                "PROJECT_CODE_INTELLIGENCE_LLAMA_MODEL": "/models/embed.gguf",
                "PROJECT_CODE_INTELLIGENCE_LLAMA_EXTRA_ARGS": "--threads 4 --ctx-size 8192",
            },
            clear=True,
        ):
            command = _embedding_utils.llama.build_command(prompt)

        self.assertEqual(
            command,
            [
                "/opt/llama/bin/embed",
                "--embd-output-format",
                "json",
                "-f",
                "prompt.txt",
                "--model",
                "/models/embed.gguf",
                "--threads",
                "4",
                "--ctx-size",
                "8192",
            ],
        )

        with patch.dict(
            os.environ,
            {
                "PROJECT_CODE_INTELLIGENCE_LLAMA_CPP_DIR": "/opt/llama",
                "PROJECT_CODE_INTELLIGENCE_LLAMA_EMBD_GEMMA_DEFAULT": "1",
            },
            clear=True,
        ):
            default_command = _embedding_utils.llama.build_command(prompt)

        self.assertEqual(default_command[0], "/opt/llama/llama-embedding")
        self.assertIn("--embd-gemma-default", default_command)

    def test_llama_parse_stdout_extracts_final_json_payload(self) -> None:
        self.assertEqual(
            _embedding_utils.llama.parse_llama_stdout('llama log line\n{"embedding":[1,2]}'),
            {"embedding": [1, 2]},
        )
        self.assertEqual(_embedding_utils.llama.parse_llama_stdout("warning\n[1, 2, 3]"), [1, 2, 3])

        with self.assertRaisesRegex(ValueError, "empty stdout"):
            _ = _embedding_utils.llama.parse_llama_stdout(" \n")
        with self.assertRaisesRegex(ValueError, "could not find JSON payload"):
            _ = _embedding_utils.llama.parse_llama_stdout("no json here")
        with self.assertRaises(json.JSONDecodeError):
            _ = _embedding_utils.llama.parse_llama_stdout('log\n{"embedding":')

    def test_llama_extract_embedding_accepts_known_shapes_and_rejects_bad_values(self) -> None:
        self.assertEqual(_embedding_utils.llama.extract_embedding([1, 2.5]), [1.0, 2.5])
        self.assertEqual(_embedding_utils.llama.extract_embedding([[3, 4]]), [3.0, 4.0])
        self.assertEqual(_embedding_utils.llama.extract_embedding({"embedding": [5, 6]}), [5.0, 6.0])
        self.assertEqual(_embedding_utils.llama.extract_embedding({"data": [{"embedding": [7, 8]}]}), [7.0, 8.0])

        bad_payloads: list[object] = [
            {"data": []},
            {"embedding": []},
            {"embedding": [True]},
            {"embedding": ["bad"]},
            "bad",
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload), self.assertRaises((TypeError, ValueError)):
                _ = _embedding_utils.llama.extract_embedding(payload)

    def test_llama_batch_embeddings_batches_prompts_and_vectors(self) -> None:
        commands: list[list[str]] = []
        payloads: list[str] = []
        run_envs: list[dict[str, str]] = []

        def fake_build_command(prompt_file: Path) -> list[str]:
            return ["llama-embed", str(prompt_file)]

        def fake_run(
            command: Sequence[str],
            options: process.RunOptions | None = None,
        ) -> process.CompletedProcess[str]:
            if options is None:
                raise AssertionError("llama embedding should pass run options")
            command_list = list(command)
            commands.append(command_list)
            payloads.append(Path(command_list[1]).read_text(encoding="utf-8"))
            if options.env is None:
                raise AssertionError("llama embedding should pass an environment")
            run_envs.append(dict(options.env))
            self.assertTrue(options.check)
            self.assertTrue(options.capture_output)
            self.assertEqual(options.timeout, 12)
            return process.CompletedProcess(command_list, 0, stdout=f"stdout-{len(commands)}")

        def fake_parse(stdout: str) -> JsonObject:
            payload_by_stdout: dict[str, JsonObject] = {
                "stdout-1": {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]},
                "stdout-2": {"data": [{"embedding": [0.5, 0.6]}]},
            }
            return payload_by_stdout[stdout]

        with (
            patch.dict(os.environ, {"LD_LIBRARY_PATH": "/existing"}, clear=True),
            patch.object(_embedding_utils.llama, "build_command", side_effect=fake_build_command),
            patch.object(_embedding_utils.llama, "llama_dir", return_value=Path("/opt/llama")),
            patch.object(_embedding_utils.llama, "llama_timeout_seconds", return_value=12),
            patch.object(_embedding_utils.llama, "parse_llama_stdout", side_effect=fake_parse),
            patch.object(_embedding_utils.process, "run", side_effect=fake_run),
        ):
            vectors = _embedding_utils.llama_batch_embeddings(["alpha", "beta", "gamma"], batch_size=2)

        separator = "<#project-code-intelligence-embedding-separator#>"
        self.assertEqual(vectors, ["[0.1,0.2]", "[0.3,0.4]", "[0.5,0.6]"])
        self.assertEqual(payloads, [f"alpha\n{separator}\nbeta", "gamma"])
        self.assertEqual(commands[0][-2:], ["--embd-separator", separator])
        self.assertEqual(commands[1][-2:], ["--embd-separator", separator])
        self.assertEqual(run_envs[0]["LD_LIBRARY_PATH"], "/opt/llama:/existing")

    def test_llama_batch_embeddings_rejects_invalid_inputs_and_response_shape(self) -> None:
        def fake_build_command(prompt_file: Path) -> list[str]:
            return ["llama-embed", str(prompt_file)]

        with self.assertRaisesRegex(ValueError, "batch_size"):
            _ = _embedding_utils.llama_batch_embeddings(["alpha"], batch_size=0)
        with self.assertRaisesRegex(ValueError, "embedding separator"):
            _ = _embedding_utils.llama_batch_embeddings(
                ["alpha <#project-code-intelligence-embedding-separator#> beta"],
                batch_size=1,
            )

        with (
            patch.object(
                _embedding_utils.llama,
                "build_command",
                side_effect=fake_build_command,
            ),
            patch.object(_embedding_utils.llama, "llama_dir", return_value=Path("/opt/llama")),
            patch.object(_embedding_utils.llama, "llama_timeout_seconds", return_value=12),
            patch.object(_embedding_utils.llama, "parse_llama_stdout", return_value={"data": []}),
            patch.object(
                _embedding_utils.process,
                "run",
                return_value=process.CompletedProcess(["llama-embed"], 0, stdout="stdout"),
            ),
            self.assertRaisesRegex(ValueError, "returned 0 embeddings for a batch of 1 chunks"),
        ):
            _ = _embedding_utils.llama_batch_embeddings(["alpha"], batch_size=1)

    def test_llama_batch_embeddings_includes_stderr_tail_on_process_failure(self) -> None:
        long_stderr = "prefix-" + ("x" * 1300)

        def fake_build_command(prompt_file: Path) -> list[str]:
            return ["llama-embed", str(prompt_file)]

        with (
            patch.object(
                _embedding_utils.llama,
                "build_command",
                side_effect=fake_build_command,
            ),
            patch.object(_embedding_utils.llama, "llama_dir", return_value=Path("/opt/llama")),
            patch.object(_embedding_utils.llama, "llama_timeout_seconds", return_value=12),
            patch.object(
                _embedding_utils.process,
                "run",
                side_effect=process.CalledProcessError(1, ["llama-embed"], stderr=long_stderr),
            ),
            self.assertRaisesRegex(RuntimeError, "llama.cpp embedding failed"),
        ):
            _ = _embedding_utils.llama_batch_embeddings(["alpha"], batch_size=1)


class EmbeddingContractTests(unittest.TestCase):
    def test_embedding_input_text_truncates_with_audit_marker(self) -> None:
        self.assertEqual(embedding_input_text("short", 10), "short")

        truncated = embedding_input_text("abcdef" * 20, 90)

        self.assertLessEqual(len(truncated), 90)
        self.assertIn("[embedding input truncated from 120 chars", truncated)

    def test_smaller_embedding_max_chars_respects_retry_floor(self) -> None:
        with patch("project_code_intelligence.embedding.core.embedding_retry_min_chars", return_value=800):
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

    def test_resolve_embedding_endpoint_model_uses_local_health_model(self) -> None:
        with patch(
            "project_code_intelligence.embedding.endpoint.http_client.read_text",
            return_value=json.dumps({"model": "jinaai/jina-embeddings-v2-base-code"}),
        ):
            model = resolve_embedding_endpoint_model(
                "http://127.0.0.1:18082/v1/embeddings",
                "local",
                env={},
            )

        self.assertEqual(model, "jinaai/jina-embeddings-v2-base-code")

    def test_resolve_embedding_endpoint_model_probes_lemonade_when_healthz_is_not_json(self) -> None:
        def read_text(request_or_url: object, *, timeout: float) -> str:
            _ = timeout
            if isinstance(request_or_url, str):
                if request_or_url.endswith("/v1/models"):
                    return json.dumps({"data": []})
                return "<html>Lemonade</html>"
            data = getattr(request_or_url, "data", b"")
            payload_value = cast("object", json.loads(data.decode("utf-8")))
            if not isinstance(payload_value, dict):
                raise TypeError("expected object payload")
            payload = cast("JsonObject", payload_value)
            if payload["model"] == "embed-gemma-300m-FLM":
                return json.dumps({
                    "model": "embed-gemma-300m-FLM",
                    "data": [{"index": 0, "embedding": [0.1, 0.2]}],
                })
            raise urllib.error.URLError("not found")

        with patch("project_code_intelligence.embedding.endpoint.http_client.read_text", side_effect=read_text):
            model = resolve_embedding_endpoint_model(
                "http://127.0.0.1:18084/v1/embeddings",
                "local",
                env={},
            )

        self.assertEqual(model, "embed-gemma-300m-FLM")

    def test_resolve_embedding_endpoint_model_prefers_single_listed_model(self) -> None:
        def read_text(request_or_url: object, *, timeout: float) -> str:
            _ = timeout
            if isinstance(request_or_url, str) and request_or_url.endswith("/v1/models"):
                return json.dumps({
                    "data": [
                        {
                            "id": "Qwen3-Embedding-0.6B-Q8_0.gguf",
                            "object": "model",
                            "owned_by": "llamacpp",
                        }
                    ]
                })
            if isinstance(request_or_url, str):
                return json.dumps({"error": {"message": "File Not Found"}})
            raise AssertionError("listed model should avoid embedding probes")

        with patch("project_code_intelligence.embedding.endpoint.http_client.read_text", side_effect=read_text):
            model = resolve_embedding_endpoint_model(
                "http://127.0.0.1:18085/v1/embeddings",
                "local",
                env={},
            )

        self.assertEqual(model, "Qwen3-Embedding-0.6B-Q8_0.gguf")

    def test_resolve_embedding_endpoint_model_respects_explicit_model(self) -> None:
        with patch("project_code_intelligence.embedding.endpoint.http_client.read_text") as read_text:
            model = resolve_embedding_endpoint_model(
                "http://127.0.0.1:18083/v1/embeddings",
                "local",
                env={"PROJECT_CODE_INTELLIGENCE_EMBEDDING_ENDPOINT_MODEL": "local"},
            )

        self.assertEqual(model, "local")
        read_text.assert_not_called()

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

    def test_vector_literal_dimensions_counts_pgvector_items(self) -> None:
        self.assertEqual(vector_literal_dimensions("[0.1,0.2,0.3]"), 3)

        with self.assertRaises(ValueError):
            _ = vector_literal_dimensions("0.1,0.2")

    def test_embedding_metadata_records_model_and_dimensions(self) -> None:
        run_config = EmbeddingRunConfig(
            backend=EmbeddingBackend(
                endpoint="http://127.0.0.1:18081/v1/embeddings", endpoint_model="demo", use_llama_cli=False
            ),
            max_chars=800,
        )

        self.assertEqual(
            embedding_metadata(run_config, "[0.1,0.2]"),
            {
                "embedding_backend": "endpoint",
                "embedding_model": "demo",
                "embedding_dimensions": 2,
            },
        )
        self.assertEqual(
            embedding_contract(run_config, "[0.1,0.2]"),
            {"version": 1, "backend": "endpoint", "model": "demo", "dimensions": 2},
        )

    def test_preembedding_records_same_metadata_contract_as_post_insert_embedding(self) -> None:
        record = IntelRecord(
            collection="test",
            source_path="src/main.py",
            language="python",
            file_role="source",
            content_class="source",
            record_type="code_chunk",
            record_id="src/main.py::chunk::000001-000002",
            parent_record_id=None,
            title="src/main.py:1-2",
            summary="python chunk",
            embedding_text="type: code_chunk\ncontent:\ndef main(): pass",
            display_content="# src/main.py:1-2",
            line_start=1,
            line_end=2,
            metadata={"project": "demo"},
        )
        run_config = EmbeddingRunConfig(
            backend=EmbeddingBackend(
                endpoint="http://127.0.0.1:18081/v1/embeddings", endpoint_model="demo-model", use_llama_cli=False
            ),
            max_chars=800,
        )

        with patch(
            "project_code_intelligence.embedding.preembedding.embed_items_with_retry",
            return_value=([(record, "[0.1,0.2,0.3]")], 0),
        ):
            embedded, skipped = embed_record_batch([record], run_config=run_config)

        self.assertEqual((embedded, skipped), (1, 0))
        self.assertEqual(record.embedding, "[0.1,0.2,0.3]")
        self.assertEqual(
            record.metadata,
            {
                "project": "demo",
                "embedding_backend": "endpoint",
                "embedding_model": "demo-model",
                "embedding_dimensions": 3,
            },
        )

    def test_start_record_preembedding_selects_only_embedding_eligible_records(self) -> None:
        selected = make_record("selected")
        wrong_type = make_record("file-record", record_type="file")
        already_embedded = make_record("embedded", embedding="[0.1]")
        skipped = make_record("skipped", metadata={"embedding_skipped": True})
        run_config = EmbeddingRunConfig(
            backend=EmbeddingBackend(
                endpoint="http://127.0.0.1:18081/v1/embeddings", endpoint_model="demo-model", use_llama_cli=False
            ),
            max_chars=800,
        )

        with (
            patch.object(_preembedding.threading, "Thread", FakeThread),
            patch.object(_preembedding, "preembedding_ahead_batches", return_value=2),
            patch.object(_preembedding, "progress_event") as progress_event,
            patch.object(_preembedding.runtime_state.active_metrics, "add") as metrics_add,
        ):
            state = _preembedding.start_record_preembedding(
                [selected, wrong_type, already_embedded, skipped],
                record_types={"code_chunk"},
                batch_size=0,
                run_config=run_config,
            )

        if state is None:
            raise AssertionError("expected preembedding state")
        self.assertEqual(state.batches, [[selected]])
        self.assertEqual(state.selected_ids, {id(selected)})
        self.assertEqual(state.total_records, 1)
        self.assertEqual(state.results.maxsize, 2)
        self.assertIsInstance(state.thread, FakeThread)
        self.assertTrue(cast("FakeThread", state.thread).started)
        progress_event.assert_called_once()
        metrics_add.assert_called_once_with("preembedding_records_selected", 1)

    def test_insert_records_with_preembedding_inserts_ready_and_deferred_batches(self) -> None:
        non_embedding = make_record("file-record", record_type="file")
        ready = make_record("ready")
        deferred = make_record("deferred")
        state = PreEmbeddingState(
            batches=[[ready], [deferred]],
            selected_ids={id(ready), id(deferred)},
            results=queue.Queue[PreEmbeddingResult](),
            total_records=2,
        )
        state.results.put(PreEmbeddingResult(batch=[ready], embedded=1, skipped=0))
        inserted_batches: list[list[str]] = []

        def fake_insert_records(
            context: RecordInsertContext,
            records: list[IntelRecord],
            progress_fn: Callable[[int], None] | None = None,
        ) -> int:
            _ = context
            if progress_fn is not None:
                progress_fn(len(records))
            inserted_batches.append([record.record_id for record in records])
            return len(records)

        progress: list[int] = []

        with (
            patch.object(_preembedding, "insert_records", side_effect=fake_insert_records),
            patch.object(_preembedding, "progress_event") as progress_event,
        ):
            inserted, embedded, skipped = _preembedding.insert_records_with_preembedding(
                insert_context(),
                [non_embedding, ready, deferred],
                state,
                progress_fn=progress.append,
            )

        self.assertEqual((inserted, embedded, skipped), (3, 1, 0))
        self.assertEqual(inserted_batches, [["file-record"], ["ready"], ["deferred"]])
        self.assertEqual(progress, [1, 1, 1])
        self.assertEqual(state.consumed_batches, 2)
        self.assertTrue(state.cancel_event.is_set())
        self.assertEqual(progress_event.call_args_list[-1].args[0], "code_intel_preembedding_deferred")

    def test_consume_preembedding_results_falls_back_to_unembedded_insert_on_error(self) -> None:
        failed = make_record("failed")
        remaining = make_record("remaining")
        state = PreEmbeddingState(
            batches=[[failed], [remaining]],
            selected_ids={id(failed), id(remaining)},
            results=queue.Queue[PreEmbeddingResult](),
            total_records=2,
        )
        state.results.put(PreEmbeddingResult(batch=[failed], error=RuntimeError("endpoint failed")))
        inserted_batches: list[list[str]] = []

        def fake_insert_records(
            context: RecordInsertContext,
            records: list[IntelRecord],
            progress_fn: Callable[[int], None] | None = None,
        ) -> int:
            _ = context
            if progress_fn is not None:
                progress_fn(len(records))
            inserted_batches.append([record.record_id for record in records])
            return len(records)

        progress: list[int] = []

        with (
            patch.object(_preembedding, "insert_records", side_effect=fake_insert_records),
            patch.object(_preembedding, "progress_event") as progress_event,
        ):
            inserted = _preembedding.consume_preembedding_results(
                insert_context(), state, block=False, progress_fn=progress.append
            )

        self.assertEqual(inserted, 2)
        self.assertEqual(inserted_batches, [["failed", "remaining"]])
        self.assertEqual(progress, [2])
        self.assertTrue(state.cancel_event.is_set())
        self.assertEqual(state.consumed_batches, 2)
        progress_event.assert_called_once()
        self.assertEqual(progress_event.call_args.args[0], "code_intel_preembedding_disabled")

    def test_snapshot_embedding_contract_requires_same_model_and_dimensions(self) -> None:
        existing = {"version": 1, "backend": "endpoint", "model": "embed-gemma-300m-FLM", "dimensions": 768}
        compatible = {"version": 1, "backend": "endpoint", "model": "embed-gemma-300m-FLM", "dimensions": 768}
        incompatible_model = {
            "version": 1,
            "backend": "endpoint",
            "model": "Qwen3-Embedding-0.6B-Q8_0.gguf",
            "dimensions": 768,
        }
        incompatible_dimensions = {
            "version": 1,
            "backend": "endpoint",
            "model": "embed-gemma-300m-FLM",
            "dimensions": 4096,
        }

        require_compatible_embedding_contract(existing, compatible)
        with self.assertRaises(ValueError):
            require_compatible_embedding_contract(existing, incompatible_model)
        with self.assertRaises(ValueError):
            require_compatible_embedding_contract(existing, incompatible_dimensions)

    def test_snapshot_embedding_contract_parses_snapshot_metadata(self) -> None:
        metadata = {"embedding_contract": {"version": 1, "backend": "endpoint", "model": "demo", "dimensions": 3}}

        self.assertEqual(
            embedding_contract_from_metadata(metadata),
            {"version": 1, "backend": "endpoint", "model": "demo", "dimensions": 3},
        )
        self.assertIsNone(embedding_contract_from_metadata({}))

        with self.assertRaises(ValueError):
            _ = embedding_contract_from_metadata({"embedding_contract": "demo"})

    def test_embed_with_endpoint_retries_transient_endpoint_errors(self) -> None:
        raw_response = json.dumps({"data": [{"index": 0, "embedding": [0.1, 0.2]}]})

        with (
            patch(
                "project_code_intelligence.embedding.endpoint.read_embedding_response",
                side_effect=[EmbeddingEndpointUnavailableError("temporary endpoint failure"), raw_response],
            ) as read_response,
            patch("project_code_intelligence.embedding.endpoint.embedding_request_retries", return_value=1),
            patch("time.sleep"),
        ):
            vectors = embed_with_endpoint(
                "http://127.0.0.1:18081/v1/embeddings",
                ["hello"],
                "demo",
                track_metrics=False,
            )

        self.assertEqual(vectors, ["[0.1,0.2]"])
        self.assertEqual(read_response.call_count, 2)

    def test_embed_items_with_retry_splits_recoverable_batch_errors(self) -> None:
        run_config = EmbeddingRunConfig(
            backend=EmbeddingBackend(endpoint=None, endpoint_model="local", use_llama_cli=True),
            max_chars=1200,
        )
        skipped: list[tuple[str, str, int]] = []

        def skip_item(item: str, reason: BaseException, max_chars: int) -> None:
            skipped.append((item, str(reason), max_chars))

        with patch(
            "project_code_intelligence.embedding.core.embed_texts_once",
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
            "project_code_intelligence.embedding.core.embed_texts_once",
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


class ResolveEmbeddingEndpointFrameworkTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        clear_embedding_endpoint_framework_cache()

    def test_returns_apple_mlx_from_healthz(self) -> None:
        with patch(
            "project_code_intelligence.embedding.endpoint.http_client.read_text",
            return_value=json.dumps({
                "ok": True,
                "model": "mlx-community/Qwen3-Embedding-0.6B-8bit",
                "framework": "Apple MLX",
            }),
        ):
            framework = resolve_embedding_endpoint_framework("http://127.0.0.1:18091/v1/embeddings")

        self.assertEqual(framework, "Apple MLX")

    def test_returns_fastembed_cpu_from_healthz(self) -> None:
        with patch(
            "project_code_intelligence.embedding.endpoint.http_client.read_text",
            return_value=json.dumps({
                "ok": True,
                "model": "jinaai/jina-embeddings-v2-base-code",
                "framework": "Fastembed CPU",
            }),
        ):
            framework = resolve_embedding_endpoint_framework("http://127.0.0.1:18092/v1/embeddings")

        self.assertEqual(framework, "Fastembed CPU")

    def test_returns_none_when_healthz_omits_field(self) -> None:
        with patch(
            "project_code_intelligence.embedding.endpoint.http_client.read_text",
            return_value=json.dumps({"ok": True, "model": "some-third-party-model"}),
        ):
            framework = resolve_embedding_endpoint_framework("http://127.0.0.1:18093/v1/embeddings")

        self.assertIsNone(framework)

    def test_returns_none_for_remote_endpoint_without_probe(self) -> None:
        with patch(
            "project_code_intelligence.embedding.endpoint.http_client.read_text",
        ) as read_text:
            framework = resolve_embedding_endpoint_framework("https://f5ai.pd.f5net.com/v1/embeddings")

        self.assertIsNone(framework)
        read_text.assert_not_called()

    def test_returns_none_when_endpoint_is_unreachable(self) -> None:
        with patch(
            "project_code_intelligence.embedding.endpoint.http_client.read_text",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            framework = resolve_embedding_endpoint_framework("http://127.0.0.1:18094/v1/embeddings")

        self.assertIsNone(framework)

    def test_caches_repeated_calls(self) -> None:
        with patch(
            "project_code_intelligence.embedding.endpoint.http_client.read_text",
            return_value=json.dumps({"framework": "Apple MLX", "model": "x"}),
        ) as read_text:
            first = resolve_embedding_endpoint_framework("http://127.0.0.1:18095/v1/embeddings")
            second = resolve_embedding_endpoint_framework("http://127.0.0.1:18095/v1/embeddings")

        self.assertEqual(first, "Apple MLX")
        self.assertEqual(second, "Apple MLX")
        self.assertEqual(read_text.call_count, 1)


if __name__ == "__main__":
    _ = unittest.main()
