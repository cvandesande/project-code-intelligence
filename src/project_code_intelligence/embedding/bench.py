"""Benchmark OpenAI-compatible embedding endpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from project_code_intelligence import config, http_client, power
from project_code_intelligence.embeddings import (
    http_error_detail,
    resolve_embedding_endpoint_model,
    validate_embedding_endpoint,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from project_code_intelligence.models import JsonObject

DEFAULT_ENDPOINT = config.DEFAULT_FASTEMBED_EMBEDDING_ENDPOINT
MIN_REPOSITORY_TEXT_CHARS = 100


def write_stdout(message: str = "") -> None:
    _ = sys.stdout.write(message + "\n")


def write_stderr(message: str) -> None:
    _ = sys.stderr.write(message + "\n")


CODE_TEMPLATES = (
    """def resolve_route(packet: Packet, table: RoutingTable) -> Route | None:
    for prefix, route in table.entries:
        if packet.destination in prefix and route.enabled:
            return route
    return None
""",
    """static int device_probe(struct platform_device *pdev)
{
    struct device *dev = &pdev->dev;
    return devm_request_irq(dev, irq, handler, 0, "demo", dev);
}
""",
    """async function refreshIndex(client, workspace) {
  const snapshot = await client.createSnapshot(workspace);
  await client.uploadRecords(snapshot.records);
  return snapshot.id;
}
""",
    """package indexer

func InsertBatch(ctx context.Context, db *sql.DB, rows []Record) error {
    tx, err := db.BeginTx(ctx, nil)
    if err != nil {
        return err
    }
    defer tx.Rollback()
    return tx.Commit()
}
""",
)

REPOSITORY_INPUT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}
REPOSITORY_INPUT_SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "models",
}


@dataclass(frozen=True)
class EmbeddingRequestResult:
    seconds: float
    dimensions: int
    response_model: str | None
    response_bytes: int


@dataclass(frozen=True)
class TextInput:
    source: str
    texts: list[str]


@dataclass(frozen=True)
class TextStats:
    count: int
    min_chars: int
    p50_chars: int
    p95_chars: int
    max_chars: int
    average_chars: float


@dataclass(frozen=True)
class BenchmarkResult:
    endpoint: str
    model: str
    response_model: str | None
    input_source: str
    input_stats: TextStats
    min_duration_seconds: float
    batch_size: int
    runs: int
    warmup: int
    text_chars: int
    total_texts: int
    total_chars: int
    total_seconds: float
    request_seconds: list[float]
    vector_dimensions: int
    response_bytes: int
    power_measurements: list[power.PowerMeasurement]


@dataclass(frozen=True)
class BenchmarkConfig:
    endpoint: str
    model: str
    batch_size: int
    runs: int
    warmup: int
    text_chars: int
    min_duration_seconds: float
    input_root: str | None
    input_max_texts: int
    timeout: float
    power_sensors: list[power.PowerSensorSource]
    power_sample_interval: float


class BenchNamespace(argparse.Namespace):
    endpoint: str
    model: str
    batch_size: int
    runs: int
    warmup: int
    text_chars: int
    timeout: float
    min_duration_seconds: float
    input_root: str | None
    input_max_texts: int
    json_output: bool
    power_enabled: bool
    power_sources: list[str] | None
    power_sample_interval: float
    list_power_sources: bool


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or positive")
    return parsed


def default_endpoint() -> str:
    return config.default_embedding_endpoint(local_default=True) or DEFAULT_ENDPOINT


def default_model() -> str:
    endpoint = default_endpoint()
    return resolve_embedding_endpoint_model(endpoint, config.default_embedding_endpoint_model(endpoint=endpoint))


def sample_text(index: int, target_chars: int) -> str:
    template = CODE_TEMPLATES[index % len(CODE_TEMPLATES)]
    suffix = f"\n# embedding_benchmark_sample={index}\n"
    chunks: list[str] = [suffix]
    while sum(len(chunk) for chunk in chunks) < target_chars:
        chunks.extend((template, suffix))
    return "".join(chunks)[:target_chars]


def generated_texts(count: int, target_chars: int) -> list[str]:
    return [sample_text(index, target_chars) for index in range(count)]


def text_stats(texts: Sequence[str]) -> TextStats:
    if not texts:
        raise ValueError("text statistics require at least one text")
    lengths = [len(text) for text in texts]
    return TextStats(
        count=len(texts),
        min_chars=min(lengths),
        p50_chars=int(percentile(lengths, 50)),
        p95_chars=int(percentile(lengths, 95)),
        max_chars=max(lengths),
        average_chars=sum(lengths) / len(lengths),
    )


def is_candidate_repository_file(path: Path) -> bool:
    if any(part in REPOSITORY_INPUT_SKIP_DIRS for part in path.parts):
        return False
    if path.name == "Dockerfile":
        return True
    return path.suffix.lower() in REPOSITORY_INPUT_SUFFIXES


def line_chunks(text: str, target_chars: int, overlap_lines: int) -> list[str]:
    if target_chars < MIN_REPOSITORY_TEXT_CHARS:
        raise ValueError(f"repository benchmark target chars must be at least {MIN_REPOSITORY_TEXT_CHARS}")
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0
    for line in lines:
        chunk_line = line
        if len(chunk_line) > target_chars:
            chunk_line = chunk_line[: target_chars - 22].rstrip() + " [line truncated]\n"
        line_chars = len(chunk_line)
        if current and current_chars + line_chars > target_chars:
            chunks.append("".join(current).strip())
            overlap = current[-overlap_lines:] if overlap_lines > 0 else []
            while overlap and sum(len(item) for item in overlap) > target_chars:
                _ = overlap.pop(0)
            current = list(overlap)
            current_chars = sum(len(item) for item in current)
            while current and current_chars + line_chars > target_chars:
                removed = current.pop(0)
                current_chars -= len(removed)
        current.append(chunk_line)
        current_chars += line_chars
    if current:
        chunks.append("".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def repository_input_files(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and is_candidate_repository_file(path.relative_to(root))
    ]


def read_repository_input_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    except OSError as exc:
        raise ValueError(f"failed to read repository input file {path}: {exc}") from exc


def repository_texts(root: Path, target_chars: int, max_texts: int) -> list[str]:
    if max_texts <= 0:
        raise ValueError("repository benchmark max texts must be positive")
    root = root.resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"repository input root is not a directory: {root}")
    texts: list[str] = []
    for path in repository_input_files(root):
        if len(texts) >= max_texts:
            break
        content = read_repository_input_text(path)
        if content is None:
            continue
        for chunk in line_chunks(content, target_chars, overlap_lines=4):
            texts.append(chunk[:target_chars])
            if len(texts) >= max_texts:
                break
    if not texts:
        raise ValueError(f"no benchmark input texts found under {root}")
    return texts


def benchmark_input(*, batch_size: int, text_chars: int, input_root: str | None, input_max_texts: int) -> TextInput:
    if input_root:
        root = Path(input_root)
        texts = repository_texts(root, text_chars, input_max_texts)
        return TextInput(source=f"repository:{root}", texts=texts)
    return TextInput(source="synthetic", texts=generated_texts(batch_size, text_chars))


def batch_for_run(texts: Sequence[str], batch_size: int, run_index: int) -> list[str]:
    if not texts:
        raise ValueError("benchmark requires at least one text")
    offset = run_index * batch_size
    return [texts[(offset + index) % len(texts)] for index in range(batch_size)]


def embedding_dimensions(value: object) -> int:
    if not isinstance(value, list):
        raise TypeError("embedding response item missing embedding list")
    dimensions = 0
    for item in cast("list[object]", value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError("embedding response contains a non-numeric vector value")
        dimensions += 1
    if dimensions == 0:
        raise ValueError("embedding response contains an empty vector")
    return dimensions


def parse_embedding_response(raw_response: str, expected_count: int) -> tuple[int, str | None]:
    value = cast("object", json.loads(raw_response))
    if not isinstance(value, dict):
        raise TypeError("embedding API response must be an object")
    data = cast("JsonObject", value)
    response_model_value = data.get("model")
    response_model = response_model_value if isinstance(response_model_value, str) else None
    items_value = data.get("data")
    if not isinstance(items_value, list) or len(items_value) != expected_count:
        raise ValueError("embedding API response has unexpected data length")
    dimensions: int | None = None
    for item_value in cast("list[object]", items_value):
        if not isinstance(item_value, dict):
            raise TypeError("embedding API response items must be objects")
        item = cast("JsonObject", item_value)
        item_dimensions = embedding_dimensions(item.get("embedding"))
        if dimensions is None:
            dimensions = item_dimensions
        elif item_dimensions != dimensions:
            raise ValueError("embedding API response vectors have inconsistent dimensions")
    if dimensions is None:
        raise ValueError("embedding API response did not include vectors")
    return dimensions, response_model


def request_embeddings(endpoint: str, model: str, texts: list[str], timeout: float) -> EmbeddingRequestResult:
    validate_embedding_endpoint(endpoint)
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = config.embedding_api_key(endpoint)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = http_client.request(endpoint, data=payload, headers=headers, method="POST")
    started = time.perf_counter()
    try:
        raw_response = http_client.read_text(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(http_error_detail(exc)) from exc
    except (ConnectionError, TimeoutError, urllib.error.URLError) as exc:
        raise RuntimeError(str(exc)) from exc
    seconds = time.perf_counter() - started
    dimensions, response_model = parse_embedding_response(raw_response, len(texts))
    return EmbeddingRequestResult(
        seconds=seconds,
        dimensions=dimensions,
        response_model=response_model,
        response_bytes=len(raw_response.encode("utf-8")),
    )


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((percent / 100.0) * len(ordered)) - 1))
    return ordered[index]


def run_benchmark(benchmark: BenchmarkConfig) -> BenchmarkResult:
    text_input = benchmark_input(
        batch_size=benchmark.batch_size,
        text_chars=benchmark.text_chars,
        input_root=benchmark.input_root,
        input_max_texts=benchmark.input_max_texts,
    )
    for warmup_index in range(benchmark.warmup):
        _ = request_embeddings(
            benchmark.endpoint,
            benchmark.model,
            batch_for_run(text_input.texts, benchmark.batch_size, warmup_index),
            benchmark.timeout,
        )

    request_seconds: list[float] = []
    vector_dimensions: int | None = None
    response_model: str | None = None
    response_bytes = 0
    total_texts = 0
    total_chars = 0
    power_monitor = None
    if benchmark.power_sensors:
        power_monitor = power.PowerMonitor(benchmark.power_sensors, benchmark.power_sample_interval)
    if power_monitor is not None:
        power_monitor.start()
    total_started = time.perf_counter()
    try:
        while (
            len(request_seconds) < benchmark.runs
            or time.perf_counter() - total_started < benchmark.min_duration_seconds
        ):
            texts = batch_for_run(text_input.texts, benchmark.batch_size, len(request_seconds))
            result = request_embeddings(benchmark.endpoint, benchmark.model, texts, benchmark.timeout)
            if vector_dimensions is None:
                vector_dimensions = result.dimensions
            elif result.dimensions != vector_dimensions:
                raise ValueError("endpoint returned inconsistent vector dimensions between runs")
            response_model = result.response_model or response_model
            response_bytes += result.response_bytes
            request_seconds.append(result.seconds)
            total_texts += len(texts)
            total_chars += sum(len(text) for text in texts)
    finally:
        total_seconds = time.perf_counter() - total_started
        power_measurements = power_monitor.stop() if power_monitor is not None else []
    if vector_dimensions is None:
        raise ValueError("benchmark did not run")
    return BenchmarkResult(
        endpoint=benchmark.endpoint,
        model=benchmark.model,
        response_model=response_model,
        input_source=text_input.source,
        input_stats=text_stats(text_input.texts),
        min_duration_seconds=benchmark.min_duration_seconds,
        batch_size=benchmark.batch_size,
        runs=len(request_seconds),
        warmup=benchmark.warmup,
        text_chars=benchmark.text_chars,
        total_texts=total_texts,
        total_chars=total_chars,
        total_seconds=total_seconds,
        request_seconds=request_seconds,
        vector_dimensions=vector_dimensions,
        response_bytes=response_bytes,
        power_measurements=power_measurements,
    )


def result_json(result: BenchmarkResult) -> JsonObject:
    request_seconds = result.request_seconds
    requests_per_second = result.runs / result.total_seconds
    texts_per_second = result.total_texts / result.total_seconds
    chars_per_second = result.total_chars / result.total_seconds
    payload: JsonObject = {
        "endpoint": result.endpoint,
        "model": result.model,
        "response_model": result.response_model,
        "input_source": result.input_source,
        "input_texts": result.input_stats.count,
        "input_chars": {
            "min": result.input_stats.min_chars,
            "p50": result.input_stats.p50_chars,
            "p95": result.input_stats.p95_chars,
            "max": result.input_stats.max_chars,
            "average": round(result.input_stats.average_chars, 3),
        },
        "min_duration_seconds": result.min_duration_seconds,
        "batch_size": result.batch_size,
        "runs": result.runs,
        "warmup": result.warmup,
        "text_chars": result.text_chars,
        "total_texts": result.total_texts,
        "total_chars": result.total_chars,
        "total_seconds": round(result.total_seconds, 6),
        "requests_per_second": round(requests_per_second, 3),
        "texts_per_second": round(texts_per_second, 3),
        "chars_per_second": round(chars_per_second, 3),
        "vector_dimensions": result.vector_dimensions,
        "response_bytes": result.response_bytes,
        "latency_seconds": {
            "min": round(min(request_seconds), 6),
            "p50": round(percentile(request_seconds, 50), 6),
            "p95": round(percentile(request_seconds, 95), 6),
            "max": round(max(request_seconds), 6),
        },
    }
    if result.power_measurements:
        payload["power"] = [
            {
                "label": measurement.label,
                "source": measurement.source,
                "source_type": measurement.source_type,
                "elapsed_seconds": round(measurement.elapsed_seconds, 6),
                "average_watts": round(measurement.average_watts, 6) if measurement.average_watts is not None else None,
                "energy_joules": round(measurement.energy_joules, 6) if measurement.energy_joules is not None else None,
                "joules_per_text": round(measurement.energy_joules / result.total_texts, 6)
                if measurement.energy_joules is not None
                else None,
                "joules_per_kchar": round(measurement.energy_joules / (result.total_chars / 1000.0), 6)
                if measurement.energy_joules is not None and result.total_chars > 0
                else None,
                "texts_per_joule": round(result.total_texts / measurement.energy_joules, 6)
                if measurement.energy_joules and measurement.energy_joules > 0
                else None,
                "kchars_per_joule": round((result.total_chars / 1000.0) / measurement.energy_joules, 6)
                if measurement.energy_joules and measurement.energy_joules > 0
                else None,
                "samples": measurement.samples,
                "note": measurement.note,
            }
            for measurement in result.power_measurements
        ]
    return payload


def print_human_result(result: BenchmarkResult) -> None:
    data = result_json(result)
    latency = data["latency_seconds"]
    if not isinstance(latency, dict):
        raise TypeError("latency summary must be an object")
    write_stdout("Embedding endpoint benchmark")
    write_stdout(f"endpoint: {result.endpoint}")
    write_stdout(f"model: {result.model}")
    if result.response_model:
        write_stdout(f"response model: {result.response_model}")
    write_stdout(f"vector dimensions: {result.vector_dimensions}")
    write_stdout(f"input source: {result.input_source}")
    write_stdout(
        "input chars: "
        f"{result.input_stats.count} texts, "
        f"p50 {result.input_stats.p50_chars}, "
        f"p95 {result.input_stats.p95_chars}, "
        f"max {result.input_stats.max_chars}"
    )
    write_stdout(f"batch size: {result.batch_size}")
    write_stdout(f"runs: {result.runs} warmup: {result.warmup} text chars: {result.text_chars}")
    write_stdout(f"total: {data['total_seconds']}s for {result.total_texts} texts / {result.total_chars} chars")
    write_stdout(
        "throughput: "
        f"{data['requests_per_second']} req/s, "
        f"{data['chars_per_second']} chars/s, "
        f"{data['texts_per_second']} texts/s"
    )
    write_stdout(f"latency: min {latency['min']}s, p50 {latency['p50']}s, p95 {latency['p95']}s, max {latency['max']}s")
    if result.power_measurements:
        write_stdout("power:")
        for measurement in result.power_measurements:
            if measurement.average_watts is None or measurement.energy_joules is None:
                write_stdout(f"  {measurement.label}: unavailable ({measurement.note or 'no reading'})")
                write_stdout(f"    source: {measurement.source}")
                continue
            joules_per_text = measurement.energy_joules / result.total_texts
            joules_per_kchar = measurement.energy_joules / (result.total_chars / 1000.0)
            texts_per_joule = result.total_texts / measurement.energy_joules if measurement.energy_joules > 0 else 0.0
            write_stdout(
                f"  {measurement.label}: "
                f"{measurement.average_watts:.3f} W avg, "
                f"{measurement.energy_joules:.3f} J total, "
                f"{joules_per_kchar:.4f} J/kchar, "
                f"{joules_per_text:.4f} J/text, "
                f"{texts_per_joule:.3f} texts/J"
            )
            write_stdout(f"    source: {measurement.source} ({measurement.source_type}, samples={measurement.samples})")
            if measurement.note:
                write_stdout(f"    note: {measurement.note}")


def configured_power_sensors(parsed: BenchNamespace) -> list[power.PowerSensorSource]:
    if parsed.power_sources:
        sensors: list[power.PowerSensorSource] = []
        for source in parsed.power_sources:
            sensors.extend(power.parse_power_source(source))
        return power.unique_sources(sensors)
    if parsed.power_enabled:
        return power.discover_power_sources()
    return []


def print_power_sources(sensors: list[power.PowerSensorSource]) -> None:
    if not sensors:
        write_stdout("No supported Linux power sources found.")
        return
    write_stdout("Supported Linux power sources:")
    for sensor in sensors:
        source_type = "energy_uj" if isinstance(sensor, power.EnergySensor) else "power_uw"
        write_stdout(f"  {sensor.label}: {sensor.path} ({source_type})")


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible embeddings endpoint.")
    _ = argument_parser.add_argument("--endpoint", default=default_endpoint(), help="Embeddings endpoint URL.")
    _ = argument_parser.add_argument("--model", default=default_model(), help="Embedding model name to send.")
    _ = argument_parser.add_argument("--batch-size", type=positive_int, default=16, help="Texts per request.")
    _ = argument_parser.add_argument("--runs", type=positive_int, default=5, help="Measured requests to send.")
    _ = argument_parser.add_argument("--warmup", type=non_negative_int, default=1, help="Unmeasured warmup requests.")
    _ = argument_parser.add_argument("--text-chars", type=positive_int, default=600, help="Characters per text.")
    _ = argument_parser.add_argument(
        "--min-duration",
        type=non_negative_float,
        default=0.0,
        dest="min_duration_seconds",
        help="Keep sending measured requests until at least this many seconds have elapsed.",
    )
    _ = argument_parser.add_argument(
        "--input-root",
        help="Use line-window chunks from repository files under this directory instead of synthetic snippets.",
    )
    _ = argument_parser.add_argument(
        "--input-max-texts",
        type=positive_int,
        default=4096,
        help="Maximum repository chunks to load when --input-root is set.",
    )
    _ = argument_parser.add_argument("--timeout", type=positive_float, default=300.0, help="Request timeout seconds.")
    _ = argument_parser.add_argument(
        "--power",
        action="store_true",
        dest="power_enabled",
        help="Measure available Linux power/energy sensors during measured runs.",
    )
    _ = argument_parser.add_argument(
        "--power-source",
        action="append",
        dest="power_sources",
        help=(
            "Explicit sysfs power source path. May be repeated; accepts energy_uj, power*_input, "
            "power*_average, or a sensor directory."
        ),
    )
    _ = argument_parser.add_argument(
        "--power-sample-interval",
        type=positive_float,
        default=0.2,
        help="Seconds between sampled power readings for power*_input sensors.",
    )
    _ = argument_parser.add_argument(
        "--list-power-sources",
        action="store_true",
        help="List auto-detected Linux power/energy sensors and exit.",
    )
    _ = argument_parser.add_argument("--json", action="store_true", dest="json_output", help="Print JSON only.")
    return argument_parser


def required_power_sensors(parsed: BenchNamespace) -> list[power.PowerSensorSource]:
    power_sensors = configured_power_sensors(parsed)
    if (parsed.power_enabled or parsed.power_sources) and not power_sensors:
        raise ValueError("no supported power sources found; use --list-power-sources or --power-source /sys/...")
    return power_sensors


def main(argv: list[str] | None = None) -> int:
    parsed = parser().parse_args(argv, namespace=BenchNamespace())
    try:
        if parsed.list_power_sources:
            print_power_sources(power.discover_power_sources())
            return 0
        power_sensors = required_power_sensors(parsed)
        result = run_benchmark(
            BenchmarkConfig(
                endpoint=parsed.endpoint,
                model=parsed.model,
                batch_size=parsed.batch_size,
                runs=parsed.runs,
                warmup=parsed.warmup,
                text_chars=parsed.text_chars,
                min_duration_seconds=parsed.min_duration_seconds,
                input_root=parsed.input_root,
                input_max_texts=parsed.input_max_texts,
                timeout=parsed.timeout,
                power_sensors=power_sensors,
                power_sample_interval=parsed.power_sample_interval,
            )
        )
    except (OSError, RuntimeError, ValueError, urllib.error.URLError) as exc:
        write_stderr(f"embedding benchmark failed: {exc}")
        return 1
    if parsed.json_output:
        write_stdout(json.dumps(result_json(result), indent=2, sort_keys=True))
    else:
        print_human_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
