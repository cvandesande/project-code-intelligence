"""Shared HTTP helpers for OpenAI-compatible embedding servers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from http.server import BaseHTTPRequestHandler

    from project_code_intelligence.models import JsonObject


def normalize_input(value: object) -> list[str]:
    """Parse the ``input`` field of an embedding request into a list of strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        texts: list[str] = []
        for item in cast("list[object]", value):
            if not isinstance(item, str):
                raise TypeError("embedding input array items must be strings")
            texts.append(item)
        if not texts:
            raise ValueError("embedding input array must not be empty")
        return texts
    raise TypeError("embedding input must be a string or an array of strings")


def json_error(message: str) -> JsonObject:
    """Build a standard OpenAI-style error response body."""
    return {"error": {"message": message, "type": "invalid_request_error"}}


def write_json(handler: BaseHTTPRequestHandler, status: int, payload: JsonObject) -> None:
    """Write a JSON response with the given HTTP status code."""
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    _ = handler.wfile.write(body)


def parse_json_body(handler: BaseHTTPRequestHandler, *, max_bytes: int) -> JsonObject:
    """Read and parse a JSON request body, enforcing a size limit."""
    length_header = handler.headers.get("Content-Length")
    if not length_header:
        raise ValueError("Content-Length is required")
    try:
        length = int(length_header)
    except ValueError as exc:
        raise ValueError("Content-Length must be an integer") from exc
    if length <= 0:
        raise ValueError("request body is empty")
    if length > max_bytes:
        raise ValueError("request body exceeds size limit")
    raw = handler.rfile.read(length)
    value = cast("object", json.loads(raw.decode("utf-8")))
    if not isinstance(value, dict):
        raise TypeError("request body must be a JSON object")
    return cast("JsonObject", value)
