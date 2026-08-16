"""Small HTTP helpers with scheme validation."""

from __future__ import annotations

import urllib.request
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Mapping
    from http.client import HTTPResponse


def validate_http_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        msg = "HTTP URL must use http or https"
        raise ValueError(msg)
    if not parsed.netloc or not parsed.hostname:
        msg = "HTTP URL must include a host"
        raise ValueError(msg)


def request(
    url: str,
    *,
    data: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    method: str | None = None,
) -> urllib.request.Request:
    validate_http_url(url)
    # URL scheme and host are validated above.
    return urllib.request.Request(url, data=data, headers=dict(headers or {}), method=method)


def read_text(request_or_url: urllib.request.Request | str, *, timeout: float) -> str:
    url = request_or_url.full_url if isinstance(request_or_url, urllib.request.Request) else request_or_url
    validate_http_url(url)
    # URL scheme and host are validated above.
    with cast("HTTPResponse", urllib.request.urlopen(request_or_url, timeout=timeout)) as response:
        return response.read().decode("utf-8")
