"""
Network and input hardening helpers for DesertFox.

The crawl endpoints fetch URLs supplied by API clients, which makes the server a
Server-Side Request Forgery (SSRF) vector: without checks, a client can make the
engine read localhost services or cloud metadata endpoints (e.g. 169.254.169.254).
Uploads are also decoded from base64 with no size bound.

This module provides:
  * validate_public_url()   - rejects non-http(s), credentialed, and non-public targets
  * safe_get()              - requests-based fetch that re-validates every redirect hop
                              and enforces a hard cap on downloaded bytes
  * decode_base64_image()   - bounded base64 -> PIL decoder with pixel-count limit

Set DESERTFOX_ALLOW_PRIVATE_URLS=1 to disable the address check (local development
against a test server only).

Known limitation: the hostname is resolved here for validation and resolved again by
the HTTP client when connecting, so a hostile DNS server could in theory answer
differently the second time (DNS rebinding). Deploy the crawler workers on a network
segment with egress filtering for defense in depth.
"""

import base64
import binascii
import io
import ipaddress
import os
import socket
from typing import Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from PIL import Image

MAX_REDIRECTS = 5
MAX_PAGE_BYTES = 5 * 1024 * 1024        # 5 MiB for HTML pages
MAX_IMAGE_BYTES = 15 * 1024 * 1024      # 15 MiB for downloaded images
MAX_UPLOAD_BYTES = 10 * 1024 * 1024     # 10 MiB decoded size for API uploads
MAX_IMAGE_PIXELS = 50_000_000           # reject decompression-bomb images


class UnsafeURLError(ValueError):
    """Raised when a URL targets a disallowed scheme, credential form, or non-public address."""


class PayloadTooLargeError(ValueError):
    """Raised when a download or upload exceeds the configured byte or pixel limit."""


def _private_urls_allowed() -> bool:
    return os.environ.get("DESERTFOX_ALLOW_PRIVATE_URLS", "").strip().lower() in {"1", "true", "yes"}


def _is_public_ip(addr: str) -> bool:
    ip = ipaddress.ip_address(addr.split("%", 1)[0])  # strip IPv6 zone id
    # Unwrap IPv4-mapped IPv6 (::ffff:127.0.0.1) so it is judged as the IPv4 address.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def validate_public_url(url: str) -> str:
    """
    Validates that `url` is an http(s) URL pointing at publicly routable addresses.
    Returns the URL unchanged on success; raises UnsafeURLError otherwise.
    """
    if not isinstance(url, str) or not url.strip():
        raise UnsafeURLError("URL must be a non-empty string")
    parsed = urlparse(url.strip())

    if parsed.scheme not in ("http", "https"):
        raise UnsafeURLError(f"Unsupported URL scheme: {parsed.scheme or '(none)'}")
    if parsed.username or parsed.password:
        raise UnsafeURLError("URLs with embedded credentials are not allowed")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError("URL has no hostname")

    if _private_urls_allowed():
        return url

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, ValueError) as exc:
        raise UnsafeURLError(f"Could not resolve host '{host}': {exc}") from exc

    if not infos:
        raise UnsafeURLError(f"Host '{host}' resolved to no addresses")
    for info in infos:
        addr = info[4][0]
        if not _is_public_ip(addr):
            raise UnsafeURLError(f"Host '{host}' resolves to a non-public address and is blocked")
    return url


class SafeResponse:
    """Minimal response object exposing the attributes the harvesters rely on."""

    def __init__(self, status_code: int, content: bytes, url: str, encoding: Optional[str], content_type: str = ""):
        self.status_code = status_code
        self.content_type = content_type
        self.content = content
        self.url = url
        self._encoding = encoding

    @property
    def text(self) -> str:
        return self.content.decode(self._encoding or "utf-8", errors="replace")


def safe_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10,
    max_bytes: int = MAX_PAGE_BYTES,
    max_redirects: int = MAX_REDIRECTS,
    truncate: bool = False,
) -> SafeResponse:
    """
    GET `url` with SSRF protection: every hop (including redirects) is validated,
    and the body is streamed and aborted once it exceeds `max_bytes`
    (or, with truncate=True, silently cut off at `max_bytes`).
    Raises UnsafeURLError / PayloadTooLargeError / requests.RequestException.
    """
    current = url
    for _ in range(max_redirects + 1):
        validate_public_url(current)
        resp = requests.get(current, headers=headers, timeout=timeout, stream=True, allow_redirects=False)
        try:
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                if not location:
                    return SafeResponse(resp.status_code, b"", current, resp.encoding)
                current = urljoin(current, location)
                continue

            declared = resp.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes and not truncate:
                raise PayloadTooLargeError(f"Response exceeds {max_bytes} bytes")

            chunks, total = [], 0
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    if truncate:
                        break
                    raise PayloadTooLargeError(f"Response exceeds {max_bytes} bytes")
            body = b"".join(chunks)
            if truncate:
                body = body[:max_bytes]
            return SafeResponse(resp.status_code, body, current, resp.encoding, resp.headers.get("Content-Type", ""))
        finally:
            resp.close()
    raise UnsafeURLError(f"Too many redirects (limit {max_redirects})")


def decode_base64_image(b64_str: str, max_bytes: int = MAX_UPLOAD_BYTES) -> Image.Image:
    """Decodes a base64 image payload; see decode_base64_payload for the limits enforced."""
    return decode_base64_payload(b64_str, max_bytes)[0]


def decode_base64_payload(b64_str: str, max_bytes: int = MAX_UPLOAD_BYTES) -> Tuple[Image.Image, bytes]:
    """
    Decodes a base64 string (optionally with a data-URI header) into (PIL image, raw bytes).
    Enforces a decoded-size cap and a pixel-count cap before any pixel data is decoded.
    """
    if not isinstance(b64_str, str) or not b64_str:
        raise ValueError("Image payload is empty")
    if "," in b64_str[:100]:  # data URI header, e.g. "data:image/png;base64,...."
        b64_str = b64_str.split(",", 1)[1]

    # Reject oversized payloads before allocating the decoded buffer (4 chars -> 3 bytes).
    if (len(b64_str) * 3) // 4 > max_bytes + 3:
        raise PayloadTooLargeError(f"Image exceeds {max_bytes} bytes")
    try:
        raw = base64.b64decode(b64_str, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Image payload is not valid base64") from exc
    if len(raw) > max_bytes:
        raise PayloadTooLargeError(f"Image exceeds {max_bytes} bytes")

    try:
        img = Image.open(io.BytesIO(raw))  # lazy: reads header only
    except Exception as exc:
        raise ValueError("Payload is not a decodable image") from exc
    if img.size[0] * img.size[1] > MAX_IMAGE_PIXELS:
        raise PayloadTooLargeError(f"Image exceeds {MAX_IMAGE_PIXELS} pixels")
    return img, raw
