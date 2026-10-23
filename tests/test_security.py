"""Tests for desertfox.security: SSRF guard, bounded fetch, and bounded image decoding."""

import base64
import io
import os
import unittest
from unittest import mock

from PIL import Image

from desertfox import security
from desertfox.security import (
    PayloadTooLargeError,
    UnsafeURLError,
    decode_base64_image,
    safe_get,
    validate_public_url,
)


class FakeResponse:
    def __init__(self, status=200, headers=None, body=b"", encoding="utf-8"):
        self.status_code = status
        self.headers = headers or {}
        self._body = body
        self.encoding = encoding
        self.closed = False

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def close(self):
        self.closed = True


def _png_b64(size=(8, 8)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class TestValidatePublicUrl(unittest.TestCase):
    def setUp(self):
        os.environ.pop("DESERTFOX_ALLOW_PRIVATE_URLS", None)

    def test_blocks_non_public_targets(self):
        for url in [
            "http://127.0.0.1/",
            "http://localhost:8000/admin",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.5/",
            "http://192.168.1.1/",
            "http://[::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://0.0.0.0/",
        ]:
            with self.subTest(url=url):
                with self.assertRaises(UnsafeURLError):
                    validate_public_url(url)

    def test_blocks_bad_schemes_and_credentials(self):
        for url in ["file:///etc/passwd", "ftp://example.com/x", "gopher://x/", "",
                    "http://user:pw@8.8.8.8/", "http:///nohost"]:
            with self.subTest(url=url):
                with self.assertRaises(UnsafeURLError):
                    validate_public_url(url)

    def test_allows_public_ip_literal(self):
        self.assertEqual(validate_public_url("https://8.8.8.8/x.jpg"), "https://8.8.8.8/x.jpg")

    def test_hostname_resolving_to_private_is_blocked(self):
        fake = [(2, 1, 6, "", ("10.1.2.3", 80))]
        with mock.patch.object(security.socket, "getaddrinfo", return_value=fake):
            with self.assertRaises(UnsafeURLError):
                validate_public_url("http://innocent.example/")

    def test_any_private_answer_blocks_mixed_results(self):
        fake = [(2, 1, 6, "", ("8.8.8.8", 80)), (2, 1, 6, "", ("127.0.0.1", 80))]
        with mock.patch.object(security.socket, "getaddrinfo", return_value=fake):
            with self.assertRaises(UnsafeURLError):
                validate_public_url("http://mixed.example/")

    def test_env_override_allows_private(self):
        with mock.patch.dict(os.environ, {"DESERTFOX_ALLOW_PRIVATE_URLS": "1"}):
            self.assertEqual(validate_public_url("http://127.0.0.1:9/"), "http://127.0.0.1:9/")
            with self.assertRaises(UnsafeURLError):  # scheme check still applies
                validate_public_url("file:///etc/passwd")


class TestSafeGet(unittest.TestCase):
    def setUp(self):
        os.environ.pop("DESERTFOX_ALLOW_PRIVATE_URLS", None)

    def test_returns_body(self):
        resp = FakeResponse(body=b"hello")
        with mock.patch.object(security.requests, "get", return_value=resp) as g:
            out = safe_get("http://8.8.8.8/")
        self.assertEqual((out.status_code, out.content, out.text), (200, b"hello", "hello"))
        self.assertFalse(g.call_args.kwargs["allow_redirects"])
        self.assertTrue(resp.closed)

    def test_redirect_to_private_address_is_blocked(self):
        redirect = FakeResponse(status=302, headers={"Location": "http://169.254.169.254/latest/"})
        with mock.patch.object(security.requests, "get", return_value=redirect) as g:
            with self.assertRaises(UnsafeURLError):
                safe_get("http://8.8.8.8/")
        self.assertEqual(g.call_count, 1)  # the private hop was never requested

    def test_follows_safe_relative_redirect(self):
        responses = [
            FakeResponse(status=301, headers={"Location": "/final"}),
            FakeResponse(body=b"ok"),
        ]
        with mock.patch.object(security.requests, "get", side_effect=responses) as g:
            out = safe_get("http://8.8.8.8/start")
        self.assertEqual(out.content, b"ok")
        self.assertEqual(out.url, "http://8.8.8.8/final")
        self.assertEqual(g.call_count, 2)

    def test_redirect_loop_is_capped(self):
        loop = lambda *a, **k: FakeResponse(status=302, headers={"Location": "/again"})
        with mock.patch.object(security.requests, "get", side_effect=loop):
            with self.assertRaises(UnsafeURLError):
                safe_get("http://8.8.8.8/", max_redirects=3)

    def test_declared_content_length_over_cap(self):
        resp = FakeResponse(headers={"Content-Length": "999999"}, body=b"x")
        with mock.patch.object(security.requests, "get", return_value=resp):
            with self.assertRaises(PayloadTooLargeError):
                safe_get("http://8.8.8.8/", max_bytes=1000)

    def test_streamed_body_over_cap_without_header(self):
        resp = FakeResponse(body=b"x" * 5000)
        with mock.patch.object(security.requests, "get", return_value=resp):
            with self.assertRaises(PayloadTooLargeError):
                safe_get("http://8.8.8.8/", max_bytes=1000)
        self.assertTrue(resp.closed)


class TestSiteCrawlerGuard(unittest.TestCase):
    def test_site_crawler_refuses_loopback_by_default(self):
        from desertfox.ingestion.site_crawler import SiteCrawler
        with mock.patch.dict(os.environ, {"DESERTFOX_ALLOW_PRIVATE_URLS": ""}):
            self.assertIsNone(SiteCrawler(delay_s=0)._http_fetch("http://127.0.0.1:9/"))

    def test_truncate_mode_cuts_instead_of_raising(self):
        resp = FakeResponse(headers={"Content-Length": "5000"}, body=b"x" * 5000)
        with mock.patch.object(security.requests, "get", return_value=resp):
            out = safe_get("http://8.8.8.8/", max_bytes=1000, truncate=True)
        self.assertEqual(len(out.content), 1000)


class TestDecodeBase64Image(unittest.TestCase):
    def test_plain_and_data_uri(self):
        b64 = _png_b64()
        self.assertEqual(decode_base64_image(b64).size, (8, 8))
        self.assertEqual(decode_base64_image("data:image/png;base64," + b64).size, (8, 8))

    def test_rejects_invalid_base64_and_non_images(self):
        with self.assertRaises(ValueError):
            decode_base64_image("!!!not base64!!!")
        with self.assertRaises(ValueError):
            decode_base64_image(base64.b64encode(b"plain text, not an image").decode())
        with self.assertRaises(ValueError):
            decode_base64_image("")

    def test_rejects_oversized_payload(self):
        with self.assertRaises(PayloadTooLargeError):
            decode_base64_image(_png_b64(), max_bytes=10)

    def test_rejects_pixel_bomb(self):
        with mock.patch.object(security, "MAX_IMAGE_PIXELS", 10):
            with self.assertRaises(PayloadTooLargeError):
                decode_base64_image(_png_b64((8, 8)))


if __name__ == "__main__":
    unittest.main()
