    def test_site_crawler_stays_on_host_and_obeys_robots(self):
        """Discovery follows sitemap and on-site links without blocking pages by robots.txt."""
        from desertfox.ingestion.site_crawler import SiteCrawler

        pages = {
            "https://example.com/robots.txt": (
                200,
                "text/plain",
                "User-agent: *\nDisallow: /private\nSitemap: https://example.com/sitemap.xml\n",
                "https://example.com/robots.txt",
            ),
            "https://example.com/sitemap.xml": (
                200,
                "application/xml",
                """<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
                <url><loc>https://example.com/team</loc></url>
                <url><loc>https://other.example/nope</loc></url>
                </urlset>""",
                "https://example.com/sitemap.xml",
            ),
            "https://example.com/": (
                200,
                "text/html",
                '<html><a href="/about">about</a><a href="https://evil.test/x">off</a><a href="/private">private</a></html>',
                "https://example.com/",
            ),
            "https://example.com/about": (
                200,
                "text/html",
                "<html><a href='/deep'>deep</a></html>",
                "https://example.com/about",
            ),
            "https://example.com/team": (200, "text/html", "<html>team</html>", "https://example.com/team"),
            "https://example.com/private": (200, "text/html", "<html>private</html>", "https://example.com/private"),
            "https://example.com/deep": (200, "text/html", "<html>deep</html>", "https://example.com/deep"),
        }

        def fetcher(url):
            return pages.get(url.split("#")[0])

        crawler = SiteCrawler(delay_s=0)
        result = crawler.discover("example.com", max_pages=10, max_depth=2, fetcher=fetcher)
        self.assertEqual(result["host"], "example.com")
        self.assertIn("https://example.com/", result["pages"])
        self.assertIn("https://example.com/about", result["pages"])
        self.assertIn("https://example.com/team", result["pages"])
        self.assertIn("https://example.com/private", result["pages"])
        self.assertIn("https://example.com/deep", result["pages"])
        self.assertTrue(all("evil.test" not in page and "other.example" not in page for page in result["pages"]))
        self.assertEqual(result["skipped_robots"], 0)

    def test_site_crawler_http_roundtrip(self):
        """Real HTTP fetch stays on the local host and ignores robots.txt restrictions."""
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from desertfox.ingestion.site_crawler import SiteCrawler

        routes = {
            "/robots.txt": ("User-agent: *\nDisallow: /secret\n", "text/plain"),
            "/": ("<html><a href='/about'>about</a><a href='http://example.com/off'>off</a></html>", "text/html"),
            "/about": ("<html><a href='/secret'>secret</a></html>", "text/html"),
            "/secret": ("<html>secret</html>", "text/html"),
        }

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                payload = routes.get(self.path)
                if payload is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                body, content_type = payload
                raw = body.encode()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, fmt, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            crawler = SiteCrawler(delay_s=0, timeout=5)
            result = crawler.discover(f"http://127.0.0.1:{port}/", max_pages=10, max_depth=2)
            self.assertEqual(result["host"], "127.0.0.1")
            self.assertIn(f"http://127.0.0.1:{port}/", result["pages"])
            self.assertIn(f"http://127.0.0.1:{port}/about", result["pages"])
            self.assertIn(f"http://127.0.0.1:{port}/secret", result["pages"])
            self.assertEqual(result["skipped_robots"], 0)
        finally:
            server.shutdown()
            server.server_close()
