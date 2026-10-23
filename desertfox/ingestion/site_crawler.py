"""
Same-host site walker for DesertFox.

Given one seed URL, discovers pages on that host from its sitemap and from
links on the pages themselves. It does not query a search engine and it does
not follow links to other hosts. robots.txt is respected by default, but can
be explicitly disabled for a broad audit.
"""

import logging
import time
import xml.etree.ElementTree as ET
from collections import deque
from typing import Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from desertfox.security import UnsafeURLError, safe_get

logger = logging.getLogger(__name__)

FetchResult = Tuple[int, str, str, str]
Fetcher = Callable[[str], Optional[FetchResult]]

_SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".ico", ".bmp",
    ".css", ".js", ".mjs", ".pdf", ".zip", ".gz", ".mp4", ".mp3", ".webm",
    ".woff", ".woff2", ".ttf", ".eot", ".json", ".xml",
}


def host_key(url: str) -> str:
    """Hostname used to keep a crawl on one site. Treats www as the same host."""
    host = (urlparse(url).hostname or "").lower()
    if host.startswith("www."):
        return host[4:]
    return host


def normalize_url(url: str) -> str:
    """Strip fragments and default a bare host to https."""
    raw = (url or "").strip()
    if not raw or raw.startswith(("mailto:", "javascript:", "tel:", "data:")):
        return ""
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    parts = urlparse(raw)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return ""
    path = parts.path or "/"
    return urlunparse((parts.scheme, parts.netloc, path, "", parts.query, ""))


def _local_name(tag: str) -> str:
    return tag.split("}")[-1].lower()


def parse_sitemap(xml_text: str) -> Tuple[List[str], List[str]]:
    """Return (page URLs, child sitemap URLs) from a sitemap or sitemap index."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return [], []
    pages: List[str] = []
    children: List[str] = []
    for node in list(root):
        name = _local_name(node.tag)
        if name not in ("url", "sitemap"):
            continue
        loc = ""
        for child in list(node):
            if _local_name(child.tag) == "loc" and child.text:
                loc = child.text.strip()
                break
        if not loc:
            continue
        if name == "sitemap":
            children.append(loc)
        else:
            pages.append(loc)
    return pages, children


def extract_links(html: str, base_url: str, limit: int = 100) -> List[str]:
    """Absolute http(s) links from anchor tags, capped per page."""
    soup = BeautifulSoup(html, "html.parser")
    found: List[str] = []
    seen: Set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue
        absolute = normalize_url(urljoin(base_url, href))
        if not absolute or absolute in seen:
            continue
        seen.add(absolute)
        found.append(absolute)
        if len(found) >= limit:
            break
    return found


class SiteCrawler:
    """Polite same-host discovery. One seed in, a bounded page list out."""

    def __init__(
        self,
        user_agent: str = "DesertFox-OSINT-Defense-Audit/3.0 (+https://github.com/itthj/desertfox)",
        delay_s: float = 0.25,
        timeout: int = 10,
        max_body_bytes: int = 2_000_000,
    ):
        self.user_agent = user_agent
        self.delay_s = delay_s
        self.timeout = timeout
        self.max_body_bytes = max_body_bytes
        self.headers = {"User-Agent": user_agent}
        self._last_request = 0.0

    def discover(
        self,
        seed_url: str,
        max_pages: int = 30,
        max_depth: int = 2,
        fetcher: Optional[Fetcher] = None,
        enforce_robots: bool = True,
    ) -> Dict[str, object]:
        """
        Walk one host. max_pages and max_depth are hard bounds.
        fetcher is a test hook: (url) -> (status, content_type, text, final_url) or None.
        When enforce_robots is False, the crawl ignores robots.txt restrictions intentionally.
        """
        fetch = fetcher or self._http_fetch
        seed = normalize_url(seed_url)
        host = host_key(seed)
        if not seed or not host:
            return {"error": "Seed URL is not a valid http(s) address", "pages": [], "skipped_robots": 0, "host": ""}

        max_pages = max(1, min(int(max_pages), 100))
        max_depth = max(0, min(int(max_depth), 4))
        origin = f"{urlparse(seed).scheme}://{urlparse(seed).netloc}"

        parser = None
        sitemap_urls = []
        if enforce_robots:
            parser, sitemap_urls = self._load_robots(origin, fetch)
            if parser is not None:
                crawl_delay = None
                if hasattr(parser, "crawl_delay"):
                    crawl_delay = parser.crawl_delay(self.user_agent)
                if crawl_delay:
                    self.delay_s = max(self.delay_s, float(crawl_delay))

        def allowed(url: str) -> bool:
            if not enforce_robots or parser is None:
                return True
            return parser.can_fetch(self.user_agent, url)

        seen: Set[str] = set()
        queue: deque = deque()
        skipped = 0

        def consider(url: str, depth: int) -> None:
            nonlocal skipped
            normalized = normalize_url(url)
            if not normalized or host_key(normalized) != host:
                return
            path = urlparse(normalized).path.lower()
            if any(path.endswith(ext) for ext in _SKIP_EXTENSIONS):
                return
            if normalized in seen:
                return
            seen.add(normalized)
            if enforce_robots and not allowed(normalized):
                skipped += 1
                return
            queue.append((normalized, depth))

        consider(seed, 0)
        self._ingest_sitemaps(sitemap_urls or [origin + "/sitemap.xml"], fetch, consider, max_pages)

        pages: List[str] = []
        while queue and len(pages) < max_pages:
            url, depth = queue.popleft()
            fetched = fetch(url)
            if not fetched or fetched[0] != 200:
                continue
            _status, content_type, text, final_url = fetched
            if host_key(final_url) != host:
                continue
            if not self._looks_like_html(content_type, text):
                continue
            pages.append(url)
            if depth >= max_depth:
                continue
            for link in extract_links(text, final_url or url):
                consider(link, depth + 1)

        return {
            "host": host,
            "pages": pages,
            "skipped_robots": skipped,
            "seed_url": seed,
        }

    def _load_robots(self, origin: str, fetch: Fetcher):
        import urllib.robotparser as robotparser

        fetched = fetch(origin + "/robots.txt")
        if not fetched or fetched[0] != 200 or not fetched[2].strip():
            return None, []
        lines = fetched[2].splitlines()
        parser = robotparser.RobotFileParser()
        parser.parse(lines)
        sitemaps = []
        for line in lines:
            if line.lower().startswith("sitemap:"):
                location = line.split(":", 1)[1].strip()
                if location:
                    sitemaps.append(location)
        return parser, sitemaps

    def _ingest_sitemaps(self, seeds: List[str], fetch: Fetcher, consider, max_pages: int) -> None:
        pending = list(seeds[:8])
        seen_maps: Set[str] = set()
        fetches = 0
        accepted = 0
        cap = max_pages * 5
        while pending and fetches < 8 and accepted < cap:
            sitemap_url = normalize_url(pending.pop(0))
            if not sitemap_url or sitemap_url in seen_maps:
                continue
            seen_maps.add(sitemap_url)
            fetched = fetch(sitemap_url)
            fetches += 1
            if not fetched or fetched[0] != 200:
                continue
            pages, children = parse_sitemap(fetched[2])
            for page in pages:
                if accepted >= cap:
                    break
                consider(page, 0)
                accepted += 1
            for child in children:
                if len(pending) < 8:
                    pending.append(child)

    def _http_fetch(self, url: str) -> Optional[FetchResult]:
        self._polite_wait()
        try:
            response = safe_get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                max_bytes=self.max_body_bytes,
                truncate=True,
            )
            return response.status_code, response.content_type, response.text, response.url
        except (requests.RequestException, UnsafeURLError) as exc:
            logger.debug("Fetch failed for %s: %s", url, exc)
            return None

    def _polite_wait(self) -> None:
        if self.delay_s <= 0:
            self._last_request = time.monotonic()
            return
        elapsed = time.monotonic() - self._last_request
        remaining = self.delay_s - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request = time.monotonic()

    @staticmethod
    def _looks_like_html(content_type: str, text: str) -> bool:
        if "html" in (content_type or "").lower():
            return True
        sample = text.lstrip()[:300].lower()
        return sample.startswith("<!doctype html") or sample.startswith("<html") or "<html" in sample
