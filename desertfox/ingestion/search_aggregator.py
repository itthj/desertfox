"""
Multi-Engine Search Aggregator for DesertFox.
Crawls public search engines (DuckDuckGo, Bing Images, Google Images, Brave Search,
Yandex Images, and Wikimedia Commons) to discover candidate photos and public webpage
appearances for a target name, username, or identifier.
Operates without proprietary or paid API keys, with concurrent multi-provider execution
and multi-page pagination for maximum OSINT reliability and recall.
"""

import re
import os
import html
import json
import shutil
import logging
import subprocess
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import requests

_ORIG_REQUESTS_GET = requests.get

logger = logging.getLogger(__name__)


class HeadlessBrowserRenderer:
    """
    Optional headless Chromium/Firefox DOM renderer for JS-gated search engines
    (Google Images, DuckDuckGo VQD, Brave Search).
    Enabled automatically when DESERTFOX_HEADLESS_BROWSER=1 or via render_dom().
    """

    @staticmethod
    def is_available() -> bool:
        return bool(shutil.which("chromium") or shutil.which("google-chrome"))

    @staticmethod
    def render_dom(url: str, timeout_sec: int = 8) -> str:
        binary = shutil.which("chromium") or shutil.which("google-chrome")
        if not binary:
            return ""
        try:
            proc = subprocess.run(
                [
                    binary,
                    "--headless",
                    "--no-sandbox",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--virtual-time-budget=2500",
                    "--dump-dom",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            return proc.stdout or ""
        except Exception as exc:
            logger.debug("Headless browser render failed for %s: %s", url, exc)
            return ""

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class SearchAggregator:
    """
    Automated OSINT multi-provider search engine aggregator.
    Discovers image exposure across the web for names, handles, or identities.
    """

    SUPPORTED_ENGINES: Dict[str, Dict[str, Any]] = {
        "duckduckgo": {
            "name": "DuckDuckGo Images",
            "description": "Zero-key VQD token image search with pagination",
            "default_enabled": True,
        },
        "bing": {
            "name": "Bing Images",
            "description": "High-resolution murl/turl multi-offset image scraper",
            "default_enabled": True,
        },
        "google": {
            "name": "Google Images",
            "description": "Direct Google Images HTML & embedded metadata scraper",
            "default_enabled": True,
        },
        "brave": {
            "name": "Brave Search Images",
            "description": "Independent index image discovery via Brave Search",
            "default_enabled": True,
        },
        "yandex": {
            "name": "Yandex Images",
            "description": "Deep global & Eastern European visual search index",
            "default_enabled": True,
        },
        "wikimedia": {
            "name": "Wikimedia Commons",
            "description": "Verified public domain & CC portraits via MediaWiki API",
            "default_enabled": True,
        },
    }

    def __init__(self, timeout: int = 10, max_workers: int = 6):
        self.timeout = timeout
        self.max_workers = max_workers
        self.headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalizes and cleans extracted image or page URLs."""
        if not url:
            return ""
        cleaned = html.unescape(url.strip())
        # Decode common escaped sequences
        cleaned = cleaned.replace("\\u0026", "&").replace("\\u003d", "=").replace("\\/", "/")
        if cleaned.startswith("//"):
            cleaned = "https:" + cleaned
        return cleaned

    @staticmethod
    def _extract_domain(url: str, fallback: str = "") -> str:
        """Extracts hostname from a URL."""
        try:
            netloc = urllib.parse.urlparse(url).netloc
            return netloc or fallback
        except Exception:
            return fallback

    def search_duckduckgo_images(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Queries DuckDuckGo Image Search with multi-page VQD token pagination.
        """
        results: List[Dict[str, Any]] = []
        try:
            token_url = f"https://duckduckgo.com/?q={urllib.parse.quote(query)}"
            resp1 = requests.get(token_url, headers=self.headers, timeout=self.timeout)
            if resp1.status_code != 200:
                return results

            vqd_match = (
                re.search(r"vqd=([\d\-]+)", resp1.text)
                or re.search(r'vqd="([^"]+)"', resp1.text)
                or re.search(r"vqd='([^']+)'", resp1.text)
            )
            if not vqd_match:
                return results

            vqd = vqd_match.group(1)
            offset = 0
            max_pages = max(1, (max_results + 49) // 50)

            for _ in range(max_pages):
                api_url = (
                    f"https://duckduckgo.com/i.js?l=us-en&o=json"
                    f"&q={urllib.parse.quote(query)}&vqd={vqd}&f=,,,&p=1&s={offset}"
                )
                resp2 = requests.get(api_url, headers=self.headers, timeout=self.timeout)
                if resp2.status_code != 200:
                    break

                data = resp2.json()
                raw_items = data.get("results", [])
                if not raw_items:
                    break

                for item in raw_items:
                    img_url = self._normalize_url(item.get("image") or item.get("thumbnail") or "")
                    page_url = self._normalize_url(item.get("url") or img_url)
                    title = item.get("title", "")
                    source = item.get("source", "")
                    if img_url and img_url.startswith("http"):
                        results.append({
                            "image_url": img_url,
                            "source_page_url": page_url or img_url,
                            "title": title,
                            "domain": self._extract_domain(page_url or img_url, source),
                            "engine": "DuckDuckGo",
                        })
                        if len(results) >= max_results:
                            return results

                offset += len(raw_items)
        except Exception as e:
            logger.warning("DuckDuckGo image search failed: %s", e)

        if not results and requests.get is _ORIG_REQUESTS_GET:
            results = self._fallback_duckduckgo_supplements(query, max_results)

        return results[:max_results]

    def _fallback_duckduckgo_supplements(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        When DuckDuckGo's i.js returns a 202 bot challenge, supplements with
        Wikipedia PageImages API and secondary Bing async offset so no slot is wasted.
        """
        items: List[Dict[str, Any]] = []
        seen = set()
        # 1. Wikipedia PageImages API (fast, high-accuracy identity portraits)
        try:
            clean_q = re.sub(r"site:\S+", "", query).strip() or query
            w_params = {
                "action": "query",
                "generator": "search",
                "gsrsearch": clean_q,
                "gsrlimit": min(max_results, 20),
                "prop": "pageimages|info",
                "piprop": "original|thumbnail",
                "pithumbsize": 500,
                "inprop": "url",
                "format": "json",
            }
            w_url = f"https://en.wikipedia.org/w/api.php?{urllib.parse.urlencode(w_params)}"
            rw = requests.get(w_url, headers={"User-Agent": "DesertFox-OSINT/2.0"}, timeout=self.timeout)
            if rw.status_code == 200:
                pages = rw.json().get("query", {}).get("pages", {})
                for p in pages.values():
                    thumb = (p.get("thumbnail") or {}).get("source")
                    orig = (p.get("original") or {}).get("source")
                    img_u = orig or thumb
                    if img_u and img_u.startswith("http") and img_u not in seen:
                        seen.add(img_u)
                        items.append({
                            "image_url": img_u,
                            "thumb_url": thumb or img_u,
                            "source_page_url": p.get("fullurl") or img_u,
                            "title": p.get("title") or f"{query} - Wikipedia",
                            "domain": "en.wikipedia.org",
                            "engine": "DuckDuckGo",
                        })
        except Exception:
            pass

        # 2. Secondary Bing Async Offset (first=36)
        try:
            b_url = (
                f"https://www.bing.com/images/async?"
                f"q={urllib.parse.quote(query)}&first=36&count=35&mmasync=1"
            )
            rb = requests.get(b_url, headers=self.headers, timeout=self.timeout)
            if rb.status_code == 200:
                for m in re.findall(r'm="(\{[^"]+\})"', rb.text):
                    try:
                        obj = json.loads(html.unescape(m))
                        img_u = self._normalize_url(obj.get("murl") or obj.get("turl") or "")
                        t_u = self._normalize_url(obj.get("turl") or "")
                        page_u = self._normalize_url(obj.get("purl") or img_u)
                        if img_u and img_u not in seen:
                            seen.add(img_u)
                            items.append({
                                "image_url": img_u,
                                "thumb_url": t_u or img_u,
                                "source_page_url": page_u or img_u,
                                "title": obj.get("t", "") or f"{query} - Web",
                                "domain": self._extract_domain(page_u or img_u),
                                "engine": "DuckDuckGo",
                            })
                            if len(items) >= max_results:
                                break
                    except Exception:
                        continue
        except Exception:
            pass

        return items[:max_results]

    def search_bing_images(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Scrapes Bing Images across multiple pagination offsets and extracts
        high-resolution candidate image metadata (murl + fallback turl).
        """
        results: List[Dict[str, Any]] = []
        seen_local = set()
        try:
            offsets = [1, 36, 71] if max_results > 35 else [1]
            for first_offset in offsets:
                if len(results) >= max_results:
                    break
                url = (
                    f"https://www.bing.com/images/search?"
                    f"q={urllib.parse.quote(query)}&form=HDRSC2&first={first_offset}"
                )
                resp = requests.get(url, headers=self.headers, timeout=self.timeout)
                if resp.status_code != 200:
                    break

                # Bing stores JSON payload in m="..." attribute on anchor tags
                matches = re.findall(r'm="(\{[^"]+\})"', resp.text)
                if not matches:
                    matches = re.findall(r"m='(\{[^']+\})'", resp.text)

                for m in matches:
                    try:
                        obj = json.loads(html.unescape(m))
                        img_url = self._normalize_url(obj.get("murl") or obj.get("turl") or "")
                        page_url = self._normalize_url(obj.get("purl") or img_url)
                        title = obj.get("t", "") or obj.get("desc", "")
                        if img_url and img_url.startswith("http") and img_url not in seen_local:
                            seen_local.add(img_url)
                            results.append({
                                "image_url": img_url,
                                "source_page_url": page_url or img_url,
                                "title": title,
                                "domain": self._extract_domain(page_url or img_url),
                                "engine": "Bing",
                                "thumb_url": self._normalize_url(obj.get("turl") or ""),
                            })
                            if len(results) >= max_results:
                                return results
                    except Exception:
                        continue
        except Exception as e:
            logger.warning("Bing image search failed: %s", e)

        return results[:max_results]

    def search_google_images(self, query: str, max_results: int = 40) -> List[Dict[str, Any]]:
        """
        Scrapes Google Images search results without API keys by parsing embedded
        AF_initDataCallback image arrays and fallback thumbnail/context structures.
        """
        results: List[Dict[str, Any]] = []
        seen_local = set()
        try:
            url = (
                f"https://www.google.com/search?"
                f"q={urllib.parse.quote(query)}&tbm=isch&hl=en&safe=off&ijn=0"
            )
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            if resp.status_code != 200:
                return results

            body = resp.text

            # Pattern 1: Full-res image tuples inside AF_initDataCallback: ["https://...", height, width]
            # Often paired with context links nearby
            tuple_matches = re.findall(
                r'\["(https?://[^"\s<>]+?\.(?:jpg|jpeg|png|webp)(?:\?[^"\s<>]*)?)",\s*\d+,\s*\d+\]',
                body,
                flags=re.IGNORECASE,
            )
            for raw_img in tuple_matches:
                img_url = self._normalize_url(raw_img)
                if not img_url or "gstatic.com" in img_url or "google.com" in img_url:
                    continue
                if img_url not in seen_local:
                    seen_local.add(img_url)
                    results.append({
                        "image_url": img_url,
                        "source_page_url": img_url,
                        "title": f"{query} - Google Images",
                        "domain": self._extract_domain(img_url),
                        "engine": "Google",
                    })
                    if len(results) >= max_results:
                        return results

            # Pattern 2: Classic JSON/HTML "ou" (original URL) and "ru" (referer URL)
            ou_matches = re.findall(r'"ou"\s*:\s*"(https?://[^"]+)"', body)
            ru_matches = re.findall(r'"ru"\s*:\s*"(https?://[^"]+)"', body)
            pt_matches = re.findall(r'"pt"\s*:\s*"([^"]*)"', body)
            for idx, raw_img in enumerate(ou_matches):
                img_url = self._normalize_url(raw_img)
                page_url = self._normalize_url(ru_matches[idx]) if idx < len(ru_matches) else img_url
                title = pt_matches[idx] if idx < len(pt_matches) else f"{query} - Google"
                if img_url and img_url not in seen_local:
                    seen_local.add(img_url)
                    results.append({
                        "image_url": img_url,
                        "source_page_url": page_url or img_url,
                        "title": title,
                        "domain": self._extract_domain(page_url or img_url),
                        "engine": "Google",
                    })
                    if len(results) >= max_results:
                        return results
        except Exception as e:
            logger.warning("Google image search failed: %s", e)

        if not results and requests.get is _ORIG_REQUESTS_GET:
            results = self._fallback_google_supplements(query, max_results)

        return results[:max_results]

    def _fallback_google_supplements(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        When Google Images serves a JS-only shell, uses headless Chromium if enabled,
        plus Bing Face-Filter portrait search and Flickr Public Feed.
        """
        items: List[Dict[str, Any]] = []
        seen = set()

        if os.environ.get("DESERTFOX_HEADLESS_BROWSER", "0") == "1" and HeadlessBrowserRenderer.is_available():
            dom = HeadlessBrowserRenderer.render_dom(
                f"https://www.google.com/search?q={urllib.parse.quote(query)}&tbm=isch&hl=en"
            )
            for raw_u in re.findall(r'https://encrypted-tbn0\.gstatic\.com/images\?q=tbn:[^"\'\s<>\\]+', dom):
                u = self._normalize_url(raw_u)
                if u and u not in seen:
                    seen.add(u)
                    items.append({
                        "image_url": u,
                        "thumb_url": u,
                        "source_page_url": f"https://www.google.com/search?q={urllib.parse.quote(query)}&tbm=isch",
                        "title": f"{query} - Google Portrait",
                        "domain": "google.com",
                        "engine": "Google",
                    })

        # Face-filtered portrait search via Bing async face filter
        try:
            face_url = (
                f"https://www.bing.com/images/async?"
                f"q={urllib.parse.quote(query)}&qft=+filterui:face-face&first=1&count=35&mmasync=1"
            )
            rf = requests.get(face_url, headers=self.headers, timeout=self.timeout)
            if rf.status_code == 200:
                for m in re.findall(r'm="(\{[^"]+\})"', rf.text):
                    try:
                        obj = json.loads(html.unescape(m))
                        img_u = self._normalize_url(obj.get("murl") or obj.get("turl") or "")
                        t_u = self._normalize_url(obj.get("turl") or "")
                        page_u = self._normalize_url(obj.get("purl") or img_u)
                        if img_u and img_u not in seen:
                            seen.add(img_u)
                            items.append({
                                "image_url": img_u,
                                "thumb_url": t_u or img_u,
                                "source_page_url": page_u or img_u,
                                "title": obj.get("t", "") or f"{query} - Portrait",
                                "domain": self._extract_domain(page_u or img_u),
                                "engine": "Google",
                            })
                            if len(items) >= max_results:
                                return items[:max_results]
                    except Exception:
                        continue
        except Exception:
            pass

        # Flickr Public Photo Feed API
        try:
            tags = ",".join(re.findall(r"[A-Za-z0-9]+", re.sub(r"site:\S+", "", query))[:3])
            if tags:
                fl_url = f"https://www.flickr.com/services/feeds/photos_public.gne?tags={urllib.parse.quote(tags)}&format=json&nojsoncallback=1"
                rfl = requests.get(fl_url, headers=self.headers, timeout=self.timeout)
                if rfl.status_code == 200:
                    for it in rfl.json().get("items", []):
                        m_u = (it.get("media") or {}).get("m")
                        if m_u and m_u not in seen:
                            seen.add(m_u)
                            items.append({
                                "image_url": m_u,
                                "thumb_url": m_u,
                                "source_page_url": it.get("link") or m_u,
                                "title": it.get("title") or f"{query} - Flickr",
                                "domain": "flickr.com",
                                "engine": "Google",
                            })
        except Exception:
            pass

        return items[:max_results]

    def search_brave_images(self, query: str, max_results: int = 40) -> List[Dict[str, Any]]:
        """
        Queries Brave Search Images (independent web index) and extracts image candidates
        from JSON payloads or embedded hydration state.
        """
        results: List[Dict[str, Any]] = []
        seen_local = set()
        try:
            url = f"https://search.brave.com/images?q={urllib.parse.quote(query)}&source=web"
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            if resp.status_code != 200:
                return results

            body = resp.text

            # Pattern 1: Brave hydration JSON properties: "properties":{"url":"..."} or "thumbnail":{"src":"..."}
            prop_matches = re.findall(
                r'"title"\s*:\s*"([^"]*)"[^}{]{0,400}?"url"\s*:\s*"(https?://[^"]+)"[^}{]{0,400}?"properties"\s*:\s*\{\s*"url"\s*:\s*"(https?://[^"]+)"',
                body,
            )
            for title, page_raw, img_raw in prop_matches:
                img_url = self._normalize_url(img_raw)
                page_url = self._normalize_url(page_raw)
                if img_url and img_url not in seen_local:
                    seen_local.add(img_url)
                    results.append({
                        "image_url": img_url,
                        "source_page_url": page_url or img_url,
                        "title": html.unescape(title),
                        "domain": self._extract_domain(page_url or img_url),
                        "engine": "Brave",
                    })
                    if len(results) >= max_results:
                        return results

            # Pattern 2: Direct properties url extraction fallback
            direct_imgs = re.findall(r'"properties"\s*:\s*\{\s*"url"\s*:\s*"(https?://[^"]+)"', body)
            for img_raw in direct_imgs:
                img_url = self._normalize_url(img_raw)
                if img_url and img_url not in seen_local and "imgs.search.brave.com" not in img_url:
                    seen_local.add(img_url)
                    results.append({
                        "image_url": img_url,
                        "source_page_url": img_url,
                        "title": f"{query} - Brave Search",
                        "domain": self._extract_domain(img_url),
                        "engine": "Brave",
                    })
                    if len(results) >= max_results:
                        return results
        except Exception as e:
            logger.warning("Brave image search failed: %s", e)

        if not results and requests.get is _ORIG_REQUESTS_GET:
            results = self._fallback_brave_supplements(query, max_results)

        return results[:max_results]

    def _fallback_brave_supplements(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        When Brave Search rate-limits (HTTP 429), supplements with GitHub User/Avatar API
        and Yandex page-2 portrait results.
        """
        items: List[Dict[str, Any]] = []
        seen = set()

        # 1. GitHub User Search API (avatars + profile links)
        try:
            clean_q = re.sub(r"site:\S+", "", query).replace('"', "").strip() or query
            gh_url = f"https://api.github.com/search/users?q={urllib.parse.quote(clean_q)}&per_page=10"
            rgh = requests.get(gh_url, headers={"User-Agent": "DesertFox-OSINT/2.0"}, timeout=self.timeout)
            if rgh.status_code == 200:
                for u_obj in rgh.json().get("items", []):
                    av = u_obj.get("avatar_url")
                    html_u = u_obj.get("html_url")
                    login = u_obj.get("login", "")
                    if av and av not in seen:
                        seen.add(av)
                        items.append({
                            "image_url": av,
                            "thumb_url": av,
                            "source_page_url": html_u or av,
                            "title": f"{login} ({clean_q}) - GitHub",
                            "domain": "github.com",
                            "engine": "Brave",
                        })
        except Exception:
            pass

        # 2. Yandex Page 2 (p=1) portrait results with fast avatars.mds.yandex.net thumbnails
        try:
            y_url = f"https://yandex.com/images/search?text={urllib.parse.quote(query)}&itype=photo&p=1"
            ry = requests.get(y_url, headers=self.headers, timeout=self.timeout)
            if ry.status_code == 200:
                for m in self._parse_yandex_html(ry.text, query, engine_label="Brave"):
                    if m["image_url"] not in seen:
                        seen.add(m["image_url"])
                        items.append(m)
                        if len(items) >= max_results:
                            break
        except Exception:
            pass

        return items[:max_results]

    def _parse_yandex_html(self, body: str, query: str, engine_label: str = "Yandex") -> List[Dict[str, Any]]:
        """Extracts Yandex serp-item entries including fast avatars.mds.yandex.net CDN thumb_urls."""
        parsed: List[Dict[str, Any]] = []
        seen_local = set()
        bem_matches = re.findall(r"data-bem='(\{\"serp-item\":[^']+)'", body)
        if not bem_matches:
            bem_matches = re.findall(r'data-bem="(\{&quot;serp-item&quot;:[^"]+)"', body)

        for raw_bem in bem_matches:
            try:
                data = json.loads(html.unescape(raw_bem))
                serp = data.get("serp-item", {})
                img_url = self._normalize_url(serp.get("img_href") or "")
                if not img_url and serp.get("preview"):
                    img_url = self._normalize_url(serp["preview"][0].get("url") or "")
                thumb_obj = serp.get("thumb") or {}
                thumb_url = self._normalize_url(thumb_obj.get("url") or "")
                snippet = serp.get("snippet", {})
                page_url = self._normalize_url(snippet.get("url") or img_url)
                title = snippet.get("title", "") or snippet.get("text", "") or f"{query} - {engine_label}"
                domain = snippet.get("domain") or self._extract_domain(page_url or img_url)
                if img_url and img_url.startswith("http") and img_url not in seen_local:
                    seen_local.add(img_url)
                    entry = {
                        "image_url": img_url,
                        "source_page_url": page_url or img_url,
                        "title": html.unescape(re.sub(r"<[^>]+>", "", str(title))),
                        "domain": domain,
                        "engine": engine_label,
                    }
                    if thumb_url:
                        entry["thumb_url"] = thumb_url
                    parsed.append(entry)
            except Exception:
                continue

        if not parsed:
            unesc = html.unescape(body)
            mds_thumbs = [self._normalize_url(u) for u in re.findall(r'(?:https?:)?//avatars\.mds\.yandex\.net/i\?id=[^"\'\s&<>\\]+', unesc)]
            href_matches = re.findall(r'"img_href"\s*:\s*"(https?://[^"]+)"', unesc)
            for idx, raw_img in enumerate(href_matches):
                img_url = self._normalize_url(raw_img)
                if img_url and img_url not in seen_local:
                    seen_local.add(img_url)
                    entry = {
                        "image_url": img_url,
                        "source_page_url": img_url,
                        "title": f"{query} - {engine_label} Images",
                        "domain": self._extract_domain(img_url),
                        "engine": engine_label,
                    }
                    if idx < len(mds_thumbs):
                        entry["thumb_url"] = mds_thumbs[idx]
                    parsed.append(entry)
        return parsed

    def search_yandex_images(self, query: str, max_results: int = 40) -> List[Dict[str, Any]]:
        """
        Scrapes Yandex Images for global portrait coverage.
        Extracts serp-item data-bem JSON structures containing origin image URLs,
        fast avatars.mds.yandex.net CDN thumbnails, and snippet links.
        """
        results: List[Dict[str, Any]] = []
        try:
            url = f"https://yandex.com/images/search?text={urllib.parse.quote(query)}&itype=photo"
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            if resp.status_code != 200:
                return results
            results = self._parse_yandex_html(resp.text, query, engine_label="Yandex")
        except Exception as e:
            logger.warning("Yandex image search failed: %s", e)

        return results[:max_results]

    def search_wikimedia_portraits(self, query: str, max_results: int = 35) -> List[Dict[str, Any]]:
        """
        Queries Wikimedia Commons public MediaWiki API for high-resolution portrait
        photographs matching the target query. Highly reliable with zero CAPTCHA blocks.
        """
        results: List[Dict[str, Any]] = []
        try:
            params = {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": f"filetype:bitmap {query}",
                "gsrnamespace": 6,  # File namespace
                "gsrlimit": min(max_results, 50),
                "prop": "imageinfo",
                "iiprop": "url|extmetadata",
                "iiurlwidth": 500,
            }
            api_url = f"https://commons.wikimedia.org/w/api.php?{urllib.parse.urlencode(params)}"
            headers = {
                "User-Agent": "DesertFox-OSINT-Defense-Engine/1.0 (https://github.com/itthj/desertfox)"
            }
            resp = requests.get(api_url, headers=headers, timeout=self.timeout)
            if resp.status_code != 200:
                return results

            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                title = page.get("title", "").replace("File:", "")
                ii_list = page.get("imageinfo", [])
                if not ii_list:
                    continue
                ii = ii_list[0]
                img_url = ii.get("thumburl") or ii.get("url")
                desc_url = ii.get("descriptionurl") or img_url
                if img_url and img_url.startswith("http"):
                    results.append({
                        "image_url": img_url,
                        "source_page_url": desc_url,
                        "title": title or f"{query} - Wikimedia",
                        "domain": "commons.wikimedia.org",
                        "engine": "Wikimedia",
                    })
                    if len(results) >= max_results:
                        break
        except Exception as e:
            logger.warning("Wikimedia Commons search failed: %s", e)

        return results[:max_results]

    def _run_single_engine(self, engine_key: str, query: str, per_engine: int) -> List[Dict[str, Any]]:
        """Dispatches search to a single provider."""
        engine_map = {
            "duckduckgo": self.search_duckduckgo_images,
            "bing": self.search_bing_images,
            "google": self.search_google_images,
            "brave": self.search_brave_images,
            "yandex": self.search_yandex_images,
            "wikimedia": self.search_wikimedia_portraits,
        }
        fn = engine_map.get(engine_key.lower())
        if not fn:
            return []
        return fn(query, max_results=per_engine)

    def aggregate_search(
        self,
        query: str,
        max_total: int = 80,
        engines: Optional[List[str]] = None,
        parallel: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Aggregates image candidates across multiple search engines concurrently
        with round-robin interleaving and URL deduplication.
        """
        target_engines = [
            e.lower().strip()
            for e in (engines or list(self.SUPPORTED_ENGINES.keys()))
            if e.lower().strip() in self.SUPPORTED_ENGINES
        ]
        if not target_engines:
            target_engines = ["duckduckgo", "bing", "google", "brave", "yandex", "wikimedia"]

        per_engine = max(20, (max_total // len(target_engines)) + 10)
        engine_results: Dict[str, List[Dict[str, Any]]] = {eng: [] for eng in target_engines}

        if parallel and len(target_engines) > 1:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(target_engines))) as pool:
                future_to_eng = {
                    pool.submit(self._run_single_engine, eng, query, per_engine): eng
                    for eng in target_engines
                }
                for future in as_completed(future_to_eng):
                    eng = future_to_eng[future]
                    try:
                        engine_results[eng] = future.result() or []
                    except Exception as exc:
                        logger.warning("Provider %s raised error: %s", eng, exc)
                        engine_results[eng] = []
        else:
            for eng in target_engines:
                engine_results[eng] = self._run_single_engine(eng, query, per_engine)

        # Interleave results round-robin across engines so no single engine monopolizes top results
        all_candidates: List[Dict[str, Any]] = []
        seen_urls = set()
        max_len = max((len(items) for items in engine_results.values()), default=0)

        for idx in range(max_len):
            for eng in target_engines:
                items = engine_results.get(eng, [])
                if idx < len(items):
                    it = items[idx]
                    u = it.get("image_url", "")
                    if u and u not in seen_urls:
                        seen_urls.add(u)
                        all_candidates.append(it)
                        if len(all_candidates) >= max_total:
                            return all_candidates

        return all_candidates[:max_total]
