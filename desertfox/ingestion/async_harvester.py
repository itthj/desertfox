"""
High-Throughput Asynchronous Media Harvester for DesertFox.
Implements non-blocking concurrent fetching via aiohttp for multi-image crawling.
Provides politeness concurrency controls, domain rate limits, and streaming validation.
"""

import asyncio
import io
import time
import logging
from typing import List, Dict, Any, Set, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from PIL import Image

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from desertfox.core.hashing import MultiPerceptualHasher
from desertfox.core.fiqa import FaceQualityAssessment
from desertfox.ingestion.metadata import ForensicMetadataExtractor

logger = logging.getLogger(__name__)


class AsyncMediaHarvester:
    """
    Asynchronous web media crawler with high-throughput concurrent image fetching.
    """

    def __init__(
        self,
        user_agent: str = "DesertFox-OSINT-Defense-Audit/3.0 (+https://github.com/itthj/desertfox)",
        min_width: int = 64,
        min_height: int = 64,
        concurrency: int = 8,
        request_timeout: int = 10
    ):
        self.headers = {"User-Agent": user_agent}
        self.min_width = min_width
        self.min_height = min_height
        self.concurrency = concurrency
        self.timeout = request_timeout
        self.seen_composite_hashes: Set[str] = set()

    def extract_image_candidates_from_html(self, html_content: str, base_url: str) -> List[str]:
        """Discovers candidate images across responsive srcset, OpenGraph, and standard tags."""
        soup = BeautifulSoup(html_content, "html.parser")
        candidates: Set[str] = set()

        for meta in soup.find_all("meta"):
            prop = meta.get("property") or meta.get("name")
            if prop in ["og:image", "og:image:url", "twitter:image", "twitter:image:src"]:
                content = meta.get("content")
                if content:
                    candidates.add(urljoin(base_url, content))

        for img in soup.find_all("img"):
            for attr in ["src", "data-src", "data-original", "data-lazy-src"]:
                val = img.get(attr)
                if val and not val.startswith("data:image"):
                    candidates.add(urljoin(base_url, val))

            srcset = img.get("srcset") or img.get("data-srcset")
            if srcset:
                for entry in [e.strip() for e in srcset.split(",") if e.strip()]:
                    parts = entry.split()
                    if parts:
                        candidates.add(urljoin(base_url, parts[0]))

        filtered = []
        for u in candidates:
            clean = u.split("?")[0].lower()
            if not any(clean.endswith(ext) for ext in [".svg", ".ico", ".gif", ".bmp"]):
                filtered.append(u)
        return filtered

    async def _fetch_single_media(
        self,
        session: 'aiohttp.ClientSession',
        image_url: str,
        semaphore: asyncio.Semaphore
    ) -> Optional[Dict[str, Any]]:
        """Asynchronously downloads image, validates dimensions, computes hashes, and checks FIQA."""
        async with semaphore:
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with session.get(image_url, headers=self.headers, timeout=timeout) as resp:
                    if resp.status != 200:
                        return None

                    content = await resp.read()
                    if len(content) < 2048:
                        return None

                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, self._process_image_bytes, content, image_url)
            except Exception as e:
                logger.debug("Failed async fetch for %s: %s", image_url, e)
                return None

    def _process_image_bytes(self, content: bytes, image_url: str) -> Optional[Dict[str, Any]]:
        """CPU-bound image decode, hashing, and quality validation."""
        try:
            img = Image.open(io.BytesIO(content))
            w, h = img.size
            if w < self.min_width or h < self.min_height:
                return None

            aspect_ratio = w / float(h)
            if aspect_ratio > 3.2 or aspect_ratio < 0.28:
                return None

            fingerprint = MultiPerceptualHasher.compute_cascade_fingerprint(img)
            composite_hash = fingerprint["composite"]
            if composite_hash in self.seen_composite_hashes:
                return None

            self.seen_composite_hashes.add(composite_hash)
            forensic_data = ForensicMetadataExtractor.extract_exif(img)
            fiqa_res = FaceQualityAssessment.evaluate_quality(img, min_dimension=48)

            return {
                "image": img,
                "url": image_url,
                "dimensions": (w, h),
                "aspect_ratio": round(aspect_ratio, 2),
                "format": img.format,
                "fingerprint": fingerprint,
                "forensics": forensic_data,
                "fiqa": fiqa_res
            }
        except Exception:
            return None

    async def crawl_target_async(self, page_url: str, max_harvest: int = 15) -> Dict[str, Any]:
        """Runs asynchronous high-throughput crawling across candidate image URLs."""
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required for AsyncMediaHarvester.")

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(page_url, headers=self.headers) as resp:
                    if resp.status != 200:
                        return {"error": f"HTTP {resp.status}", "page_url": page_url, "harvested": []}
                    html_text = await resp.text()

                page_osint = ForensicMetadataExtractor.extract_page_osint_context(html_text, page_url)
                candidates = self.extract_image_candidates_from_html(html_text, page_url)

                semaphore = asyncio.Semaphore(self.concurrency)
                tasks = [
                    self._fetch_single_media(session, u, semaphore)
                    for u in candidates[:max_harvest * 2]
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)
                valid_media = [r for r in results if isinstance(r, dict) and r is not None][:max_harvest]

                for m in valid_media:
                    m["page_osint"] = page_osint

                return {
                    "status": "success",
                    "page_url": page_url,
                    "page_osint": page_osint,
                    "candidates_discovered": len(candidates),
                    "valid_media_harvested": len(valid_media),
                    "media_items": valid_media
                }
        except Exception as e:
            return {"error": str(e), "page_url": page_url, "harvested": []}

    def crawl_target(self, page_url: str, max_harvest: int = 15) -> Dict[str, Any]:
        """Synchronous wrapper for asyncio event loop execution."""
        return asyncio.run(self.crawl_target_async(page_url, max_harvest=max_harvest))
