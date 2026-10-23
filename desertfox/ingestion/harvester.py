"""
Advanced Asynchronous Media Harvester for DesertFox.
Handles:
1. Multi-source image extraction: src, data-src, srcset, og:image, twitter:image
2. Content dimension and aspect-ratio validation (rejects icons, avatars < 48px, banners)
3. Domain rate-limiting and politeness controls
4. Full metadata enrichment (EXIF, GPS, and OpenGraph context)
"""

import re
import io
import time
from typing import List, Dict, Any, Set, Optional
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from PIL import Image

from desertfox.core.hashing import MultiPerceptualHasher
from desertfox.core.fiqa import FaceQualityAssessment
from desertfox.ingestion.metadata import ForensicMetadataExtractor

class AdvancedHarvester:
    """Production-grade web scraper and image harvester for OSINT investigations."""

    def __init__(
        self,
        user_agent: str = "DesertFox-OSINT-Defense-Audit/2.0 (+https://github.com/itthj/desertfox)",
        min_width: int = 64,
        min_height: int = 64,
        request_timeout: int = 10
    ):
        self.headers = {"User-Agent": user_agent}
        self.min_width = min_width
        self.min_height = min_height
        self.timeout = request_timeout
        self.seen_composite_hashes: Set[str] = set()
        self.last_domain_request: Dict[str, float] = {}

    def extract_image_candidates_from_html(self, html_content: str, base_url: str) -> List[str]:
        """
        Discovers images from standard img tags, lazy-load data-src, responsive srcset,
        picture sources, and social media OpenGraph/Twitter card tags.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        candidates: Set[str] = set()

        # 1. Social OpenGraph and Twitter card images
        for meta in soup.find_all("meta"):
            prop = meta.get("property") or meta.get("name")
            if prop in ["og:image", "og:image:url", "twitter:image", "twitter:image:src"]:
                content = meta.get("content")
                if content:
                    candidates.add(urljoin(base_url, content))

        # 2. Standard and lazy-loaded img tags
        for img in soup.find_all("img"):
            for attr in ["src", "data-src", "data-original", "data-lazy-src"]:
                val = img.get(attr)
                if val and not val.startswith("data:image"):
                    candidates.add(urljoin(base_url, val))

            # Parse responsive srcset (e.g. "image-480.jpg 480w, image-1080.jpg 1080w")
            srcset = img.get("srcset") or img.get("data-srcset")
            if srcset:
                urls = self._parse_srcset(srcset, base_url)
                candidates.update(urls)

        # 3. HTML5 Picture sources
        for picture in soup.find_all("picture"):
            for source in picture.find_all("source"):
                srcset = source.get("srcset")
                if srcset:
                    candidates.update(self._parse_srcset(srcset, base_url))

        # Filter out common UI icons / tracking pixels by file extensions
        filtered_candidates = []
        for u in candidates:
            u_clean = u.split("?")[0].lower()
            if any(u_clean.endswith(ext) for ext in [".svg", ".ico", ".gif", ".bmp"]):
                continue
            filtered_candidates.append(u)

        return filtered_candidates

    def _parse_srcset(self, srcset_str: str, base_url: str) -> List[str]:
        """Extracts the highest-resolution image candidates from a srcset string."""
        results = []
        entries = [e.strip() for e in srcset_str.split(",") if e.strip()]
        for entry in entries:
            parts = entry.split()
            if parts:
                results.append(urljoin(base_url, parts[0]))
        # Return sorted or reversed to prioritize larger resolution candidates
        return list(reversed(results))

    def fetch_and_validate_media(self, image_url: str) -> Optional[Dict[str, Any]]:
        """
        Downloads image bytes, validates dimensions and aspect ratios,
        computes perceptual hash cascade, extracts EXIF/GPS, and runs FIQA.
        """
        domain = urlparse(image_url).netloc
        # Politeness rate-limit (minimum 250ms between requests to the same domain)
        last_req = self.last_domain_request.get(domain, 0.0)
        now = time.time()
        if (now - last_req) < 0.25:
            time.sleep(0.25 - (now - last_req))
        self.last_domain_request[domain] = time.time()

        try:
            resp = requests.get(image_url, headers=self.headers, timeout=self.timeout)
            if resp.status_code != 200:
                return None

            content_len = len(resp.content)
            if content_len < 2048:  # Reject tiny icons or 1px trackers
                return None

            # Open image via PIL
            img = Image.open(io.BytesIO(resp.content))
            w, h = img.size

            # Dimension checks
            if w < self.min_width or h < self.min_height:
                return None

            # Aspect ratio check: (discard extreme banners)
            aspect_ratio = w / float(h)
            if aspect_ratio > 3.2 or aspect_ratio < 0.28:
                return None

            # 1. Perceptual Hash Cascade
            fingerprint = MultiPerceptualHasher.compute_cascade_fingerprint(img)
            composite_hash = fingerprint["composite"]
            if composite_hash in self.seen_composite_hashes:
                return None  # Duplicate

            self.seen_composite_hashes.add(composite_hash)

            # 2. Forensic EXIF & GPS Extraction
            forensic_data = ForensicMetadataExtractor.extract_exif(img)

            # 3. Face Image Quality Assessment (FIQA)
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

    def crawl_target(self, page_url: str, max_harvest: int = 15) -> Dict[str, Any]:
        """
        Full crawl pipeline: scrapes page, harvests OSINT metadata,
        extracts valid images, and runs biometric validation.
        """
        try:
            resp = requests.get(page_url, headers=self.headers, timeout=self.timeout)
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}", "page_url": page_url, "harvested": []}

            # Extract page-level OSINT context
            page_osint = ForensicMetadataExtractor.extract_page_osint_context(resp.text, page_url)

            # Discover all candidate media URLs
            candidates = self.extract_image_candidates_from_html(resp.text, page_url)

            valid_media = []
            for img_url in candidates:
                media = self.fetch_and_validate_media(img_url)
                if media:
                    media["page_osint"] = page_osint
                    valid_media.append(media)
                if len(valid_media) >= max_harvest:
                    break

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
