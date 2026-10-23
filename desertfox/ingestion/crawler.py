"""
Web crawler and media ingestion pipeline for DesertFox.
Extracts public image links from target URLs, computes perceptual hashes to eliminate
duplicates, and prepares images for biometric embedding.
"""

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Any, Set, Optional
import io
from PIL import Image
from desertfox.core.embedder import PerceptualHasher
from desertfox.security import MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, MAX_PAGE_BYTES, safe_get

class WebHarvester:
    def __init__(self, user_agent: str = "DesertFox-OSINT-Defense-Audit/1.0 (+https://github.com/itthj/desertfox)"):
        self.headers = {"User-Agent": user_agent}
        self.seen_hashes: Set[str] = set()

    def extract_image_urls(self, page_url: str, max_images: int = 20) -> List[str]:
        """Scrapes a public webpage for absolute image URLs."""
        try:
            resp = safe_get(page_url, headers=self.headers, timeout=10, max_bytes=MAX_PAGE_BYTES)
            if resp.status_code != 200:
                return []
            
            soup = BeautifulSoup(resp.text, "html.parser")
            found_urls = []
            
            for img in soup.find_all("img"):
                src = img.get("src") or img.get("data-src")
                if src:
                    abs_url = urljoin(page_url, src)
                    # Filter for typical image extensions or data URLs
                    if any(abs_url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]) or "image" in abs_url:
                        if abs_url not in found_urls:
                            found_urls.append(abs_url)
                if len(found_urls) >= max_images:
                    break
                    
            return found_urls
        except Exception as e:
            print(f"[!] Error crawling {page_url}: {e}")
            return []

    def fetch_and_hash_image(self, image_url: str) -> Optional[Dict[str, Any]]:
        """
        Downloads image bytes, parses via PIL, and computes dHash.
        Returns None if image is duplicate or invalid.
        """
        try:
            resp = safe_get(image_url, headers=self.headers, timeout=10, max_bytes=MAX_IMAGE_BYTES)
            if resp.status_code != 200 or len(resp.content) < 1024:  # skip tiny 1px tracking pixels
                return None

            img = Image.open(io.BytesIO(resp.content))
            if img.size[0] * img.size[1] > MAX_IMAGE_PIXELS:
                return None
            # Verify image is valid
            img.verify()
            # Reopen for actual processing (verify closes the image stream)
            img = Image.open(io.BytesIO(resp.content))

            # Compute perceptual hash
            dhash = PerceptualHasher.compute_dhash(img)
            if dhash in self.seen_hashes:
                return None  # Duplicate

            self.seen_hashes.add(dhash)
            return {
                "image": img,
                "dhash": dhash,
                "url": image_url,
                "size": img.size,
                "format": img.format
            }
        except Exception:
            return None
