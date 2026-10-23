import os
import logging
import re
import requests
from urllib.parse import urlparse, urljoin

from desertfox.ingestion.site_crawler import SiteCrawler, normalize_url

logger = logging.getLogger(__name__)


class DesertFoxEngine:
    # ... existing class body continues below; this method is appended to the class.

    def expand_seed_urls_from_query(self, query: str, limit: int = 20) -> List[str]:
        """Uses a lightweight public search result page to expand an audit beyond a single URL."""
        text = (query or "").strip()
        if not text:
            return []
        try:
            response = requests.get(
                "https://html.duckduckgo.com/html/",
                params={"q": text},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=12,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning("Search expansion failed for %s: %s", text, exc)
            return []

        urls: List[str] = []
        seen: set[str] = set()
        for match in re.findall(r'<a rel="nofollow" class="result-link" href="([^"]+)"', response.text):
            candidate = normalize_url(match)
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            urls.append(candidate)
            if len(urls) >= limit:
                break
        if urls:
            return urls

        # Fallback: extract any http(s) urls from the search page.
        for match in re.findall(r'https?://[^\"\'\s>]+', response.text):
            candidate = normalize_url(match)
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            urls.append(candidate)
            if len(urls) >= limit:
                break
        return urls

    def audit_sites(
        self,
        seed_urls: List[str],
        max_pages: int = 30,
        max_images: int = 10,
        max_sites: int = 10,
    ) -> Dict[str, Any]:
        """Expand a broader audit across multiple seeds and aggregate all discovered pages."""
        seeds: List[str] = []
        seen_seeds: set[str] = set()
        for candidate in (seed_urls or []):
            normalized = normalize_url(candidate)
            if normalized and normalized not in seen_seeds:
                seen_seeds.add(normalized)
                seeds.append(normalized)
            if len(seeds) >= max_sites:
                break

        if not seeds:
            return {"status": "error", "message": "No valid seed URLs supplied for the extended audit.", "newly_indexed_images": 0, "pages": []}

        all_pages: List[str] = []
        seen_pages: set[str] = set()
        for seed in seeds:
            crawler = SiteCrawler(
                user_agent=self.harvester.headers["User-Agent"],
                delay_s=max(self.config.rate_limit_ms, 0) / 1000.0,
            )
            discovered = crawler.discover(seed, max_pages=max_pages, max_depth=self.config.audit_max_depth)
            if discovered.get("error"):
                continue
            for page in discovered.get("pages", []):
                if page not in seen_pages:
                    seen_pages.add(page)
                    all_pages.append(page)

        reports = []
        total_indexed = 0
        skipped_other_faces = 0
        for page in all_pages:
            try:
                result = self.crawl_and_index_target(
                    page,
                    max_images=max_images,
                    use_async=False,
                    persist=False,
                    only_enrolled=False,
                )
            except Exception as exc:
                reports.append({"url": page, "status": "error", "indexed": 0, "error": str(exc)})
                continue
            indexed = int(result.get("newly_indexed_images", 0))
            skipped_other_faces += int(result.get("skipped_other_faces", 0))
            total_indexed += indexed
            reports.append({
                "url": page,
                "status": result.get("status", "error"),
                "indexed": indexed,
                "candidates": result.get("candidates_found", 0),
                "skipped_other_faces": result.get("skipped_other_faces", 0),
            })

        if total_indexed > 0:
            self.partitioned_index.build_partitions()
            self.save_index()

        return {
            "status": "success",
            "seed_urls": seeds,
            "pages_discovered": len(all_pages),
            "pages_skipped_robots": 0,
            "newly_indexed_images": total_indexed,
            "skipped_other_faces": skipped_other_faces,
            "pages": reports,
        }

    def audit_site(self, seed_url: str, max_pages: int = 30, max_images: int = 10, max_sites: int = 10) -> Dict[str, Any]:
        """Backward-compatible single-site wrapper for the broader multi-site audit."""
        return self.audit_sites([seed_url], max_pages=max_pages, max_images=max_images, max_sites=max_sites)
