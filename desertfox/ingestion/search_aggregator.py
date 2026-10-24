"""
Multi-Engine Search Aggregator for DesertFox.
Crawls public search engines (DuckDuckGo, Bing Images) to discover candidate photos
and public webpage appearances for a target name, username, or identifier.
Operates without proprietary or paid API keys.
"""

import re
import json
import logging
import urllib.parse
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


class SearchAggregator:
    """
    Automated OSINT search engine aggregator.
    Discovers image exposure across the web for names, handles, or identities.
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.headers = {"User-Agent": USER_AGENT}

    def search_duckduckgo_images(self, query: str, max_results: int = 35) -> List[Dict[str, Any]]:
        """
        Queries DuckDuckGo Image Search and extracts candidate image metadata.
        """
        results = []
        try:
            # Step 1: Fetch VQD authentication token from DuckDuckGo search page
            token_url = f"https://duckduckgo.com/?q={urllib.parse.quote(query)}"
            resp1 = requests.get(token_url, headers=self.headers, timeout=self.timeout)
            if resp1.status_code != 200:
                return results

            vqd_match = re.search(r'vqd=([\d\-]+)', resp1.text) or re.search(r'vqd=\"([^\"]+)\"', resp1.text)
            if not vqd_match:
                return results

            vqd = vqd_match.group(1)

            # Step 2: Fetch image payload using VQD token
            api_url = f"https://duckduckgo.com/i.js?l=us-en&o=json&q={urllib.parse.quote(query)}&vqd={vqd}"
            resp2 = requests.get(api_url, headers=self.headers, timeout=self.timeout)
            if resp2.status_code != 200:
                return results

            data = resp2.json()
            raw_items = data.get("results", [])

            for item in raw_items[:max_results]:
                img_url = item.get("image")
                page_url = item.get("url")
                title = item.get("title", "")
                source = item.get("source", "")
                if img_url:
                    results.append({
                        "image_url": img_url,
                        "source_page_url": page_url or img_url,
                        "title": title,
                        "domain": source or urllib.parse.urlparse(page_url or img_url).netloc,
                        "engine": "DuckDuckGo"
                    })
        except Exception as e:
            logger.warning("DuckDuckGo image search failed: %s", e)

        return results

    def search_bing_images(self, query: str, max_results: int = 35) -> List[Dict[str, Any]]:
        """
        Scrapes Bing Images and extracts high-resolution candidate image metadata.
        """
        results = []
        try:
            url = f"https://www.bing.com/images/search?q={urllib.parse.quote(query)}&form=HDRSC2&first=1"
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            if resp.status_code != 200:
                return results

            # Bing stores JSON payload in m="..." attribute on anchor tags
            matches = re.findall(r'm=\"(\{[^\"]+\})\"', resp.text)
            for m in matches[:max_results]:
                try:
                    obj = json.loads(m.replace('&quot;', '"'))
                    img_url = obj.get("murl")
                    page_url = obj.get("purl")
                    title = obj.get("t", "")
                    if img_url:
                        domain = urllib.parse.urlparse(page_url or img_url).netloc
                        results.append({
                            "image_url": img_url,
                            "source_page_url": page_url or img_url,
                            "title": title,
                            "domain": domain,
                            "engine": "Bing"
                        })
                except Exception:
                    continue
        except Exception as e:
            logger.warning("Bing image search failed: %s", e)

        return results

    def aggregate_search(
        self,
        query: str,
        max_total: int = 50,
        engines: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Aggregates image candidates across multiple search engines with deduplication.
        """
        target_engines = engines or ["duckduckgo", "bing"]
        all_candidates: List[Dict[str, Any]] = []
        seen_urls = set()

        per_engine = max(15, max_total // len(target_engines))

        if "duckduckgo" in target_engines:
            ddg_items = self.search_duckduckgo_images(query, max_results=per_engine)
            for it in ddg_items:
                u = it["image_url"]
                if u not in seen_urls:
                    seen_urls.add(u)
                    all_candidates.append(it)

        if "bing" in target_engines:
            bing_items = self.search_bing_images(query, max_results=per_engine)
            for it in bing_items:
                u = it["image_url"]
                if u not in seen_urls:
                    seen_urls.add(u)
                    all_candidates.append(it)

        return all_candidates[:max_total]
