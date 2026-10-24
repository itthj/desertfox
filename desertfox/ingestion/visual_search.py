"""
Reverse Image & Visual Search Harvester for DesertFox (PimEyes-style Probe Discovery).
Uploads a probe portrait (or aligned face crop) to zero-API-key visual search engines
(Yandex CBIR Visual Search & Bing Visual Search) to discover visually similar faces,
matching web pages, fast CDN thumbnails, and detected identity tags across the open web.
"""

import io
import re
import html
import json
import logging
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
import requests
from PIL import Image

from desertfox.ingestion.search_aggregator import SearchAggregator, USER_AGENT

logger = logging.getLogger(__name__)


class VisualSearchHarvester:
    """
    Zero-API-key reverse image search harvester.
    Finds pages and visually similar portraits directly from an uploaded face image.
    """

    def __init__(self, timeout: int = 12):
        self.timeout = timeout
        self.headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    @staticmethod
    def _encode_probe_jpeg(image: Image.Image, max_side: int = 600) -> bytes:
        """Resizes probe image to a compact JPEG for fast upload over slow links."""
        rgb = image.convert("RGB")
        w, h = rgb.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            rgb = rgb.resize((max(64, int(w * scale)), max(64, int(h * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=84, optimize=True)
        return buf.getvalue()

    @staticmethod
    def _normalize_url(url: str) -> str:
        return SearchAggregator._normalize_url(url)

    @staticmethod
    def _extract_domain(url: str, fallback: str = "web") -> str:
        return SearchAggregator._extract_domain(url, fallback)

    def search_yandex_visual(self, image: Image.Image, max_results: int = 50) -> Dict[str, Any]:
        """
        Uploads probe image to Yandex CBIR Visual Search and harvests visually similar
        portraits, fast avatars.mds.yandex.net thumbnails, source URLs, and entity tags.
        """
        candidates: List[Dict[str, Any]] = []
        detected_tags: List[str] = []
        cbir_id = ""
        seen = set()

        try:
            jpeg_bytes = self._encode_probe_jpeg(image)
            upload_url = (
                "https://yandex.com/images/touch/search?rpt=imageview&format=json"
                "&request=%7B%22blocks%22%3A%5B%7B%22block%22%3A%22cbir-uploader__get-cbir-id%22%7D%5D%7D"
            )
            up_resp = requests.post(
                upload_url,
                files={"upfile": ("probe.jpg", jpeg_bytes, "image/jpeg")},
                headers=self.headers,
                timeout=self.timeout,
            )
            if up_resp.status_code != 200:
                return {"candidates": [], "items": [], "detected_tags": [], "cbir_id": ""}

            try:
                payload = up_resp.json()
                if isinstance(payload, dict):
                    cbir_id = str(payload.get("cbirId") or payload.get("cbir_id") or "")
                    blocks = payload.get("blocks") or []
                    if not cbir_id and isinstance(blocks, list) and blocks and isinstance(blocks[0], dict):
                        params = blocks[0].get("params") or {}
                        cbir_id = str(params.get("cbirId") or "")
            except Exception:
                pass

            if not cbir_id:
                raw_up_text = str(getattr(up_resp, "text", "") or "")
                raw_up_url = str(getattr(up_resp, "url", "") or "")
                m_id = (
                    re.search(r'"cbirId"\s*:\s*"([^"]+)"', raw_up_text)
                    or re.search(r"cbir_id=([^&\"'\s]+)", raw_up_url)
                    or re.search(r"cbir_id=([^&\"'\s]+)", raw_up_text)
                )
                if m_id:
                    cbir_id = urllib.parse.unquote(m_id.group(1))

            if not cbir_id:
                return {"candidates": [], "items": [], "detected_tags": [], "cbir_id": ""}

            sim_url = (
                f"https://yandex.com/images/search?rpt=imageview"
                f"&cbir_id={urllib.parse.quote(cbir_id)}&cbir_page=similar"
            )
            sim_resp = requests.get(sim_url, headers=self.headers, timeout=self.timeout)
            if sim_resp.status_code != 200:
                return {"candidates": [], "items": [], "detected_tags": [], "cbir_id": cbir_id}

            raw_html = sim_resp.text
            unesc = html.unescape(raw_html)

            # Extract any detected entity/person tags from Yandex CBIR
            for tag_pat in (
                r'"CbirTags"[^}]*?"text"\s*:\s*"([^"]+)"',
                r'"CbirTags"\s*:\s*\[\s*"([^"]+)"',
                r'class=["\'][^"\']*cbir-similar__tag[^"\']*["\'][^>]*>([^<]+)<',
                r'"tags"\s*:\s*\[\s*\{\s*"text"\s*:\s*"([^"]+)"',
            ):
                for t in re.findall(tag_pat, unesc):
                    t_clean = t.strip()
                    if t_clean and len(t_clean) > 2 and t_clean not in detected_tags:
                        detected_tags.append(t_clean)

            # 1. Structured serp-item JSON blocks
            bem_matches = re.findall(r"data-bem='(\{\"serp-item\":[^']+)'", raw_html)
            if not bem_matches:
                bem_matches = re.findall(r'data-bem="(\{&quot;serp-item&quot;:[^"]+)"', raw_html)

            for raw_bem in bem_matches:
                try:
                    data = json.loads(html.unescape(raw_bem))
                    serp = data.get("serp-item", {})
                    img_u = self._normalize_url(serp.get("img_href") or "")
                    if not img_u and serp.get("preview"):
                        img_u = self._normalize_url(serp["preview"][0].get("url") or "")
                    thumb_u = self._normalize_url((serp.get("thumb") or {}).get("url") or "")
                    snippet = serp.get("snippet", {})
                    page_u = self._normalize_url(snippet.get("url") or img_u)
                    title = snippet.get("title") or snippet.get("text") or "Yandex Visual Match"
                    if img_u and img_u.startswith("http") and img_u not in seen:
                        seen.add(img_u)
                        candidates.append({
                            "image_url": img_u,
                            "thumb_url": thumb_u or img_u,
                            "source_page_url": page_u or img_u,
                            "title": html.unescape(re.sub(r"<[^>]+>", "", str(title))),
                            "domain": snippet.get("domain") or self._extract_domain(page_u or img_u),
                            "engine": "YandexVisual",
                        })
                        if len(candidates) >= max_results:
                            return {"candidates": candidates, "detected_tags": detected_tags, "cbir_id": cbir_id}
                except Exception:
                    continue

            # 2. Fast CDN thumbnail + original URL extraction from CBIR React/SSR state
            mds_thumbs = [
                self._normalize_url(u)
                for u in re.findall(r'(?:https?:)?//avatars\.mds\.yandex\.net/i\?id=[^"\'\s&<>\\]+', unesc)
            ]
            orig_urls = [
                self._normalize_url(u)
                for u in re.findall(
                    r'"(?:img_href|origUrl|url)"\s*:\s*"(https?://[^"\s<>]+?\.(?:jpg|jpeg|png|webp)(?:\?[^"\s<>]*)?)"',
                    unesc,
                    flags=re.IGNORECASE,
                )
                if "yastatic.net" not in u and "yandex." not in u
            ]

            # Pair thumbnails and original URLs
            max_pair = max(len(mds_thumbs), len(orig_urls))
            for idx in range(max_pair):
                thumb_u = mds_thumbs[idx] if idx < len(mds_thumbs) else ""
                orig_u = orig_urls[idx] if idx < len(orig_urls) else thumb_u
                primary = orig_u or thumb_u
                if primary and primary.startswith("http") and primary not in seen:
                    seen.add(primary)
                    candidates.append({
                        "image_url": primary,
                        "thumb_url": thumb_u or primary,
                        "source_page_url": orig_u or sim_url,
                        "title": (detected_tags[0] if detected_tags else "Visual Match") + f" #{len(candidates) + 1}",
                        "domain": self._extract_domain(orig_u or primary, "yandex.com"),
                        "engine": "YandexVisual",
                    })
                    if len(candidates) >= max_results:
                        break

        except Exception as exc:
            logger.warning("Yandex visual reverse search failed: %s", exc)

        return {
            "candidates": candidates[:max_results],
            "items": candidates[:max_results],
            "detected_tags": detected_tags,
            "cbir_id": cbir_id,
        }

    def search_bing_visual(self, image: Image.Image, max_results: int = 30) -> Dict[str, Any]:
        """
        Uploads probe image to Bing Visual Search (sbiupload) and extracts matching
        thumbnails and source pages.
        """
        candidates: List[Dict[str, Any]] = []
        seen = set()
        try:
            jpeg_bytes = self._encode_probe_jpeg(image)
            url = "https://www.bing.com/images/search?view=detailv2&iss=sbiupload&FORM=SBIHMP"
            resp = requests.post(
                url,
                files={"imageBin": ("probe.jpg", jpeg_bytes, "image/jpeg")},
                headers=self.headers,
                timeout=self.timeout,
                allow_redirects=True,
            )
            if resp.status_code != 200:
                return {"candidates": []}

            unesc = html.unescape(resp.text)
            for m in re.findall(r'm="(\{[^"]+\})"', resp.text):
                try:
                    obj = json.loads(html.unescape(m))
                    img_u = self._normalize_url(obj.get("murl") or obj.get("turl") or "")
                    t_u = self._normalize_url(obj.get("turl") or "")
                    page_u = self._normalize_url(obj.get("purl") or img_u)
                    if img_u and img_u.startswith("http") and img_u not in seen:
                        seen.add(img_u)
                        candidates.append({
                            "image_url": img_u,
                            "thumb_url": t_u or img_u,
                            "source_page_url": page_u or img_u,
                            "title": obj.get("t", "") or "Bing Visual Match",
                            "domain": self._extract_domain(page_u or img_u),
                            "engine": "BingVisual",
                        })
                        if len(candidates) >= max_results:
                            return {"candidates": candidates}
                except Exception:
                    continue

            tse_thumbs = [
                self._normalize_url(u)
                for u in re.findall(r'https?://tse\d+\.mm\.bing\.net/th\?id=[^"\'\s&<>\\]+', unesc)
            ]
            for t_u in tse_thumbs:
                if t_u not in seen:
                    seen.add(t_u)
                    candidates.append({
                        "image_url": t_u,
                        "thumb_url": t_u,
                        "source_page_url": resp.url or url,
                        "title": f"Bing Visual Match #{len(candidates) + 1}",
                        "domain": "bing.com",
                        "engine": "BingVisual",
                    })
                    if len(candidates) >= max_results:
                        break
        except Exception as exc:
            logger.warning("Bing visual reverse search failed: %s", exc)

        return {"candidates": candidates[:max_results]}

    def discover_from_probe(
        self,
        image: Optional[Image.Image] = None,
        query_hint: str = "",
        max_results: int = 60,
        search_aggregator: Optional[SearchAggregator] = None,
        probe_image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """
        Combined Visual + Entity OSINT Discovery:
        1. Runs Yandex CBIR Visual Search and Bing Visual Search concurrently on the probe face.
        2. If `query_hint` is supplied (or an entity tag is detected by CBIR), also runs
           multi-engine text/portrait search in parallel.
        3. Returns deduplicated candidates with fast CDN thumbnails ready for parallel indexing.
        """
        target_img = probe_image if probe_image is not None else image
        if target_img is None:
            return {"candidates": [], "detected_tags": [], "cbir_id": "", "provider_counts": {}}
        aggregator = search_aggregator or SearchAggregator()
        hint = (query_hint or "").strip()

        with ThreadPoolExecutor(max_workers=3) as pool:
            f_yandex = pool.submit(self.search_yandex_visual, target_img, max_results)
            f_bing = pool.submit(self.search_bing_visual, target_img, max(20, max_results // 2))
            f_text = (
                pool.submit(aggregator.aggregate_search, hint, max_results)
                if hint
                else None
            )

            y_res = f_yandex.result()
            b_res = f_bing.result()
            t_items = f_text.result() if f_text else []

        detected_tags = y_res.get("detected_tags") or []
        # If user didn't pass a query hint but Yandex CBIR identified a person/tag, expand on it
        if not hint and detected_tags:
            auto_query = detected_tags[0]
            try:
                t_items = aggregator.aggregate_search(auto_query, max_total=max(20, max_results // 2))
            except Exception:
                t_items = []

        combined: List[Dict[str, Any]] = []
        seen = set()
        provider_counts: Dict[str, int] = {}

        # Interleave visual matches and text-expanded matches
        lists = [y_res.get("candidates", []), b_res.get("candidates", []), t_items]
        max_len = max((len(lst) for lst in lists), default=0)
        for idx in range(max_len):
            for lst in lists:
                if idx < len(lst):
                    item = lst[idx]
                    u = item.get("image_url") or item.get("thumb_url")
                    if u and u not in seen:
                        seen.add(u)
                        eng = item.get("engine", "Visual")
                        provider_counts[eng] = provider_counts.get(eng, 0) + 1
                        combined.append(item)
                        if len(combined) >= max_results:
                            return {
                                "candidates": combined,
                                "detected_tags": detected_tags,
                                "cbir_id": y_res.get("cbir_id", ""),
                                "provider_counts": provider_counts,
                            }

        return {
            "candidates": combined[:max_results],
            "detected_tags": detected_tags,
            "cbir_id": y_res.get("cbir_id", ""),
            "provider_counts": provider_counts,
        }
