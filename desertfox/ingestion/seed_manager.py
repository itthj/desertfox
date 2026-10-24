"""
Curated Seed Hub and Bulk URL / Sitemap Ingestion Manager for DesertFox.
Allows batch crawling of hundreds of targeted public directories, company rosters,
and university staff portals.
"""

import re
import logging
from typing import List, Dict, Any, Set
import xml.etree.ElementTree as ET
import requests

logger = logging.getLogger(__name__)

USER_AGENT = "DesertFox-OSINT-Defense-Audit/3.0 (+https://github.com/itthj/desertfox)"


class SeedHubManager:
    """Manages curated seed directories and bulk URL discovery."""

    # Pre-packaged curated seed bundles for out-of-the-box wide-spectrum indexing
    CURATED_BUNDLES: Dict[str, Dict[str, Any]] = {
        "tech_execs": {
            "name": "Global Tech & Startup Directories",
            "description": "Leadership and public team rosters of prominent tech organizations.",
            "urls": [
                "https://about.meta.com/company-info/",
                "https://www.apple.com/leadership/",
                "https://www.microsoft.com/en-us/about/leadership",
                "https://about.google/our-story/",
                "https://openai.com/about/",
                "https://www.anthropic.com/company",
                "https://github.com/about",
            ]
        },
        "academic_rosters": {
            "name": "Academic & Research Faculty Portals",
            "description": "Public computer science and AI research staff directories.",
            "urls": [
                "https://www.cs.stanford.edu/people/faculty",
                "https://www.eecs.mit.edu/people/faculty-advisors/",
                "https://eecs.berkeley.edu/people/faculty",
                "https://www.cs.cmu.edu/directory/faculty",
                "https://www.ox.ac.uk/about/organisation",
            ]
        },
        "open_contributors": {
            "name": "Open Source & Security Community Portals",
            "description": "Public contributors, core teams, and speakers.",
            "urls": [
                "https://www.linuxfoundation.org/about/leadership",
                "https://www.apache.org/foundation/leadership.html",
                "https://owasp.org/www-staff/",
                "https://www.eff.org/about/staff",
            ]
        }
    }

    @staticmethod
    def parse_sitemap_urls(sitemap_url: str, max_urls: int = 50, timeout: int = 10) -> List[str]:
        """
        Fetches and parses an XML sitemap (or sitemap index) to discover page URLs.
        Filters for team, about, staff, and person pages where portraits reside.
        """
        urls: List[str] = []
        try:
            headers = {"User-Agent": USER_AGENT}
            resp = requests.get(sitemap_url, headers=headers, timeout=timeout)
            if resp.status_code != 200:
                return urls

            root = ET.fromstring(resp.content)
            # Handle XML namespaces
            ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

            for loc in root.findall(".//ns:loc", ns):
                if loc.text:
                    u = loc.text.strip()
                    urls.append(u)
                    if len(urls) >= max_urls * 2:
                        break

            # Prioritize relevant keywords
            prioritized = []
            others = []
            keywords = ["team", "about", "people", "staff", "leadership", "member", "author", "user", "profile"]
            for u in urls:
                if any(kw in u.lower() for kw in keywords):
                    prioritized.append(u)
                else:
                    others.append(u)

            final_list = (prioritized + others)[:max_urls]
            return final_list
        except Exception as e:
            logger.warning("Failed to parse sitemap %s: %s", sitemap_url, e)
            return []

    @staticmethod
    def extract_urls_from_text(raw_text: str, max_urls: int = 200) -> List[str]:
        """Extracts valid HTTP/HTTPS URLs from raw text, CSV, or line-separated inputs."""
        url_regex = r'https?://[^\s,;"\'<>]+'
        found = re.findall(url_regex, raw_text)
        cleaned = []
        seen = set()
        for u in found:
            u_clean = u.rstrip("./,")
            if u_clean not in seen:
                seen.add(u_clean)
                cleaned.append(u_clean)
                if len(cleaned) >= max_urls:
                    break
        return cleaned

    @classmethod
    def get_bundle_urls(cls, bundle_id: str) -> List[str]:
        """Returns URLs for a curated bundle ID."""
        bundle = cls.CURATED_BUNDLES.get(bundle_id)
        return bundle["urls"] if bundle else []
