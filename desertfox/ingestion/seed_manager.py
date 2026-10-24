"""
Curated Seed Hub and Bulk URL / Sitemap Ingestion Manager for DesertFox.
Allows batch crawling of hundreds of targeted public directories, company rosters,
and university staff portals.
"""

import re
import os
import json
import time
import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Set, Optional
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


class SeedScheduleRegistry:
    """Manages persistent recurring crawl targets and scheduler configuration."""

    DEFAULT_STORE_PATH = "./data/crawl_schedules.json"

    def __init__(self, store_path: Optional[str] = None):
        self.store_path = store_path or os.environ.get("DESERTFOX_SCHEDULES_PATH", self.DEFAULT_STORE_PATH)
        self.schedules: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Loads schedule entries from disk or initializes defaults."""
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    self.schedules = json.load(f)
                return
            except Exception as e:
                logger.warning("Failed to load schedules from %s: %s", self.store_path, e)

        self.schedules = {}
        self._init_defaults()

    def _init_defaults(self) -> None:
        """Populates default high-probability scheduled targets."""
        default_targets = [
            ("bundle", "tech_execs", "Tech Execs & Leadership Rosters", 12, 10),
            ("bundle", "academic_rosters", "Academic Faculty Rosters", 24, 10),
            ("bundle", "open_contributors", "Open Source & Security Rosters", 24, 10),
        ]
        for t_type, target, name, interval, max_imgs in default_targets:
            self.register(target=target, target_type=t_type, name=name, interval_hours=interval, max_images=max_imgs, save=False)
        self.save()

    def save(self) -> None:
        """Atomically saves schedule dictionary to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(self.store_path)), exist_ok=True)
        tmp = f"{self.store_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.schedules, f, indent=2)
        os.replace(tmp, self.store_path)

    def register(
        self,
        target: str,
        target_type: str = "url",
        name: str = "",
        interval_hours: int = 12,
        max_images: int = 15,
        save: bool = True
    ) -> Dict[str, Any]:
        """Registers a recurring target for periodic indexing."""
        sched_id = f"sched_{hashlib.sha256(f'{target_type}:{target}'.encode()).hexdigest()[:10]}"
        now_iso = datetime.now(timezone.utc).isoformat()
        entry = {
            "id": sched_id,
            "name": name or target,
            "target_type": target_type,
            "target": target,
            "interval_hours": interval_hours,
            "max_images": max_images,
            "status": "active",
            "last_run": None,
            "last_faces_indexed": 0,
            "total_faces_indexed": 0,
            "created_at": now_iso
        }
        self.schedules[sched_id] = entry
        if save:
            self.save()
        return entry

    def delete(self, sched_id: str) -> bool:
        """Deletes a schedule by ID."""
        if sched_id in self.schedules:
            del self.schedules[sched_id]
            self.save()
            return True
        return False

    def toggle(self, sched_id: str) -> Optional[Dict[str, Any]]:
        """Toggles schedule between 'active' and 'paused'."""
        if sched_id in self.schedules:
            curr = self.schedules[sched_id].get("status", "active")
            self.schedules[sched_id]["status"] = "paused" if curr == "active" else "active"
            self.save()
            return self.schedules[sched_id]
        return None

    def get_all(self) -> List[Dict[str, Any]]:
        """Returns list of all schedule configurations."""
        return list(self.schedules.values())

    def get_due_schedules(self) -> List[Dict[str, Any]]:
        """Returns schedules that are currently due for execution."""
        due = []
        now_ts = time.time()
        for s in self.schedules.values():
            if s.get("status") != "active":
                continue
            last_run_str = s.get("last_run")
            if not last_run_str:
                due.append(s)
                continue
            try:
                last_dt = datetime.fromisoformat(last_run_str)
                interval_secs = s.get("interval_hours", 12) * 3600
                if (now_ts - last_dt.timestamp()) >= interval_secs:
                    due.append(s)
            except Exception:
                due.append(s)
        return due

    def record_run(self, sched_id: str, new_faces: int = 0) -> None:
        """Records execution timestamp and count of indexed faces."""
        if sched_id in self.schedules:
            self.schedules[sched_id]["last_run"] = datetime.now(timezone.utc).isoformat()
            self.schedules[sched_id]["last_faces_indexed"] = new_faces
            self.schedules[sched_id]["total_faces_indexed"] = self.schedules[sched_id].get("total_faces_indexed", 0) + new_faces
            self.save()
