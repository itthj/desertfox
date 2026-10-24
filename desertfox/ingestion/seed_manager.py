"""
Curated Seed Hub and Bulk URL / Sitemap Ingestion Manager for DesertFox.
Allows batch crawling of hundreds of targeted public directories, company rosters,
AI labs, venture directories, security conferences, and university staff portals.
"""

import re
import os
import gzip
import json
import time
import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET
import requests

logger = logging.getLogger(__name__)

USER_AGENT = "DesertFox-OSINT-Defense-Audit/3.0 (+https://github.com/itthj/desertfox)"


class SeedHubManager:
    """Manages curated seed directories and bulk URL discovery."""

    # Pre-packaged curated seed bundles (134 URLs across 8 high-yield OSINT sectors)
    CURATED_BUNDLES: Dict[str, Dict[str, Any]] = {
        "tech_execs": {
            "name": "Global Tech & Cloud Leadership",
            "description": "Leadership, executive bios, and board rosters of major technology and cloud platforms.",
            "urls": [
                "https://about.meta.com/company-info/",
                "https://www.apple.com/leadership/",
                "https://www.microsoft.com/en-us/about/leadership",
                "https://about.google/our-story/",
                "https://openai.com/about/",
                "https://www.anthropic.com/company",
                "https://github.com/about",
                "https://www.nvidia.com/en-us/about-nvidia/corporate-timeline/",
                "https://www.amd.com/en/corporate/leadership.html",
                "https://www.intel.com/content/www/us/en/company-overview/leadership.html",
                "https://www.cloudflare.com/about-overview/",
                "https://stripe.com/newsroom",
                "https://www.databricks.com/company/leadership",
                "https://huggingface.co/huggingface",
                "https://cohere.com/about",
                "https://mistral.ai/company/",
                "https://scale.com/about",
                "https://www.palantir.com/about/",
                "https://www.snowflake.com/en/company/overview/leadership/",
                "https://vercel.com/about",
                "https://linear.app/about",
                "https://supabase.com/company",
            ],
        },
        "academic_rosters": {
            "name": "Academic & University Faculty Portals",
            "description": "Public computer science, engineering, and AI faculty directories at top research universities.",
            "urls": [
                "https://www.cs.stanford.edu/people/faculty",
                "https://www.eecs.mit.edu/people/faculty-advisors/",
                "https://eecs.berkeley.edu/people/faculty",
                "https://www.cs.cmu.edu/directory/faculty",
                "https://www.ox.ac.uk/about/organisation",
                "https://www.cst.cam.ac.uk/people",
                "https://inf.ethz.ch/people/professors.html",
                "https://www.cms.caltech.edu/people/faculty",
                "https://www.cs.princeton.edu/people/faculty",
                "https://seas.harvard.edu/faculty-research/people",
                "https://www.cs.cornell.edu/people/faculty",
                "https://www.cs.washington.edu/people/faculty",
                "https://www.cc.gatech.edu/people/faculty",
                "https://siebelschool.illinois.edu/about/people/all-faculty",
                "https://cs.nyu.edu/dynamic/people/faculty/",
                "https://www.cs.columbia.edu/people/faculty/",
                "https://www.ucl.ac.uk/computer-science/people/academic-staff",
                "https://www.imperial.ac.uk/computing/people/academic-staff/",
                "https://web.cs.toronto.edu/people/faculty-directory",
                "https://www.cs.utexas.edu/people",
                "https://www.cse.umich.edu/cse/people/faculty/",
                "https://www.cs.ucla.edu/faculty/",
            ],
        },
        "open_contributors": {
            "name": "Open Source & Digital Rights Foundations",
            "description": "Public maintainers, technical steering committees, and foundation leadership.",
            "urls": [
                "https://www.linuxfoundation.org/about/leadership",
                "https://www.apache.org/foundation/leadership.html",
                "https://owasp.org/www-staff/",
                "https://www.eff.org/about/staff",
                "https://foundation.mozilla.org/en/who-we-are/",
                "https://www.python.org/psf/membership/",
                "https://foundation.rust-lang.org/about/",
                "https://www.cncf.io/people/staff/",
                "https://www.fsf.org/about/staff-and-board",
                "https://wikimediafoundation.org/role/staff-contractors/",
                "https://www.torproject.org/about/people/",
                "https://signal.org/foundation/",
                "https://creativecommons.org/about/team/",
                "https://www.internetsociety.org/about-internet-society/organization/staff/",
                "https://www.w3.org/staff/",
                "https://epic.org/about/staff/",
                "https://freedom.press/about/staff/",
                "https://www.accessnow.org/team/",
            ],
        },
        "ai_research_labs": {
            "name": "Frontier AI & ML Research Institutes",
            "description": "Principal scientists, fellows, and research staff across global AI laboratories.",
            "urls": [
                "https://deepmind.google/about/",
                "https://ai.meta.com/research/",
                "https://www.microsoft.com/en-us/research/people/",
                "https://research.ibm.com/people",
                "https://allenai.org/team",
                "https://hai.stanford.edu/people",
                "https://bair.berkeley.edu/faculty.html",
                "https://www.lti.cs.cmu.edu/people/faculty",
                "https://www.media.mit.edu/people/",
                "https://vectorinstitute.ai/team/",
                "https://mila.quebec/en/directory",
                "https://www.tri.global/our-team",
                "https://www.nvidia.com/en-us/research/",
                "https://machinelearning.apple.com/",
                "https://www.eleuther.ai/about",
                "https://laion.ai/team/",
            ],
        },
        "venture_founders": {
            "name": "Venture Capital & Startup Ecosystems",
            "description": "General partners, operating partners, and founder directories at top venture firms.",
            "urls": [
                "https://www.ycombinator.com/people",
                "https://www.sequoiacap.com/our-team/",
                "https://a16z.com/about/",
                "https://www.benchmark.com/",
                "https://www.accel.com/team",
                "https://greylock.com/team/",
                "https://lsvp.com/team/",
                "https://www.indexventures.com/team/",
                "https://www.bvp.com/team",
                "https://foundersfund.com/team/",
                "https://www.khoslaventures.com/team/",
                "https://www.generalcatalyst.com/team",
                "https://www.gv.com/team/",
                "https://www.nea.com/team",
                "https://firstround.com/team/",
                "https://www.insightpartners.com/team/",
            ],
        },
        "cybersecurity_conferences": {
            "name": "Cybersecurity & Infosec Speaker Rosters",
            "description": "Review boards, keynote speakers, trainers, and security researchers.",
            "urls": [
                "https://www.blackhat.com/html/review-board.html",
                "https://defcon.org/html/links/dc-speakerscorner.html",
                "https://www.rsaconference.com/experts",
                "https://www.usenix.org/about/board-of-directors",
                "https://www.first.org/about/board",
                "https://www.sans.org/instructors/",
                "https://www.offsec.com/about/",
                "https://www.hackthebox.com/about-us",
                "https://www.bugcrowd.com/about/leadership/",
                "https://www.hackerone.com/company/leadership",
                "https://www.trailofbits.com/about",
                "https://www.nccgroup.com/uk/about-us/our-leadership/",
                "https://www.crowdstrike.com/en-us/about-us/executive-team/",
                "https://www.paloaltonetworks.com/company/executive-team",
                "https://www.mandiant.com/company/leadership",
                "https://www.rapid7.com/about/leadership/",
            ],
        },
        "media_journalism": {
            "name": "Investigative Journalism & Tech Press",
            "description": "Editorial mastheads, investigative reporters, and technology columnists.",
            "urls": [
                "https://www.propublica.org/people",
                "https://theintercept.com/staff/",
                "https://www.wired.com/about/wired-masthead/",
                "https://arstechnica.com/staff-directory/",
                "https://www.technologyreview.com/about/",
                "https://www.theverge.com/about-the-verge",
                "https://techcrunch.com/about-techcrunch/",
                "https://restofworld.org/about/",
                "https://www.bellingcat.com/about/who-we-are/",
                "https://www.icij.org/about/our-team/",
                "https://www.niemanlab.org/staff/",
                "https://www.CJR.org/about_us",
            ],
        },
        "public_institutions": {
            "name": "Policy Institutes & Global Think Tanks",
            "description": "Senior fellows, policy scholars, and international governance directories.",
            "urls": [
                "https://www.weforum.org/about/leadership-and-governance/",
                "https://www.turing.ac.uk/people",
                "https://www.brookings.edu/experts/",
                "https://carnegieendowment.org/experts",
                "https://www.csis.org/people",
                "https://www.rand.org/about/people.html",
                "https://www.atlanticcouncil.org/experts/",
                "https://www.chathamhouse.org/about-us/our-people",
                "https://www.cfr.org/experts",
                "https://www.eff.org/about/board",
                "https://cdt.org/staff/",
                "https://futureoflife.org/about-us/our-people/",
            ],
        },
    }

    PORTRAIT_KEYWORDS = [
        "team", "about", "people", "staff", "leadership", "member",
        "author", "user", "profile", "faculty", "speaker", "bio",
        "board", "executive", "researcher", "founder", "partner",
        "contributor", "fellow", "director", "expert", "advisor",
    ]

    @classmethod
    def _extract_locs_from_xml(cls, xml_bytes: bytes) -> Dict[str, List[str]]:
        """
        Parses XML sitemap or sitemapindex content (namespace-agnostic) and returns
        {"urls": [...], "sitemaps": [...]}.
        """
        urls: List[str] = []
        child_sitemaps: List[str] = []
        try:
            root = ET.fromstring(xml_bytes)
            root_tag = root.tag.lower()
            is_index = "sitemapindex" in root_tag

            for elem in root.iter():
                tag_name = elem.tag.split("}")[-1].lower() if "}" in elem.tag else elem.tag.lower()
                if tag_name == "loc" and elem.text:
                    loc_val = elem.text.strip()
                    if not loc_val:
                        continue
                    if is_index or loc_val.endswith(".xml") or loc_val.endswith(".xml.gz") or "sitemap" in loc_val.lower():
                        if is_index:
                            child_sitemaps.append(loc_val)
                        else:
                            urls.append(loc_val)
                    else:
                        urls.append(loc_val)
        except Exception:
            # Regex fallback for malformed XML
            raw_text = xml_bytes.decode("utf-8", errors="ignore")
            locs = re.findall(r"<loc>\s*(https?://[^<\s]+)\s*</loc>", raw_text, flags=re.IGNORECASE)
            for loc_val in locs:
                if loc_val.endswith(".xml") or loc_val.endswith(".xml.gz"):
                    child_sitemaps.append(loc_val)
                else:
                    urls.append(loc_val)

        return {"urls": urls, "sitemaps": child_sitemaps}

    @classmethod
    def parse_sitemap_urls(
        cls,
        sitemap_url: str,
        max_urls: int = 150,
        timeout: int = 10,
        _depth: int = 0,
    ) -> List[str]:
        """
        Fetches and parses an XML sitemap (or recursive sitemap index, including .xml.gz)
        to discover page URLs. Prioritizes team, about, staff, faculty, and person pages.
        """
        urls: List[str] = []
        try:
            headers = {"User-Agent": USER_AGENT}
            resp = requests.get(sitemap_url, headers=headers, timeout=timeout)
            if resp.status_code != 200:
                return urls

            content = resp.content
            if sitemap_url.endswith(".gz") or content[:2] == b"\x1f\x8b":
                try:
                    content = gzip.decompress(content)
                except Exception:
                    pass

            parsed = cls._extract_locs_from_xml(content)
            urls.extend(parsed["urls"])

            # Recursively follow up to 4 child sitemaps if this was a sitemap index
            if parsed["sitemaps"] and _depth < 1 and len(urls) < max_urls:
                # Prioritize child sitemaps that mention people/team/staff/authors
                prio_children = [
                    s for s in parsed["sitemaps"]
                    if any(kw in s.lower() for kw in cls.PORTRAIT_KEYWORDS)
                ]
                other_children = [s for s in parsed["sitemaps"] if s not in prio_children]
                for child_url in (prio_children + other_children)[:4]:
                    sub_urls = cls.parse_sitemap_urls(
                        child_url,
                        max_urls=max_urls - len(urls),
                        timeout=timeout,
                        _depth=_depth + 1,
                    )
                    urls.extend(sub_urls)
                    if len(urls) >= max_urls * 2:
                        break

            # Deduplicate and prioritize portrait-relevant URLs
            seen = set()
            prioritized = []
            others = []
            for u in urls:
                if u in seen:
                    continue
                seen.add(u)
                if any(kw in u.lower() for kw in cls.PORTRAIT_KEYWORDS):
                    prioritized.append(u)
                else:
                    others.append(u)

            return (prioritized + others)[:max_urls]
        except Exception as e:
            logger.warning("Failed to parse sitemap %s: %s", sitemap_url, e)
            return []

    @staticmethod
    def extract_urls_from_text(raw_text: str, max_urls: int = 500) -> List[str]:
        """Extracts valid HTTP/HTTPS URLs from raw text, CSV, JSON, or line-separated inputs."""
        url_regex = r'https?://[^\s,;"\'<>\]\)]+'
        found = re.findall(url_regex, raw_text)
        cleaned = []
        seen = set()
        for u in found:
            u_clean = u.rstrip("./,;:)")
            if u_clean not in seen:
                seen.add(u_clean)
                cleaned.append(u_clean)
                if len(cleaned) >= max_urls:
                    break
        return cleaned

    @classmethod
    def get_bundle_urls(cls, bundle_id: str) -> List[str]:
        """Returns URLs for a curated bundle ID (or all bundles when bundle_id == 'all')."""
        if bundle_id == "all":
            return cls.get_all_bundle_urls()
        bundle = cls.CURATED_BUNDLES.get(bundle_id)
        return list(bundle["urls"]) if bundle else []

    @classmethod
    def get_all_bundle_urls(cls) -> List[str]:
        """Returns deduplicated list of all URLs across every curated bundle."""
        all_urls: List[str] = []
        seen = set()
        for bundle in cls.CURATED_BUNDLES.values():
            for u in bundle.get("urls", []):
                if u not in seen:
                    seen.add(u)
                    all_urls.append(u)
        return all_urls

    @classmethod
    def get_catalog_summary(cls) -> Dict[str, Any]:
        """Returns metadata and URL counts across all curated bundles."""
        total_urls = len(cls.get_all_bundle_urls())
        return {
            "total_bundles": len(cls.CURATED_BUNDLES),
            "total_curated_urls": total_urls,
            "bundles": {
                bid: {
                    "name": bdata["name"],
                    "description": bdata["description"],
                    "url_count": len(bdata["urls"]),
                    "urls": bdata["urls"],
                }
                for bid, bdata in cls.CURATED_BUNDLES.items()
            },
        }


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
        """Populates default high-probability scheduled targets across all curated bundles."""
        default_targets = [
            ("bundle", "tech_execs", "Global Tech & Cloud Leadership (22 URLs)", 12, 15),
            ("bundle", "academic_rosters", "University CS & AI Faculty Rosters (22 URLs)", 24, 15),
            ("bundle", "open_contributors", "Open Source & Digital Rights Rosters (18 URLs)", 24, 15),
            ("bundle", "ai_research_labs", "Frontier AI & ML Research Institutes (16 URLs)", 18, 15),
            ("bundle", "venture_founders", "Venture Capital & Founder Directories (16 URLs)", 24, 15),
            ("bundle", "cybersecurity_conferences", "Cybersecurity & Infosec Speaker Rosters (16 URLs)", 24, 15),
            ("bundle", "media_journalism", "Investigative Journalism & Tech Press (12 URLs)", 48, 15),
            ("bundle", "public_institutions", "Policy Institutes & Global Think Tanks (12 URLs)", 48, 15),
        ]
        for t_type, target, name, interval, max_imgs in default_targets:
            self.register(
                target=target,
                target_type=t_type,
                name=name,
                interval_hours=interval,
                max_images=max_imgs,
                save=False,
            )
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
