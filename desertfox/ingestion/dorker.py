"""
OSINT Platform Dorking Engine for DesertFox.
Generates targeted search engine queries across high-probability identity networks,
social platforms, corporate directories, academic rosters, speaker registries,
news/press archives, developer ecosystems, public records, and visual portfolios.
"""

from typing import List, Dict, Any, Optional


class OSINTDorker:
    """Generates specialized search dorks for deep identity exposure discovery."""

    CATEGORY_METADATA: Dict[str, str] = {
        "social": "Social Networks & Identity Profiles",
        "corporate": "Corporate Leadership, Boards & Team Rosters",
        "academic": "Universities, Research Labs & Scholar Portals",
        "speakers": "Conferences, Keynotes & Event Registries",
        "news_media": "Press Releases, Interviews & Media Bylines",
        "developer_tech": "Open-Source, AI Hubs & Developer Profiles",
        "public_records": "Government, Institutional & Non-Profit Rosters",
        "visual_portfolios": "Photography, Creative & Video Hosting Platforms",
    }

    PLATFORM_PATTERNS: Dict[str, List[str]] = {
        "social": [
            'site:linkedin.com/in "{query}"',
            'site:twitter.com "{query}"',
            'site:x.com "{query}"',
            'site:github.com "{query}"',
            'site:instagram.com "{query}"',
            'site:facebook.com "{query}"',
            'site:reddit.com/user "{query}"',
            'site:medium.com "@{query}"',
            'site:substack.com "{query}"',
            'site:bsky.app "{query}"',
            'site:threads.net "{query}"',
            'site:linktr.ee "{query}"',
            'site:about.me "{query}"',
        ],
        "corporate": [
            '"{query}" (inurl:team OR inurl:about OR inurl:people OR inurl:leadership OR inurl:staff)',
            '"{query}" (inurl:faculty OR inurl:board OR inurl:executives OR inurl:our-team)',
            '"{query}" (inurl:management OR inurl:founders OR inurl:advisors OR inurl:partners)',
            'site:crunchbase.com/person "{query}"',
            'site:wellfound.com "{query}"',
            'site:pitchbook.com "{query}"',
            'site:bloomberg.com/profile/person "{query}"',
            'site:theorg.com "{query}"',
        ],
        "academic": [
            'site:.edu "{query}"',
            'site:ac.uk "{query}"',
            '"{query}" (inurl:researcher OR inurl:alumni OR inurl:directory OR inurl:faculty)',
            'site:scholar.google.com "{query}"',
            'site:researchgate.net/profile "{query}"',
            'site:orcid.org "{query}"',
            'site:semanticscholar.org "{query}"',
            'site:ieeexplore.ieee.org/author "{query}"',
            'site:dl.acm.org/profile "{query}"',
        ],
        "speakers": [
            'site:eventbrite.com "{query}"',
            'site:speakerdeck.com "{query}"',
            'site:slideshare.net "{query}"',
            'site:lu.ma "{query}"',
            'site:sessionize.com "{query}"',
            'site:sched.com "{query}"',
            'site:meetup.com "{query}"',
            'site:ted.com/speakers "{query}"',
            '"{query}" (inurl:speakers OR inurl:keynote OR inurl:panelists OR inurl:presenter)',
        ],
        "news_media": [
            '"{query}" (inurl:author OR inurl:contributor OR inurl:journalist OR inurl:columnist)',
            '"{query}" (inurl:interview OR inurl:press-release OR inurl:newsroom OR inurl:spotlight)',
            'site:prnewswire.com "{query}"',
            'site:businesswire.com "{query}"',
            'site:forbes.com/profile "{query}"',
            'site:techcrunch.com "{query}"',
            'site:muckrack.com "{query}"',
        ],
        "developer_tech": [
            'site:huggingface.co "{query}"',
            'site:gitlab.com "{query}"',
            'site:kaggle.com "{query}"',
            'site:stackoverflow.com/users "{query}"',
            'site:dev.to "{query}"',
            'site:hashnode.com "{query}"',
            'site:pypi.org/user "{query}"',
            'site:npmjs.com/~ "{query}"',
        ],
        "public_records": [
            'site:.gov "{query}"',
            'site:.org "{query}" (inurl:staff OR inurl:board OR inurl:trustees OR inurl:fellows)',
            'site:littlesis.org "{query}"',
            'site:propublica.org "{query}"',
            '"{query}" (inurl:committee OR inurl:commission OR inurl:council OR inurl:directory)',
        ],
        "visual_portfolios": [
            'site:commons.wikimedia.org "{query}"',
            'site:flickr.com "{query}"',
            'site:behance.net "{query}"',
            'site:dribbble.com "{query}"',
            'site:vimeo.com "{query}"',
            'site:youtube.com "@{query}"',
            'site:imdb.com/name "{query}"',
        ],
    }

    @classmethod
    def generate_dorks(
        cls,
        query: str,
        categories: Optional[List[str]] = None,
        max_dorks: int = 24
    ) -> List[Dict[str, str]]:
        """
        Generates structured search dorks for a target name or handle.
        Interleaves patterns round-robin across selected categories so every
        requested category gets balanced representation up to max_dorks.
        """
        clean_query = query.strip().replace('"', '')
        if not clean_query:
            return []

        selected_categories = [
            c for c in (categories or list(cls.PLATFORM_PATTERNS.keys()))
            if c in cls.PLATFORM_PATTERNS
        ]
        if not selected_categories:
            selected_categories = list(cls.PLATFORM_PATTERNS.keys())

        dorks: List[Dict[str, str]] = []
        max_len = max((len(cls.PLATFORM_PATTERNS.get(cat, [])) for cat in selected_categories), default=0)

        for idx in range(max_len):
            for cat in selected_categories:
                patterns = cls.PLATFORM_PATTERNS.get(cat, [])
                if idx < len(patterns):
                    dork_query = patterns[idx].format(query=clean_query)
                    dorks.append({
                        "category": cat,
                        "category_label": cls.CATEGORY_METADATA.get(cat, cat),
                        "dork": dork_query,
                    })
                    if len(dorks) >= max_dorks:
                        return dorks

        return dorks

    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        """Returns summary telemetry of all available dork categories and pattern counts."""
        total_patterns = sum(len(v) for v in cls.PLATFORM_PATTERNS.values())
        return {
            "total_categories": len(cls.PLATFORM_PATTERNS),
            "total_patterns": total_patterns,
            "categories": {
                cat: {
                    "label": cls.CATEGORY_METADATA.get(cat, cat),
                    "pattern_count": len(patterns),
                }
                for cat, patterns in cls.PLATFORM_PATTERNS.items()
            },
        }
