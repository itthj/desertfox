"""
OSINT Platform Dorking Engine for DesertFox.
Generates targeted search engine queries across high-probability identity networks,
social platforms, corporate directories, academic rosters, and speaker registries.
"""

from typing import List, Dict, Set, Optional
import urllib.parse


class OSINTDorker:
    """Generates specialized search dorks for deep identity exposure discovery."""

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
        ],
        "corporate": [
            '"{query}" (inurl:team OR inurl:about OR inurl:people OR inurl:leadership OR inurl:staff)',
            '"{query}" (inurl:faculty OR inurl:board OR inurl:executives OR inurl:our-team)',
        ],
        "academic": [
            'site:.edu "{query}"',
            'site:ac.uk "{query}"',
            '"{query}" (inurl:researcher OR inurl:alumni OR inurl:directory)',
        ],
        "speakers": [
            'site:eventbrite.com "{query}"',
            'site:speakerdeck.com "{query}"',
            'site:slideshare.net "{query}"',
            'site:luma.com "{query}"',
        ]
    }

    @classmethod
    def generate_dorks(
        cls,
        query: str,
        categories: Optional[List[str]] = None,
        max_dorks: int = 12
    ) -> List[Dict[str, str]]:
        """
        Generates structured search dorks for a target name or handle.
        Returns a list of dicts with category and full query string.
        """
        clean_query = query.strip().replace('"', '')
        selected_categories = categories or list(cls.PLATFORM_PATTERNS.keys())

        dorks = []
        for cat in selected_categories:
            patterns = cls.PLATFORM_PATTERNS.get(cat, [])
            for pattern in patterns:
                dork_query = pattern.format(query=clean_query)
                dorks.append({
                    "category": cat,
                    "dork": dork_query
                })
                if len(dorks) >= max_dorks:
                    return dorks
        return dorks
