"""
Forensic OSINT and Metadata Mining Engine for DesertFox.
Extracts:
1. EXIF metadata (Camera make/model, original timestamps, software)
2. Geolocation coordinates (GPS Lat/Lon/Alt with direct map links)
3. OpenGraph / Twitter Cards / Schema.org page context
4. Host domain DNS and network infrastructure signatures
"""

import socket
from urllib.parse import urlparse
from typing import Dict, Any, Optional
from PIL import Image, ExifTags
from bs4 import BeautifulSoup

class ForensicMetadataExtractor:
    """Extracts forensic metadata from image bytes and webpage context."""

    @staticmethod
    def extract_exif(image: Image.Image) -> Dict[str, Any]:
        """
        Parses EXIF tags from image, resolving GPS coordinates to decimal degrees.
        """
        exif_data: Dict[str, Any] = {}
        try:
            raw_exif = image.getexif()
            if not raw_exif:
                return {"has_exif": False}

            # Map tag IDs to human-readable tag names
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if isinstance(value, (bytes, bytearray)):
                    continue  # Skip unformatted binary blobs
                exif_data[tag_name] = str(value)

            # Check for GPS IFD
            gps_info = raw_exif.get_ifd(ExifTags.IFD.GPSInfo) if hasattr(raw_exif, "get_ifd") else None
            gps_result = {}
            if gps_info:
                for key, val in gps_info.items():
                    sub_tag = ExifTags.GPSTAGS.get(key, str(key))
                    gps_result[sub_tag] = val

                lat, lon = ForensicMetadataExtractor._parse_gps_coords(gps_result)
                if lat is not None and lon is not None:
                    exif_data["gps_decimal"] = {"latitude": lat, "longitude": lon}
                    exif_data["map_link"] = f"https://www.google.com/maps?q={lat},{lon}"

            return {
                "has_exif": True,
                "camera_make": exif_data.get("Make", "Unknown"),
                "camera_model": exif_data.get("Model", "Unknown"),
                "software": exif_data.get("Software", "Unknown"),
                "datetime_original": exif_data.get("DateTimeOriginal", exif_data.get("DateTime", "Unknown")),
                "gps": exif_data.get("gps_decimal", None),
                "map_link": exif_data.get("map_link", None),
                "raw_tags": exif_data
            }
        except Exception as e:
            return {"has_exif": False, "error": str(e)}

    @staticmethod
    def _parse_gps_coords(gps_dict: Dict[str, Any]):
        try:
            def to_deg(dms):
                return float(dms[0]) + float(dms[1]) / 60.0 + float(dms[2]) / 3600.0

            lat = to_deg(gps_dict["GPSLatitude"])
            if gps_dict.get("GPSLatitudeRef") == "S":
                lat = -lat

            lon = to_deg(gps_dict["GPSLongitude"])
            if gps_dict.get("GPSLongitudeRef") == "W":
                lon = -lon

            return round(lat, 6), round(lon, 6)
        except Exception:
            return None, None

    @staticmethod
    def extract_page_osint_context(html_content: str, page_url: str) -> Dict[str, Any]:
        """
        Parses OpenGraph, Twitter Cards, author tags, and page headings.
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            parsed_url = urlparse(page_url)

            # Resolve domain IP
            domain = parsed_url.netloc
            resolved_ip = "Unknown"
            try:
                resolved_ip = socket.gethostbyname(domain.split(":")[0])
            except Exception:
                pass

            title = soup.title.string.strip() if soup.title and soup.title.string else ""

            # Extract OpenGraph meta tags
            og_tags = {}
            for meta in soup.find_all("meta"):
                prop = meta.get("property") or meta.get("name")
                content = meta.get("content")
                if prop and content:
                    if prop.startswith("og:") or prop.startswith("twitter:"):
                        og_tags[prop] = content

            # Extract article/author tags (prioritize direct author name over site name)
            author = ""
            author_tag = soup.find("meta", attrs={"name": "author"}) or soup.find("meta", attrs={"property": "article:author"})
            if author_tag:
                author = author_tag.get("content", "")

            if not author:
                author = og_tags.get("twitter:creator", og_tags.get("og:site_name", ""))

            # Extract Schema.org JSON-LD Person/Article entities
            schema_entities = []
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    import json
                    raw_ld = json.loads(script.string or "")
                    items = raw_ld if isinstance(raw_ld, list) else [raw_ld]
                    for item in items:
                        t = item.get("@type")
                        if t in ("Person", "ProfilePage", "NewsArticle", "Author"):
                            schema_entities.append({
                                "type": t,
                                "name": item.get("name"),
                                "job_title": item.get("jobTitle"),
                                "description": item.get("description")
                            })
                except Exception:
                    pass

            return {
                "domain": domain,
                "host_ip": resolved_ip,
                "page_title": title,
                "author_or_entity": author,
                "og_metadata": og_tags,
                "schema_entities": schema_entities
            }
        except Exception as e:
            return {"error": str(e), "page_url": page_url}
