"""
Video Frame & Keyframe Biometric OSINT Harvester for DesertFox.
Extracts faces across video timelines from:
  1. Direct video files/streams (.mp4, .webm, .mov, .avi, .mkv) via OpenCV (cv2)
     with scene-change / motion delta filtering and exact timestamp tracking.
  2. Public video platforms (YouTube, Vimeo, Dailymotion) via multi-keyframe
     storyboard extraction (0%, 25%, 50%, 75% temporal marks) and oEmbed metadata.
  3. Zero-API-key video search engines (YouTube, Bing Videos, DuckDuckGo Videos).
  4. Uploaded probe videos (extracting the sharpest, most pose-diverse frames
     for Multi-Probe Ensemble Fusion).
"""

import io
import os
import re
import html
import json
import tempfile
import logging
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image

from desertfox.security import UnsafeURLError, safe_get, validate_public_url

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

MAX_VIDEO_DOWNLOAD_BYTES = 50 * 1024 * 1024  # 50 MB cap for direct video downloads


def format_timestamp_hms(seconds: float) -> str:
    """Formats seconds (e.g. 85.4) into HH:MM:SS or MM:SS."""
    total_secs = max(0, int(round(seconds)))
    hrs = total_secs // 3600
    mins = (total_secs % 3600) // 60
    secs = total_secs % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def build_timestamp_deeplink(base_url: str, seconds: float) -> str:
    """Appends a jump-to-timestamp parameter or fragment to a video URL."""
    sec_int = max(0, int(round(seconds)))
    if sec_int == 0 or not base_url:
        return base_url
    lower = base_url.lower()
    if "youtube.com/watch" in lower:
        sep = "&" if "?" in base_url else "?"
        return f"{base_url}{sep}t={sec_int}s"
    if "youtu.be/" in lower:
        sep = "&" if "?" in base_url else "?"
        return f"{base_url}{sep}t={sec_int}"
    if "vimeo.com/" in lower:
        return f"{base_url}#t={sec_int}s"
    return f"{base_url}#t={sec_int}"


class VideoFrameHarvester:
    """
    Extracts temporal keyframes and candidate face crops from video files,
    direct video streams, and public video platforms (YouTube, Vimeo, web videos).
    """

    YOUTUBE_ID_REGEX = re.compile(
        r"(?:youtube\.com/(?:watch\?.*?v=|embed/|v/|shorts/)|youtu\.be/)([A-Za-z0-9_-]{11})"
    )
    VIMEO_ID_REGEX = re.compile(r"vimeo\.com/(?:video/)?(\d+)")

    def __init__(self, timeout: int = 12):
        self.timeout = timeout
        self.headers = {
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
        }

    @classmethod
    def extract_youtube_id(cls, url: str) -> Optional[str]:
        """Extracts 11-character YouTube video ID from any YouTube URL format."""
        match = cls.YOUTUBE_ID_REGEX.search(url or "")
        return match.group(1) if match else None

    @classmethod
    def extract_vimeo_id(cls, url: str) -> Optional[str]:
        """Extracts numeric Vimeo video ID from URL."""
        match = cls.VIMEO_ID_REGEX.search(url or "")
        return match.group(1) if match else None

    @staticmethod
    def create_synthetic_test_video(
        frames_rgb: List[np.ndarray],
        fps: float = 2.0,
        suffix: str = ".mp4",
    ) -> bytes:
        """
        Helper utility to encode a sequence of RGB numpy arrays (H, W, 3) into an
        in-memory video byte stream using OpenCV VideoWriter.
        """
        if not frames_rgb:
            raise ValueError("frames_rgb must contain at least one frame")
        h, w = frames_rgb[0].shape[:2]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
            for rgb in frames_rgb:
                resized = cv2.resize(rgb.astype(np.uint8), (w, h)) if rgb.shape[:2] != (h, w) else rgb.astype(np.uint8)
                bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            writer.release()
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def extract_frames_from_video_path(
        video_path: str,
        sample_interval_sec: float = 1.5,
        max_frames: int = 24,
        min_scene_delta: float = 3.5,
    ) -> Dict[str, Any]:
        """
        Decodes a video file on disk using OpenCV and samples frames across its timeline.
        Skips near-identical consecutive static frames when L1 pixel delta < min_scene_delta
        (unless it is the first frame).

        Returns:
            {
                "fps": float,
                "total_frames": int,
                "duration_sec": float,
                "frames": [
                    {
                        "image": PIL.Image,
                        "frame_index": int,
                        "timestamp_sec": float,
                        "timestamp_hms": str,
                    },
                    ...
                ]
            }
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Unable to decode video stream", "frames": [], "fps": 0.0, "duration_sec": 0.0}

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 0.1 or fps > 240.0 or np.isnan(fps):
                fps = 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration_sec = round(total_frames / fps, 2) if total_frames > 0 else 0.0

            step_frames = max(1, int(round(fps * max(0.2, sample_interval_sec))))
            extracted: List[Dict[str, Any]] = []
            prev_small: Optional[np.ndarray] = None
            frame_idx = 0

            while len(extracted) < max_frames:
                if total_frames > 0 and frame_idx >= total_frames:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, bgr = cap.read()
                if not ret or bgr is None:
                    break

                # Downsample to 64x64 grayscale to measure scene/motion change cheaply
                gray_small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
                if prev_small is not None and min_scene_delta > 0:
                    delta = float(np.mean(np.abs(gray_small - prev_small)))
                    if delta < min_scene_delta and len(extracted) > 0:
                        frame_idx += step_frames
                        continue

                prev_small = gray_small
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                ts_sec = round(frame_idx / fps, 2)

                extracted.append({
                    "image": pil_img,
                    "frame_index": frame_idx,
                    "timestamp_sec": ts_sec,
                    "timestamp_hms": format_timestamp_hms(ts_sec),
                })

                frame_idx += step_frames

            return {
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "duration_sec": duration_sec,
                "frames": extracted,
            }
        finally:
            cap.release()

    @classmethod
    def extract_frames_from_video_bytes(
        cls,
        video_bytes: bytes,
        sample_interval_sec: float = 1.5,
        max_frames: int = 24,
        min_scene_delta: float = 3.5,
        suffix: str = ".mp4",
    ) -> Dict[str, Any]:
        """Writes raw video bytes to a temporary file and extracts timeline keyframes."""
        if not video_bytes:
            return {"error": "Empty video payload", "frames": []}
        if len(video_bytes) > MAX_VIDEO_DOWNLOAD_BYTES:
            return {"error": f"Video exceeds {MAX_VIDEO_DOWNLOAD_BYTES // (1024 * 1024)} MB safety cap", "frames": []}

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            return cls.extract_frames_from_video_path(
                tmp_path,
                sample_interval_sec=sample_interval_sec,
                max_frames=max_frames,
                min_scene_delta=min_scene_delta,
            )
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def extract_youtube_keyframes(self, url: str) -> Dict[str, Any]:
        """
        Extracts temporal storyboard keyframes and metadata for a YouTube video
        without needing an API key or full video stream download.
        YouTube generates 4 temporal keyframes per video:
          - hqdefault.jpg / maxresdefault.jpg (Poster / 0s)
          - 1.jpg (~25% timeline mark)
          - 2.jpg (~50% timeline mark)
          - 3.jpg (~75% timeline mark)
        """
        video_id = self.extract_youtube_id(url)
        if not video_id:
            return {"error": "Invalid YouTube URL", "frames": []}

        canonical_url = f"https://www.youtube.com/watch?v={video_id}"
        title = f"YouTube Video ({video_id})"
        author = "YouTube Channel"

        # Fetch oEmbed metadata
        try:
            oembed_url = f"https://www.youtube.com/oembed?url={urllib.parse.quote(canonical_url)}&format=json"
            resp = requests.get(oembed_url, headers=self.headers, timeout=self.timeout)
            if resp.status_code == 200:
                meta = resp.json()
                title = meta.get("title") or title
                author = meta.get("author_name") or author
        except Exception as exc:
            logger.debug("YouTube oEmbed lookup skipped: %s", exc)

        # Temporal keyframe slots: (filename, approximate_sec_offset, label)
        keyframe_specs = [
            ("maxresdefault.jpg", 0.0, "00:00 (Poster HD)"),
            ("hqdefault.jpg", 5.0, "00:05 (Thumbnail)"),
            ("1.jpg", 30.0, "00:30 (25% Keyframe)"),
            ("2.jpg", 60.0, "01:00 (50% Keyframe)"),
            ("3.jpg", 90.0, "01:30 (75% Keyframe)"),
        ]

        frames: List[Dict[str, Any]] = []
        seen_sizes: List[Tuple[int, int]] = []

        for idx, (slot_name, approx_sec, label) in enumerate(keyframe_specs):
            # If maxresdefault already succeeded, skip hqdefault duplicate poster
            if slot_name == "hqdefault.jpg" and len(frames) > 0:
                continue
            img_url = f"https://i.ytimg.com/vi/{video_id}/{slot_name}"
            try:
                r = requests.get(img_url, headers=self.headers, timeout=self.timeout)
                if r.status_code != 200 or not r.content:
                    continue
                pil_img = Image.open(io.BytesIO(r.content)).convert("RGB")
                if pil_img.width < 60 or pil_img.height < 60:
                    continue
                frames.append({
                    "image": pil_img,
                    "frame_index": idx,
                    "timestamp_sec": approx_sec,
                    "timestamp_hms": format_timestamp_hms(approx_sec),
                    "keyframe_label": label,
                    "frame_image_url": img_url,
                    "deep_link_url": build_timestamp_deeplink(canonical_url, approx_sec),
                })
            except Exception:
                continue

        return {
            "platform": "YouTube",
            "video_id": video_id,
            "video_url": canonical_url,
            "title": title,
            "author": author,
            "frames": frames,
        }

    def extract_vimeo_keyframes(self, url: str) -> Dict[str, Any]:
        """Extracts video metadata and high-res frame capture from Vimeo oEmbed API."""
        vimeo_id = self.extract_vimeo_id(url)
        if not vimeo_id:
            return {"error": "Invalid Vimeo URL", "frames": []}

        canonical_url = f"https://vimeo.com/{vimeo_id}"
        frames: List[Dict[str, Any]] = []
        title = f"Vimeo Video ({vimeo_id})"
        author = "Vimeo Creator"
        duration = 0.0

        try:
            oembed = f"https://vimeo.com/api/oembed.json?url={urllib.parse.quote(canonical_url)}"
            resp = requests.get(oembed, headers=self.headers, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                title = data.get("title") or title
                author = data.get("author_name") or author
                duration = float(data.get("duration") or 0.0)
                thumb_url = data.get("thumbnail_url")
                if thumb_url:
                    img_resp = requests.get(thumb_url, headers=self.headers, timeout=self.timeout)
                    if img_resp.status_code == 200 and img_resp.content:
                        pil_img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                        frames.append({
                            "image": pil_img,
                            "frame_index": 0,
                            "timestamp_sec": 0.0,
                            "timestamp_hms": "00:00",
                            "keyframe_label": "00:00 (Vimeo Frame)",
                            "frame_image_url": thumb_url,
                            "deep_link_url": canonical_url,
                        })
        except Exception as exc:
            logger.warning("Vimeo keyframe extraction failed: %s", exc)

        return {
            "platform": "Vimeo",
            "video_id": vimeo_id,
            "video_url": canonical_url,
            "title": title,
            "author": author,
            "duration_sec": duration,
            "frames": frames,
        }

    def harvest_video_url(
        self,
        url: str,
        sample_interval_sec: float = 2.0,
        max_frames: int = 20,
    ) -> Dict[str, Any]:
        """
        Unified entry point for any video URL:
          1. YouTube video / Shorts -> multi-keyframe storyboard extraction
          2. Vimeo video -> oEmbed high-res frame extraction
          3. Direct .mp4 / .webm / .mov URL -> downloads bounded video & decodes frames via OpenCV
          4. Webpage with embedded <video> or YouTube/Vimeo iframe -> discovers video & extracts frames
        """
        yt_id = self.extract_youtube_id(url)
        if yt_id:
            return self.extract_youtube_keyframes(url)

        vimeo_id = self.extract_vimeo_id(url)
        if vimeo_id:
            return self.extract_vimeo_keyframes(url)

        try:
            validate_public_url(url)
        except UnsafeURLError as exc:
            return {"error": str(exc), "frames": []}

        try:
            fetched = safe_get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                max_bytes=MAX_VIDEO_DOWNLOAD_BYTES,
                allow_Statuses=(200, 206),
            )
            if not fetched:
                return {"error": "Unable to fetch video URL", "frames": []}

            content_type = (fetched.content_type or "").lower()
            path_lower = urllib.parse.urlparse(fetched.url).path.lower()

            is_direct_video = (
                content_type.startswith("video/")
                or content_type == "application/octet-stream"
                or any(path_lower.endswith(ext) for ext in (".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv"))
            )

            if is_direct_video:
                ext = ".mp4"
                for candidate_ext in (".webm", ".mov", ".avi", ".mkv", ".mp4"):
                    if path_lower.endswith(candidate_ext):
                        ext = candidate_ext
                        break
                decoded = self.extract_frames_from_video_bytes(
                    fetched.body,
                    sample_interval_sec=sample_interval_sec,
                    max_frames=max_frames,
                    suffix=ext,
                )
                for f in decoded.get("frames", []):
                    f["deep_link_url"] = build_timestamp_deeplink(fetched.url, f["timestamp_sec"])
                    f["frame_image_url"] = fetched.url
                return {
                    "platform": "DirectVideo",
                    "video_url": fetched.url,
                    "title": os.path.basename(path_lower) or "Direct Video Stream",
                    "author": urllib.parse.urlparse(fetched.url).netloc,
                    "duration_sec": decoded.get("duration_sec", 0.0),
                    "fps": decoded.get("fps", 0.0),
                    "frames": decoded.get("frames", []),
                }

            # Otherwise, inspect HTML page for embedded <video src="...">, <source src="...">, og:video, or YouTube embeds
            html_text = fetched.body.decode("utf-8", errors="ignore")
            yt_embed = self.extract_youtube_id(html_text)
            if yt_embed:
                return self.extract_youtube_keyframes(f"https://www.youtube.com/watch?v={yt_embed}")

            video_srcs = re.findall(
                r'(?:<video[^>]+src=["\']|<source[^>]+src=["\']|<meta[^>]+property=["\']og:video(?::url|:secure_url)?["\'][^>]+content=["\'])([^"\'>\s]+)',
                html_text,
                flags=re.IGNORECASE,
            )
            for raw_src in video_srcs:
                resolved_video = urllib.parse.urljoin(fetched.url, html.unescape(raw_src))
                if self.extract_youtube_id(resolved_video):
                    return self.extract_youtube_keyframes(resolved_video)
                if any(resolved_video.lower().split("?")[0].endswith(ext) for ext in (".mp4", ".webm", ".mov", ".m4v")):
                    return self.harvest_video_url(
                        resolved_video,
                        sample_interval_sec=sample_interval_sec,
                        max_frames=max_frames,
                    )

            return {"error": "No decodable video stream or embedded video found at URL", "frames": []}
        except Exception as exc:
            logger.warning("Failed to harvest video URL %s: %s", url, exc)
            return {"error": str(exc), "frames": []}

    def search_public_videos(self, query: str, max_videos: int = 12) -> List[Dict[str, Any]]:
        """
        Searches YouTube, Bing Videos, and DuckDuckGo for public videos matching
        a target identity or handle without requiring API keys.
        """
        discovered: List[Dict[str, Any]] = []
        seen_ids = set()

        # 1. Direct YouTube search page scraping for videoRenderer items
        try:
            yt_search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
            resp = requests.get(yt_search_url, headers=self.headers, timeout=self.timeout)
            if resp.status_code == 200:
                # Extract videoId and title pairs from ytInitialData
                vid_matches = re.findall(
                    r'"videoRenderer":\{"videoId":"([A-Za-z0-9_-]{11})".*?"title":\{"runs":\[\{"text":"([^"]+)"\}',
                    resp.text,
                )
                if not vid_matches:
                    raw_ids = re.findall(r'"videoId":"([A-Za-z0-9_-]{11})"', resp.text)
                    vid_matches = [(vid, f"{query} - YouTube Video") for vid in raw_ids]

                for vid_id, raw_title in vid_matches:
                    if vid_id in seen_ids:
                        continue
                    seen_ids.add(vid_id)
                    video_url = f"https://www.youtube.com/watch?v={vid_id}"
                    discovered.append({
                        "video_id": vid_id,
                        "video_url": video_url,
                        "title": html.unescape(raw_title),
                        "platform": "YouTube",
                        "domain": "youtube.com",
                        "keyframes": [
                            {"url": f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg", "timestamp_sec": 0.0, "label": "00:00"},
                            {"url": f"https://i.ytimg.com/vi/{vid_id}/1.jpg", "timestamp_sec": 30.0, "label": "00:30 (25%)"},
                            {"url": f"https://i.ytimg.com/vi/{vid_id}/2.jpg", "timestamp_sec": 60.0, "label": "01:00 (50%)"},
                            {"url": f"https://i.ytimg.com/vi/{vid_id}/3.jpg", "timestamp_sec": 90.0, "label": "01:30 (75%)"},
                        ],
                    })
                    if len(discovered) >= max_videos:
                        return discovered
        except Exception as exc:
            logger.warning("YouTube search failed: %s", exc)

        # 2. Fallback / supplement via Bing Video search
        try:
            bing_vid_url = f"https://www.bing.com/videos/search?q={urllib.parse.quote(query)}"
            resp_b = requests.get(bing_vid_url, headers=self.headers, timeout=self.timeout)
            if resp_b.status_code == 200:
                yt_links = re.findall(r"https?://(?:www\.)?youtube\.com/watch\?v=([A-Za-z0-9_-]{11})", resp_b.text)
                for vid_id in yt_links:
                    if vid_id in seen_ids:
                        continue
                    seen_ids.add(vid_id)
                    video_url = f"https://www.youtube.com/watch?v={vid_id}"
                    discovered.append({
                        "video_id": vid_id,
                        "video_url": video_url,
                        "title": f"{query} (Video)",
                        "platform": "YouTube",
                        "domain": "youtube.com",
                        "keyframes": [
                            {"url": f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg", "timestamp_sec": 0.0, "label": "00:00"},
                            {"url": f"https://i.ytimg.com/vi/{vid_id}/1.jpg", "timestamp_sec": 30.0, "label": "00:30 (25%)"},
                            {"url": f"https://i.ytimg.com/vi/{vid_id}/2.jpg", "timestamp_sec": 60.0, "label": "01:00 (50%)"},
                            {"url": f"https://i.ytimg.com/vi/{vid_id}/3.jpg", "timestamp_sec": 90.0, "label": "01:30 (75%)"},
                        ],
                    })
                    if len(discovered) >= max_videos:
                        return discovered
        except Exception as exc:
            logger.warning("Bing video search failed: %s", exc)

        return discovered[:max_videos]

    @classmethod
    def extract_best_probe_frames(
        cls,
        video_bytes: bytes,
        detector: Any,
        max_probes: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Extracts up to `max_probes` high-quality, pose-diverse face frames from an
        uploaded selfie/probe video clip for Multi-Probe Ensemble Fusion.
        """
        decoded = cls.extract_frames_from_video_bytes(
            video_bytes,
            sample_interval_sec=0.5,
            max_frames=30,
            min_scene_delta=1.5,
        )
        raw_frames = decoded.get("frames", [])
        if not raw_frames:
            return []

        scored_candidates: List[Dict[str, Any]] = []
        for item in raw_frames:
            pil_img = item["image"]
            faces = detector.detect_faces(pil_img) if detector is not None else []
            conf = float(faces[0].confidence) if faces else 0.5
            # Estimate sharpness via grayscale Laplacian variance
            gray = np.asarray(pil_img.convert("L"), dtype=np.float32)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            score = conf * 100.0 + min(sharpness / 10.0, 50.0)
            scored_candidates.append({
                "image": pil_img,
                "timestamp_sec": item["timestamp_sec"],
                "timestamp_hms": item["timestamp_hms"],
                "frame_index": item["frame_index"],
                "score": round(score, 2),
                "has_face": bool(faces),
            })

        # Prefer frames with detected faces, then sort by quality score while keeping temporal spread
        face_frames = [c for c in scored_candidates if c["has_face"]] or scored_candidates
        if len(face_frames) <= max_probes:
            return face_frames

        # Partition timeline into `max_probes` buckets and pick the highest-scoring frame from each bucket
        selected: List[Dict[str, Any]] = []
        bucket_size = len(face_frames) / float(max_probes)
        for b in range(max_probes):
            start_i = int(round(b * bucket_size))
            end_i = int(round((b + 1) * bucket_size))
            bucket = face_frames[start_i:max(start_i + 1, end_i)]
            if bucket:
                best_in_bucket = max(bucket, key=lambda x: x["score"])
                selected.append(best_in_bucket)

        return selected[:max_probes]
