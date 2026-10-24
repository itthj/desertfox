"""
FastAPI application for DesertFox Facial OSINT & Privacy Defense Suite.
Serves REST API and responsive Cyber-OSINT HUD Web Interface.
"""

import os
import base64
from typing import Dict, Any, List, Optional

from desertfox.privacy.enrollment import sha256_bytes
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from desertfox.security import (
    PayloadTooLargeError,
    UnsafeURLError,
    decode_base64_image,
    decode_base64_payload as _decode_bounded,
    validate_public_url,
)
from desertfox.pipeline.engine import DesertFoxEngine
from desertfox.privacy.takedown import TakedownNoticeGenerator
from desertfox.ingestion.seed_manager import SeedHubManager, SeedScheduleRegistry
from desertfox.ingestion.dorker import OSINTDorker
from desertfox.queue.background_worker import BackgroundIndexWorker

app = FastAPI(
    title="DesertFox OSINT & Exposure Engine",
    description="Ethical facial recognition search engine and automated privacy remediation framework.",
    version="1.0.0"
)

# Initialize Engine singleton & In-Process Autonomous Background Worker
engine = DesertFoxEngine(
    vector_store_path="./data/desertfox.index",
    optout_registry_path="./data/optout_registry.json"
)
bg_worker = BackgroundIndexWorker(lambda: engine)
bg_worker.start()

# Pydantic Request Models
class SearchRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded probe face image")
    threshold: float = Field(0.60, ge=0.1, le=1.0)
    top_k: int = Field(10, ge=1, le=50)

class MultiProbeSearchRequest(BaseModel):
    images_base64: List[str] = Field(..., min_length=1, max_length=5, description="1 to 5 base64 encoded probe face images")
    threshold: float = Field(0.55, ge=0.1, le=1.0)
    top_k: int = Field(15, ge=1, le=50)

class IndexImageRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image to index")
    record_id: str
    source_url: str
    metadata: Optional[Dict[str, Any]] = None

class CrawlRequest(BaseModel):
    target_url: str = Field(..., description="URL to scrape and ingest")
    max_images: int = Field(10, ge=1, le=50)

class AuditRequest(BaseModel):
    seed_url: str = Field(..., min_length=3, max_length=500, description="One site to walk. Other hosts are not followed.")
    max_pages: int = Field(30, ge=1, le=100)
    max_images: int = Field(10, ge=1, le=25)

class SearchHarvestRequest(BaseModel):
    query: str = Field(..., description="Target name, username, or identity to search across public web engines")
    max_results: int = Field(60, ge=5, le=250)
    engines: Optional[List[str]] = Field(
        default=["duckduckgo", "bing", "google", "brave", "yandex", "wikimedia"],
        description="Search engines to query"
    )

class DorkHarvestRequest(BaseModel):
    query: str = Field(..., description="Target name or identifier for OSINT dorking")
    categories: Optional[List[str]] = Field(
        None,
        description="Platform categories: 'social', 'corporate', 'academic', 'speakers', 'news_media', 'developer_tech', 'public_records', 'visual_portfolios'"
    )
    max_per_dork: int = Field(20, ge=1, le=100)
    max_dorks: int = Field(16, ge=1, le=60)
    engines: Optional[List[str]] = Field(None, description="Optional search engines override for dork execution")

class BulkCrawlRequest(BaseModel):
    urls: Optional[List[str]] = Field(None, description="Direct list of target URLs")
    raw_text: Optional[str] = Field(None, description="Raw text, CSV, or line-separated URLs to parse")
    sitemap_url: Optional[str] = Field(None, description="XML Sitemap URL")
    bundle_id: Optional[str] = Field(None, description="Curated bundle ID or 'all'")
    max_images_per_url: int = Field(15, ge=1, le=100)

class ScheduleRegisterRequest(BaseModel):
    target: str = Field(..., description="Target URL, XML sitemap, or bundle ID")
    target_type: str = Field("url", description="'url', 'sitemap', or 'bundle'")
    name: Optional[str] = Field(None, description="Descriptive name")
    interval_hours: int = Field(12, ge=1, le=168, description="Crawl frequency in hours")
    max_images: int = Field(15, ge=1, le=50)

class OptOutRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image to exclude permanently")

class TakedownRequest(BaseModel):
    notice_type: str = Field("gdpr", description="'gdpr' or 'dmca'")
    infringing_url: str
    subject_name: str
    subject_email: str
    notes: Optional[str] = ""

class VideoProbeSearchRequest(BaseModel):
    video_base64: str = Field(..., description="Base64 encoded probe video clip (MP4/WebM/MOV)")
    threshold: float = Field(0.55, ge=0.1, le=1.0)
    top_k: int = Field(15, ge=1, le=50)
    max_probe_frames: int = Field(4, ge=1, le=6)

class VideoUrlIndexRequest(BaseModel):
    video_url: str = Field(..., description="YouTube, Vimeo, or direct .mp4/.webm video URL")
    sample_interval_sec: float = Field(2.0, ge=0.5, le=30.0)
    max_frames: int = Field(20, ge=1, le=60)

class VideoUploadIndexRequest(BaseModel):
    video_base64: str = Field(..., description="Base64 encoded video file to extract frames and index")
    title: str = Field("Uploaded Video Clip", description="Descriptive title for the video")
    source_url: str = Field("uploaded://video.mp4", description="Source reference URL")
    sample_interval_sec: float = Field(1.5, ge=0.5, le=30.0)
    max_frames: int = Field(20, ge=1, le=60)

class VideoSearchHarvestRequest(BaseModel):
    query: str = Field(..., description="Target name or handle to search across public video platforms")
    max_videos: int = Field(8, ge=1, le=25)


class VisualDiscoverRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded probe face image for reverse visual web discovery")
    query_hint: Optional[str] = Field("", description="Optional target name or handle hint to enrich visual discovery")
    max_results: int = Field(45, ge=5, le=150)
    threshold: float = Field(0.50, ge=0.1, le=1.0)
    top_k: int = Field(25, ge=1, le=50)


def decode_base64_video_bytes(b64_str: str, max_bytes: int = 40 * 1024 * 1024) -> bytes:
    """Decodes a base64 or data-URI video payload with strict size bounds."""
    if not b64_str:
        raise ValueError("Empty video payload")
    payload = b64_str.split(",", 1)[1] if "," in b64_str else b64_str
    raw = base64.b64decode(payload)
    if len(raw) > max_bytes:
        raise PayloadTooLargeError(f"Video payload exceeds {max_bytes // (1024 * 1024)} MB limit")
    return raw

def decode_base64_payload(b64_str: str):
    """Return the PIL image and the SHA-256 of the uploaded bytes (size- and pixel-bounded)."""
    image, raw = _decode_bounded(b64_str)
    return image, sha256_bytes(raw)


def _reject_unsafe(exc: Exception) -> HTTPException:
    """Maps hardening errors to precise HTTP statuses."""
    if isinstance(exc, PayloadTooLargeError):
        return HTTPException(status_code=413, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


@app.get("/api/stats")
def get_stats():
    """Returns real-time engine telemetry."""
    lock_enabled = bool(getattr(engine.config, "enforce_single_face_lock", False))
    return {
        "status": "online",
        "engine_version": "1.0.0",
        "total_indexed_vectors": len(engine.vector_store.records),
        "embedding_dimensions": engine.embedder.embedding_dim,
        "optout_exclusions_count": len(engine.optout_registry.blacklisted_hashes),
        "vector_store_path": engine.vector_store_path,
        "single_face_lock_enabled": lock_enabled,
        "face_enrolled": engine.enrollment.enrolled if lock_enabled else False,
        "enrollment_mode": engine.enrollment.mode if lock_enabled else "multi-face-production",
    }


@app.post("/api/enrollment/reset")
def reset_enrollment_lock():
    """Clears any stored single-face enrollment file and confirms lock state."""
    engine.enrollment.reset()
    return {
        "status": "reset",
        "single_face_lock_enabled": bool(getattr(engine.config, "enforce_single_face_lock", False)),
        "message": "Enrollment state cleared. Multi-face production mode is active.",
    }


@app.post("/api/search")
def search_probe(req: SearchRequest):
    """Audits a probe photo against indexed faces for the one enrolled person."""
    try:
        image, digest = decode_base64_payload(req.image_base64)
        gate = engine.authorize_probe(image, digest)
        if not gate["allowed"]:
            raise HTTPException(status_code=403, detail={
                "status": "rejected",
                "message": gate.get("message") or "This photo does not match the enrolled face.",
                "similarity": gate.get("similarity"),
                "mode": gate.get("mode"),
            })
        results = engine.search_probe_image(
            probe_image=image,
            top_k=req.top_k,
            threshold=req.threshold
        )
        results["enrollment"] = {
            "created": gate["created"],
            "similarity": gate["similarity"],
            "mode": gate["mode"],
        }
        return results
    except HTTPException:
        raise
    except (PayloadTooLargeError, UnsafeURLError) as e:
        raise _reject_unsafe(e)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Search failed: {str(e)}")


@app.post("/api/search/visual-discover")
def search_visual_discover(req: VisualDiscoverRequest):
    """
    Reverse-Image Visual Web Discovery + Auto-Indexing + Biometric Search:
    Uploads the probe face photo to Yandex CBIR Visual Search (`rpt=imageview` + `&cbir_page=similar`),
    Bing Visual Search (`sbiupload`), and optional entity-chained multi-engine search,
    downloads candidate thumbnails concurrently, detects up to 4 faces per image,
    indexes them into FAISS, and returns all matching face exposures.
    """
    try:
        image, digest = decode_base64_payload(req.image_base64)
        gate = engine.authorize_probe(image, digest)
        if not gate["allowed"]:
            raise HTTPException(status_code=403, detail={
                "status": "rejected",
                "message": gate.get("message") or "This photo does not match the enrolled face.",
                "similarity": gate.get("similarity"),
                "mode": gate.get("mode"),
            })
        results = engine.visual_search_and_index_probe(
            probe_image=image,
            query_hint=(req.query_hint or "").strip(),
            max_results=req.max_results,
            top_k=req.top_k,
            threshold=req.threshold,
        )
        results["enrollment"] = {
            "created": gate["created"],
            "similarity": gate["similarity"],
            "mode": gate["mode"],
        }
        return results
    except HTTPException:
        raise
    except (PayloadTooLargeError, UnsafeURLError) as e:
        raise _reject_unsafe(e)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Visual discovery failed: {str(e)}")


@app.post("/api/queue/visual-discover")
def enqueue_visual_discover_job(req: VisualDiscoverRequest):
    """Dispatches a reverse-image visual discovery & indexing job to the in-process background daemon."""
    try:
        image, digest = decode_base64_payload(req.image_base64)
        gate = engine.authorize_probe(image, digest)
        if not gate["allowed"]:
            raise HTTPException(status_code=403, detail="Rejected by enrollment lock")
        task_id = bg_worker.enqueue(
            "visual_discover",
            {
                "probe_image": image,
                "query_hint": (req.query_hint or "").strip(),
                "max_results": req.max_results,
                "threshold": req.threshold,
                "top_k": req.top_k,
            },
        )
        return {
            "status": "queued",
            "task_id": task_id,
            "message": "Visual discovery job dispatched to background indexer daemon",
        }
    except HTTPException:
        raise
    except (PayloadTooLargeError, UnsafeURLError) as e:
        raise _reject_unsafe(e)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to queue visual discovery: {str(e)}")


@app.post("/api/search/multi-probe")
def search_multi_probe(req: MultiProbeSearchRequest):
    """
    Multi-photo biometric probe search:
    Aggregates 1 to 5 probe images of the enrolled person across different angles/lighting,
    performs 5-point landmark similarity transform alignment, estimates head poses,
    fuses embeddings into a composite prototype vector, and searches FAISS.
    """
    try:
        decoded_images = []
        for idx, b64 in enumerate(req.images_base64):
            img, digest = decode_base64_payload(b64)
            # Authorize first image to lock identity or verify matching
            gate = engine.authorize_probe(img, digest)
            if not gate["allowed"]:
                raise HTTPException(status_code=403, detail={
                    "status": "rejected",
                    "message": f"Probe image {idx+1} does not match enrolled identity.",
                    "similarity": gate.get("similarity"),
                    "mode": gate.get("mode")
                })
            decoded_images.append(img)

        results = engine.search_multi_probe_images(
            probe_images=decoded_images,
            top_k=req.top_k,
            threshold=req.threshold
        )
        return results
    except HTTPException:
        raise
    except (PayloadTooLargeError, UnsafeURLError) as e:
        raise _reject_unsafe(e)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Multi-probe search failed: {str(e)}")


@app.post("/api/search/video-probe")
def search_video_probe(req: VideoProbeSearchRequest):
    """
    Video Probe Biometric Search:
    Accepts an uploaded video clip of the user (e.g. a selfie video), automatically
    extracts the sharpest pose-diverse face frames across the video timeline,
    verifies identity against the enrollment lock, fuses embeddings via MultiProbeFusion,
    and searches the FAISS vector database.
    """
    try:
        video_bytes = decode_base64_video_bytes(req.video_base64)
        from desertfox.ingestion.video_harvester import VideoFrameHarvester
        best_frames = VideoFrameHarvester.extract_best_probe_frames(
            video_bytes=video_bytes,
            detector=engine.detector,
            max_probes=req.max_probe_frames,
        )
        if not best_frames:
            raise HTTPException(status_code=400, detail="Could not extract any valid frames from uploaded video.")

        # Authorize the primary extracted face frame against the enrollment lock
        primary_img = best_frames[0]["image"]
        digest = sha256_bytes(video_bytes)
        gate = engine.authorize_probe(primary_img, digest)
        if not gate["allowed"]:
            raise HTTPException(status_code=403, detail={
                "status": "rejected",
                "message": gate.get("message") or "Video probe face does not match enrolled identity.",
                "similarity": gate.get("similarity"),
                "mode": gate.get("mode"),
            })

        results = engine.search_multi_probe_images(
            probe_images=[f["image"] for f in best_frames],
            top_k=req.top_k,
            threshold=req.threshold,
        )
        results["video_probe_frames"] = [
            {
                "frame_index": f["frame_index"],
                "timestamp_sec": f["timestamp_sec"],
                "timestamp_hms": f["timestamp_hms"],
                "quality_score": f["score"],
                "has_face": f["has_face"],
            }
            for f in best_frames
        ]
        return results
    except HTTPException:
        raise
    except (PayloadTooLargeError, UnsafeURLError) as e:
        raise _reject_unsafe(e)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Video probe search failed: {str(e)}")


@app.post("/api/video/index-url")
def index_video_from_url(req: VideoUrlIndexRequest):
    """
    Extracts keyframes across a YouTube, Vimeo, or direct .mp4/.webm video URL,
    detects all faces per frame with SCRFD, extracts 512-d ArcFace embeddings,
    and indexes each appearance with exact timestamp deep links.
    """
    try:
        res = engine.crawl_and_index_video_url(
            video_url=req.video_url,
            sample_interval_sec=req.sample_interval_sec,
            max_frames=req.max_frames,
        )
        if res.get("status") == "error":
            raise HTTPException(status_code=400, detail=res.get("message", "Video URL indexing failed"))
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video URL indexing failed: {str(e)}")


@app.post("/api/video/index-upload")
def index_uploaded_video(req: VideoUploadIndexRequest):
    """
    Decodes an uploaded video file (MP4/WebM/MOV), extracts timeline frames,
    detects all faces across timestamps, and indexes them into FAISS.
    """
    try:
        video_bytes = decode_base64_video_bytes(req.video_base64)
        res = engine.index_video_bytes(
            video_bytes=video_bytes,
            source_url=req.source_url,
            title=req.title,
            sample_interval_sec=req.sample_interval_sec,
            max_frames=req.max_frames,
        )
        if res.get("status") == "error":
            raise HTTPException(status_code=400, detail=res.get("message", "Video upload indexing failed"))
        return res
    except HTTPException:
        raise
    except (PayloadTooLargeError, UnsafeURLError) as e:
        raise _reject_unsafe(e)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Video upload indexing failed: {str(e)}")


@app.post("/api/video/search-harvest")
def harvest_video_search_query(req: VideoSearchHarvestRequest):
    """
    Searches public video platforms (YouTube, Bing Videos) for a target identity/name,
    extracts temporal keyframes across discovered videos, and indexes all detected faces
    with jump-to-timestamp deep links.
    """
    try:
        return engine.harvest_and_index_video_query(
            query=req.query,
            max_videos=req.max_videos,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video search harvest failed: {str(e)}")


@app.post("/api/queue/video-harvest")
def enqueue_video_harvest_job(req: VideoSearchHarvestRequest):
    """Dispatches a background video search & keyframe harvest job to Celery."""
    try:
        from desertfox.queue.tasks import video_harvest_and_index_task
        task = video_harvest_and_index_task.delay(
            query=req.query,
            max_videos=req.max_videos,
        )
        return {
            "status": "queued",
            "task_id": task.id,
            "query": req.query,
            "message": f"Video keyframe harvest job for '{req.query}' dispatched to Celery worker queue",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue video harvest task: {str(e)}")


@app.post("/api/index/image")
def index_single_image(req: IndexImageRequest):
    """Indexes a single image into the database."""
    try:
        image = decode_base64_image(req.image_base64)
        res = engine.process_and_index_image(
            image=image,
            record_id=req.record_id,
            source_url=req.source_url,
            metadata=req.metadata
        )
        if res.get("success"):
            engine.save_index()
            return {"status": "indexed", "record_id": req.record_id, "fiqa_score": res.get("fiqa_score")}
        else:
            return {"status": "skipped", "reason": res.get("reason", "Excluded or low quality")}
    except (PayloadTooLargeError, UnsafeURLError) as e:
        raise _reject_unsafe(e)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Indexing failed: {str(e)}")


@app.post("/api/index/audit")
def audit_site(req: AuditRequest):
    """Walks one named host via its sitemap and on-site links, then indexes images."""
    try:
        return engine.audit_site(req.seed_url, max_pages=req.max_pages, max_images=req.max_images)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Site audit failed: {str(e)}")


@app.post("/api/index/crawl")
def crawl_website(req: CrawlRequest):
    """Crawls a website with advanced harvester, extracting OSINT metadata and public images."""
    try:
        validate_public_url(req.target_url)
    except UnsafeURLError as e:
        raise _reject_unsafe(e)
    try:
        crawl_res = engine.crawl_and_index_target(req.target_url, max_images=req.max_images)
        return crawl_res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Crawl failed: {str(e)}")


@app.post("/api/queue/crawl")
def enqueue_crawl_job(req: CrawlRequest):
    """Dispatches a background web crawl job to the autonomous background worker daemon."""
    try:
        validate_public_url(req.target_url)
    except UnsafeURLError as e:
        raise _reject_unsafe(e)
    try:
        task_id = bg_worker.enqueue("crawl", {"target_url": req.target_url, "max_images": req.max_images})
        return {
            "status": "queued",
            "task_id": task_id,
            "target_url": req.target_url,
            "message": "Harvesting job dispatched to autonomous background worker"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue task: {str(e)}")


@app.get("/api/queue/tasks")
def list_background_tasks():
    """Returns recent background indexing jobs and their real-time statuses."""
    return {
        "status": "success",
        "tasks": bg_worker.list_tasks(limit=25),
    }


@app.get("/api/queue/task/{task_id}")
def get_task_status(task_id: str):
    """Queries status and telemetry for an in-process or Celery background harvesting task."""
    try:
        bg_status = bg_worker.get_status(task_id)
        if bg_status is not None:
            return bg_status

        from desertfox.queue.celery_app import celery_app
        task = celery_app.AsyncResult(task_id)
        response = {
            "task_id": task_id,
            "status": task.status,
            "ready": task.ready()
        }
        if task.successful():
            response["result"] = task.result
        elif task.failed():
            response["error"] = str(task.result)
        elif task.status == "PROGRESS":
            response["progress"] = task.info
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to inspect task: {str(e)}")


@app.post("/api/search/harvest")
def harvest_search_query(req: SearchHarvestRequest):
    """
    Automated Web Search Harvester:
    Queries public search engines for an identity/name, scrapes candidate photos from the results,
    detects faces with SCRFD, extracts ArcFace embeddings, and indexes them into the vector database.
    """
    try:
        res = engine.harvest_and_index_query(
            query=req.query,
            max_results=req.max_results,
            engines=req.engines
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search harvest failed: {str(e)}")


@app.post("/api/queue/search-harvest")
def enqueue_search_harvest_job(req: SearchHarvestRequest):
    """Dispatches a background web search & crawl job to the autonomous background worker."""
    try:
        task_id = bg_worker.enqueue(
            "search_harvest",
            {
                "query": req.query,
                "max_results": req.max_results,
                "engines": req.engines,
            },
        )
        return {
            "status": "queued",
            "task_id": task_id,
            "query": req.query,
            "message": f"Automated search harvest job for '{req.query}' dispatched to background worker"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue search harvest task: {str(e)}")


@app.get("/api/search/providers")
def list_search_providers():
    """Returns supported search providers, OSINT dorking categories, and seed bundle counts."""
    from desertfox.ingestion.search_aggregator import SearchAggregator
    return {
        "status": "success",
        "providers": SearchAggregator.SUPPORTED_ENGINES,
        "dork_summary": OSINTDorker.get_summary(),
        "seed_catalog": SeedHubManager.get_catalog_summary(),
    }


@app.post("/api/search/dork-harvest")
def dork_harvest_query(req: DorkHarvestRequest):
    """
    Executes automated platform dorking across targeted identity networks
    (social networks, corporate directories, academic rosters, speakers, press, tech, records, portfolios)
    and indexes all matching discovered portraits.
    """
    try:
        res = engine.dork_and_index_query(
            query=req.query,
            categories=req.categories,
            max_per_dork=req.max_per_dork,
            max_dorks=req.max_dorks,
            engines=req.engines
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Platform dorking harvest failed: {str(e)}")


@app.post("/api/queue/dork-harvest")
def enqueue_dork_harvest_job(req: DorkHarvestRequest):
    """Dispatches a background platform dorking harvest to the autonomous background worker."""
    try:
        task_id = bg_worker.enqueue(
            "dork_harvest",
            {
                "query": req.query,
                "categories": req.categories,
                "max_per_dork": req.max_per_dork,
                "max_dorks": req.max_dorks,
                "engines": req.engines,
            },
        )
        return {
            "status": "queued",
            "task_id": task_id,
            "query": req.query,
            "message": f"Platform dorking task for '{req.query}' dispatched to background worker"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue dork harvest task: {str(e)}")


@app.get("/api/seeds/bundles")
def list_seed_bundles():
    """Lists pre-configured curated seed directories available for wide-spectrum indexing."""
    catalog = SeedHubManager.get_catalog_summary()
    return {
        "status": "success",
        "total_bundles": catalog["total_bundles"],
        "total_curated_urls": catalog["total_curated_urls"],
        "bundles": SeedHubManager.CURATED_BUNDLES
    }


@app.post("/api/seeds/bulk-crawl")
def bulk_crawl_seeds(req: BulkCrawlRequest):
    """
    Ingests and crawls bulk targets from URL lists, raw text/CSV, XML sitemaps, or curated bundles.
    """
    try:
        targets: List[str] = []
        if req.bundle_id:
            targets.extend(SeedHubManager.get_bundle_urls(req.bundle_id))
        if req.urls:
            targets.extend(req.urls)
        if req.raw_text:
            targets.extend(SeedHubManager.extract_urls_from_text(req.raw_text))
        if req.sitemap_url:
            targets.extend(SeedHubManager.parse_sitemap_urls(req.sitemap_url))

        targets = list(dict.fromkeys(t.strip() for t in targets if t.strip()))
        if not targets:
            raise HTTPException(status_code=400, detail="No valid target URLs provided or resolved.")

        res = engine.bulk_crawl_urls(
            urls=targets,
            max_images_per_url=req.max_images_per_url
        )
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk crawl execution failed: {str(e)}")


@app.post("/api/queue/bulk-crawl")
def enqueue_bulk_crawl_job(req: BulkCrawlRequest):
    """Dispatches bulk URL crawl to the autonomous background worker."""
    try:
        targets: List[str] = []
        if req.bundle_id:
            targets.extend(SeedHubManager.get_bundle_urls(req.bundle_id))
        if req.urls:
            targets.extend(req.urls)
        if req.raw_text:
            targets.extend(SeedHubManager.extract_urls_from_text(req.raw_text))
        if req.sitemap_url:
            targets.extend(SeedHubManager.parse_sitemap_urls(req.sitemap_url))

        targets = list(dict.fromkeys(t.strip() for t in targets if t.strip()))
        if not targets:
            raise HTTPException(status_code=400, detail="No valid target URLs provided or resolved.")

        task_id = bg_worker.enqueue(
            "bulk_crawl",
            {
                "urls": targets,
                "max_images_per_url": req.max_images_per_url,
            },
        )
        return {
            "status": "queued",
            "task_id": task_id,
            "targets_count": len(targets),
            "message": f"Bulk crawl task with {len(targets)} targets dispatched to background worker"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue bulk crawl task: {str(e)}")


@app.get("/api/scheduler/schedules")
def list_schedules():
    """Lists all registered recurring continuous crawler targets and schedules."""
    registry = SeedScheduleRegistry()
    return {
        "status": "success",
        "total_schedules": len(registry.schedules),
        "schedules": registry.get_all(),
        "due_count": len(registry.get_due_schedules())
    }


@app.post("/api/scheduler/register")
def register_schedule(req: ScheduleRegisterRequest):
    """Registers a new continuous crawl target into the background scheduler."""
    registry = SeedScheduleRegistry()
    entry = registry.register(
        target=req.target,
        target_type=req.target_type,
        name=req.name or "",
        interval_hours=req.interval_hours,
        max_images=req.max_images
    )
    return {
        "status": "registered",
        "schedule": entry
    }


@app.post("/api/scheduler/toggle/{schedule_id}")
def toggle_schedule(schedule_id: str):
    """Toggles a schedule between active and paused states."""
    registry = SeedScheduleRegistry()
    updated = registry.toggle(schedule_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Schedule ID not found")
    return {"status": "updated", "schedule": updated}


@app.delete("/api/scheduler/schedules/{schedule_id}")
def delete_schedule(schedule_id: str):
    """Deletes a schedule from the continuous crawler."""
    registry = SeedScheduleRegistry()
    deleted = registry.delete(schedule_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Schedule ID not found")
    return {"status": "deleted", "schedule_id": schedule_id}


@app.post("/api/scheduler/trigger-now")
def trigger_scheduler_sweep():
    """Manually triggers an immediate continuous crawler sweep pass."""
    try:
        from desertfox.queue.tasks import run_scheduled_crawl_sweep
        task = run_scheduled_crawl_sweep.delay()
        return {
            "status": "queued",
            "task_id": task.id,
            "message": "Immediate continuous crawler sweep pass dispatched to Celery"
        }
    except Exception as e:
        try:
            from desertfox.queue.tasks import run_scheduled_crawl_sweep
            res = run_scheduled_crawl_sweep.apply()
            return {
                "status": "executed",
                "result": res.result
            }
        except Exception as inner_e:
            raise HTTPException(status_code=500, detail=f"Failed to trigger scheduler sweep: {str(e)} ({str(inner_e)})")


@app.post("/api/opt-out")
def register_optout(req: OptOutRequest):
    """Registers a face into the Zero-Knowledge Opt-Out table."""
    try:
        image = decode_base64_image(req.image_base64)
        embedding = engine.embedder.extract_from_pil(image)
        exclusion_hash = engine.optout_registry.register_exclusion(embedding)
        return {
            "status": "excluded",
            "message": "Identity registered in Zero-Knowledge Opt-Out Registry. Future searches will suppress this likeness.",
            "exclusion_hash": exclusion_hash
        }
    except (PayloadTooLargeError, UnsafeURLError) as e:
        raise _reject_unsafe(e)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Opt-out registration failed: {str(e)}")


@app.post("/api/takedown")
def generate_takedown(req: TakedownRequest):
    """Generates legally grounded GDPR / CCPA / DMCA / NCII removal requests."""
    ntype = req.notice_type.lower()
    if ntype == "gdpr":
        notice = TakedownNoticeGenerator.generate_gdpr_erasure_request(
            subject_name=req.subject_name,
            subject_email=req.subject_email,
            infringing_url=req.infringing_url,
            additional_notes=req.notes or ""
        )
    elif ntype == "ccpa":
        notice = TakedownNoticeGenerator.generate_ccpa_deletion_request(
            consumer_name=req.subject_name,
            consumer_email=req.subject_email,
            infringing_url=req.infringing_url,
            additional_notes=req.notes or ""
        )
    elif ntype == "ncii":
        notice = TakedownNoticeGenerator.generate_ncii_emergency_takedown(
            subject_name=req.subject_name,
            subject_email=req.subject_email,
            infringing_url=req.infringing_url,
            additional_notes=req.notes or ""
        )
    else:
        notice = TakedownNoticeGenerator.generate_dmca_notice(
            copyright_owner=req.subject_name,
            contact_email=req.subject_email,
            infringing_url=req.infringing_url,
            original_work_description=req.notes or "Original photographic portrait owned by claimant"
        )
    return {"notice": notice, "type": req.notice_type.upper()}


@app.get("/", response_class=HTMLResponse)
def index_dashboard():
    """Serves the Cyber-OSINT HUD dashboard."""
    html_file = os.path.join(os.path.dirname(__file__), "../templates/index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>DesertFox Engine Active. (Template loading...)</h1>"
