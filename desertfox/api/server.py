"""
FastAPI application for DesertFox Facial OSINT & Privacy Defense Suite.
Serves REST API and responsive Cyber-OSINT HUD Web Interface.
"""

import os
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

app = FastAPI(
    title="DesertFox OSINT & Exposure Engine",
    description="Ethical facial recognition search engine and automated privacy remediation framework.",
    version="1.0.0"
)

# Initialize Engine singleton
engine = DesertFoxEngine(
    vector_store_path="./data/desertfox.index",
    optout_registry_path="./data/optout_registry.json"
)

# Pydantic Request Models
class SearchRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded probe face image")
    threshold: float = Field(0.60, ge=0.1, le=1.0)
    top_k: int = Field(10, ge=1, le=50)

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
    max_results: int = Field(25, ge=5, le=100)
    engines: Optional[List[str]] = Field(default=["duckduckgo", "bing"], description="Search engines to query")

class DorkHarvestRequest(BaseModel):
    query: str = Field(..., description="Target name or identifier for OSINT dorking")
    categories: Optional[List[str]] = Field(None, description="Platform categories: 'social', 'corporate', 'academic', 'speakers'")
    max_per_dork: int = Field(15, ge=1, le=50)
    max_dorks: int = Field(8, ge=1, le=20)

class BulkCrawlRequest(BaseModel):
    urls: Optional[List[str]] = Field(None, description="Direct list of target URLs")
    raw_text: Optional[str] = Field(None, description="Raw text, CSV, or line-separated URLs to parse")
    sitemap_url: Optional[str] = Field(None, description="XML Sitemap URL")
    bundle_id: Optional[str] = Field(None, description="Curated bundle ID ('tech_execs', 'academic_rosters', 'open_contributors')")
    max_images_per_url: int = Field(15, ge=1, le=50)

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
    return {
        "status": "online",
        "engine_version": "1.0.0",
        "total_indexed_vectors": len(engine.vector_store.records),
        "embedding_dimensions": engine.embedder.embedding_dim,
        "optout_exclusions_count": len(engine.optout_registry.blacklisted_hashes),
        "vector_store_path": engine.vector_store_path,
        "face_enrolled": engine.enrollment.enrolled,
        "enrollment_mode": engine.enrollment.mode,
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
    """Dispatches a background web crawl job to the distributed Celery/Redis task queue."""
    try:
        validate_public_url(req.target_url)
    except UnsafeURLError as e:
        raise _reject_unsafe(e)
    try:
        from desertfox.queue.tasks import crawl_and_index_task
        task = crawl_and_index_task.delay(target_url=req.target_url, max_images=req.max_images)
        return {
            "status": "queued",
            "task_id": task.id,
            "target_url": req.target_url,
            "message": "Harvesting job successfully dispatched to Celery worker queue"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue task: {str(e)}")


@app.get("/api/queue/task/{task_id}")
def get_task_status(task_id: str):
    """Queries status and telemetry for a distributed background harvesting task."""
    try:
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
    """Dispatches a background web search & crawl job to the distributed Celery/Redis task queue."""
    try:
        from desertfox.queue.tasks import search_and_index_task
        task = search_and_index_task.delay(
            query=req.query,
            max_results=req.max_results,
            engines=req.engines
        )
        return {
            "status": "queued",
            "task_id": task.id,
            "query": req.query,
            "message": f"Automated search harvest job for '{req.query}' dispatched to Celery worker queue"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue search harvest task: {str(e)}")


@app.post("/api/search/dork-harvest")
def dork_harvest_query(req: DorkHarvestRequest):
    """
    Executes automated platform dorking across targeted identity networks
    (social networks, corporate directories, academic rosters, speakers)
    and indexes all matching discovered portraits.
    """
    try:
        res = engine.dork_and_index_query(
            query=req.query,
            categories=req.categories,
            max_per_dork=req.max_per_dork,
            max_dorks=req.max_dorks
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Platform dorking harvest failed: {str(e)}")


@app.post("/api/queue/dork-harvest")
def enqueue_dork_harvest_job(req: DorkHarvestRequest):
    """Dispatches a background platform dorking harvest to the Celery worker queue."""
    try:
        from desertfox.queue.tasks import dork_and_index_task
        task = dork_and_index_task.delay(
            query=req.query,
            categories=req.categories,
            max_per_dork=req.max_per_dork,
            max_dorks=req.max_dorks
        )
        return {
            "status": "queued",
            "task_id": task.id,
            "query": req.query,
            "message": f"Platform dorking task for '{req.query}' dispatched to Celery worker queue"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue dork harvest task: {str(e)}")


@app.get("/api/seeds/bundles")
def list_seed_bundles():
    """Lists pre-configured curated seed directories available for wide-spectrum indexing."""
    return {
        "status": "success",
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
    """Dispatches bulk URL crawl to the Celery worker queue."""
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

        from desertfox.queue.tasks import bulk_crawl_urls_task
        task = bulk_crawl_urls_task.delay(
            urls=targets,
            max_images_per_url=req.max_images_per_url
        )
        return {
            "status": "queued",
            "task_id": task.id,
            "targets_count": len(targets),
            "message": f"Bulk crawl task with {len(targets)} targets dispatched to Celery worker queue"
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
    """Generates legally grounded GDPR / DMCA removal requests."""
    if req.notice_type.lower() == "gdpr":
        notice = TakedownNoticeGenerator.generate_gdpr_erasure_request(
            subject_name=req.subject_name,
            subject_email=req.subject_email,
            infringing_url=req.infringing_url,
            additional_notes=req.notes or ""
        )
    elif req.notice_type.lower() == "ccpa":
        notice = TakedownNoticeGenerator.generate_ccpa_deletion_request(
            consumer_name=req.subject_name,
            consumer_email=req.subject_email,
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
