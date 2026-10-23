"""
FastAPI application for DesertFox Facial OSINT & Privacy Defense Suite.
Serves REST API and responsive Cyber-OSINT HUD Web Interface.
"""

import os
import io
import base64
from typing import Dict, Any, List, Optional

from desertfox.privacy.enrollment import sha256_bytes
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

from desertfox.pipeline.engine import DesertFoxEngine
from desertfox.privacy.takedown import TakedownNoticeGenerator

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

class OptOutRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image to exclude permanently")

class TakedownRequest(BaseModel):
    notice_type: str = Field("gdpr", description="'gdpr' or 'dmca'")
    infringing_url: str
    subject_name: str
    subject_email: str
    notes: Optional[str] = ""

def decode_base64_image(b64_str: str) -> Image.Image:
    """Helper to decode base64 strings (with or without data URI header) into PIL Image."""
    image, _digest = decode_base64_payload(b64_str)
    return image


def decode_base64_payload(b64_str: str):
    """Return the PIL image and the SHA-256 of the uploaded bytes."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)), sha256_bytes(img_bytes)


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
        "face_enrolled": False,
        "enrollment_mode": "disabled",
    }


@app.post("/api/search")
def search_probe(req: SearchRequest):
    """Audits a probe photo against indexed faces without an enrollment gate."""
    try:
        image, _digest = decode_base64_payload(req.image_base64)
        results = engine.search_probe_image(
            probe_image=image,
            top_k=req.top_k,
            threshold=req.threshold
        )
        results["enrollment"] = {
            "created": False,
            "similarity": 1.0,
            "mode": "disabled",
        }
        return results
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
        crawl_res = engine.crawl_and_index_target(req.target_url, max_images=req.max_images)
        return crawl_res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Crawl failed: {str(e)}")


@app.post("/api/queue/crawl")
def enqueue_crawl_job(req: CrawlRequest):
    """Dispatches a background web crawl job to the distributed Celery/Redis task queue."""
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
