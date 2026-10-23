"""
Celery Task Definitions for Distributed Crawling and Biometric Ingestion in DesertFox.
"""

import logging
import os
from typing import Any, Dict, List

from celery import Celery

from desertfox.config import EngineConfig
from desertfox.pipeline.engine import DesertFoxEngine
from desertfox.queue.celery_app import celery_app

logger = logging.getLogger(__name__)

# Cached worker engine singleton
_worker_engine = None


def get_worker_engine() -> DesertFoxEngine:
    global _worker_engine
    if _worker_engine is None:
        index_path = os.environ.get("DESERTFOX_INDEX_PATH", "./data/desertfox.index")
        cfg = EngineConfig(vector_store_path=index_path)
        _worker_engine = DesertFoxEngine(config=cfg)
        logger.info("Initialized worker DesertFoxEngine at: %s", index_path)
    return _worker_engine


@celery_app.task(bind=True, name="desertfox.crawl_and_index")
def crawl_and_index_task(self, target_url: str, max_images: int = 25) -> Dict[str, Any]:
    """
    Background worker task to scrape a webpage, extract facial imagery,
    and index discovered identities into the vector store.
    """
    logger.info("Starting distributed harvest task %s for target: %s", self.request.id, target_url)
    self.update_state(state="PROGRESS", meta={"step": "crawling", "target_url": target_url})

    try:
        engine = get_worker_engine()
        result = engine.crawl_and_index_target(target_url=target_url, max_images=max_images, use_async=True)
        logger.info("Completed harvest task %s: %d new faces indexed", self.request.id, result.get("newly_indexed_images", 0))
        return {
            "task_id": self.request.id,
            "status": "success",
            "page_url": target_url,
            "page_osint": result.get("page_osint", {}),
            "candidates_found": result.get("candidates_found", 0),
            "newly_indexed_images": result.get("newly_indexed_images", 0),
        }
    except Exception as exc:
        logger.exception("Error executing harvest task %s: %s", self.request.id, exc)
        return {
            "task_id": self.request.id,
            "status": "error",
            "page_url": target_url,
            "error": str(exc),
        }


@celery_app.task(bind=True, name="desertfox.batch_crawl_targets")
def batch_crawl_targets_task(self, target_urls: List[str], max_images_per_target: int = 15) -> Dict[str, Any]:
    """
    Coordinates batch harvesting across multiple target websites.
    """
    logger.info("Starting batch harvest task %s for %d targets", self.request.id, len(target_urls))
    engine = get_worker_engine()

    total_indexed = 0
    results = []

    for idx, url in enumerate(target_urls):
        self.update_state(
            state="PROGRESS",
            meta={
                "current_target": url,
                "completed_targets": idx,
                "total_targets": len(target_urls),
                "total_indexed": total_indexed,
            },
        )
        try:
            res = engine.crawl_and_index_target(target_url=url, max_images=max_images_per_target, use_async=True)
            new_idx = res.get("newly_indexed_images", 0)
            total_indexed += new_idx
            results.append({
                "url": url,
                "status": "success",
                "indexed": new_idx,
            })
        except Exception as exc:
            results.append({
                "url": url,
                "status": "error",
                "error": str(exc),
            })

    return {
        "task_id": self.request.id,
        "total_targets_processed": len(target_urls),
        "total_images_indexed": total_indexed,
        "details": results,
    }


@celery_app.task(bind=True, name="desertfox.reverse_search")
def reverse_search_task(self, image_base64: str, seed_urls=None, search_query=None, max_pages: int = 30, max_images: int = 10, max_sites: int = 10, top_k: int = 10, threshold: float = 0.60):
    """Run a reverse image search job over discovered candidate pages."""
    from desertfox.api.server import decode_base64_payload

    image, _ = decode_base64_payload(image_base64)
    engine = get_worker_engine()
    return engine.reverse_search_probe(
        probe_image=image,
        seed_urls=list(seed_urls or []),
        query=search_query,
        max_sites=max_sites,
        max_pages=max_pages,
        max_images=max_images,
        top_k=top_k,
        threshold=threshold,
    )
