"""
Celery Task Definitions for Distributed Crawling and Biometric Ingestion in DesertFox.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from desertfox.queue.celery_app import celery_app
from desertfox.pipeline.engine import DesertFoxEngine
from desertfox.config import EngineConfig
from desertfox.ingestion.seed_manager import SeedScheduleRegistry, SeedHubManager

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
            "newly_indexed_images": result.get("newly_indexed_images", 0)
        }
    except Exception as e:
        logger.exception("Error executing harvest task %s: %s", self.request.id, e)
        return {
            "task_id": self.request.id,
            "status": "error",
            "page_url": target_url,
            "error": str(e)
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
                "total_indexed": total_indexed
            }
        )
        try:
            res = engine.crawl_and_index_target(target_url=url, max_images=max_images_per_target, use_async=True)
            new_idx = res.get("newly_indexed_images", 0)
            total_indexed += new_idx
            results.append({
                "url": url,
                "status": "success",
                "indexed": new_idx
            })
        except Exception as e:
            results.append({
                "url": url,
                "status": "error",
                "error": str(e)
            })

    return {
        "task_id": self.request.id,
        "total_targets_processed": len(target_urls),
        "total_images_indexed": total_indexed,
        "details": results
    }


@celery_app.task(bind=True, name="desertfox.search_and_index")
def search_and_index_task(
    self,
    query: str,
    max_results: int = 35,
    engines: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Background worker task to aggregate public search engine results for an identity/name,
    harvest discovered portraits, and index faces into the vector store.
    """
    logger.info("Starting distributed search & harvest task %s for query: %s", self.request.id, query)
    self.update_state(state="PROGRESS", meta={"step": "searching", "query": query})

    try:
        engine = get_worker_engine()
        result = engine.harvest_and_index_query(query=query, max_results=max_results, engines=engines)
        logger.info("Completed search task %s: %d new faces indexed for '%s'", self.request.id, result.get("newly_indexed_images", 0), query)
        return {
            "task_id": self.request.id,
            "status": "success",
            "query": query,
            "candidates_found": result.get("candidates_found", 0),
            "newly_indexed_images": result.get("newly_indexed_images", 0),
            "items": result.get("items", [])
        }
    except Exception as e:
        logger.exception("Error executing search task %s: %s", self.request.id, e)
        return {
            "task_id": self.request.id,
            "status": "error",
            "query": query,
            "error": str(e)
        }


@celery_app.task(bind=True, name="desertfox.dork_and_index")
def dork_and_index_task(
    self,
    query: str,
    categories: Optional[List[str]] = None,
    max_per_dork: int = 15,
    max_dorks: int = 8
) -> Dict[str, Any]:
    """
    Background worker task to run multi-platform OSINT dorking for an identity,
    crawl matching candidate image results, and index faces.
    """
    logger.info("Starting distributed dorking task %s for query: %s", self.request.id, query)
    self.update_state(state="PROGRESS", meta={"step": "dorking", "query": query})
    try:
        engine = get_worker_engine()
        result = engine.dork_and_index_query(
            query=query,
            categories=categories,
            max_per_dork=max_per_dork,
            max_dorks=max_dorks
        )
        return {
            "task_id": self.request.id,
            "status": "success",
            "query": query,
            "dorks_executed": result.get("dorks_executed", 0),
            "total_candidates_found": result.get("total_candidates_found", 0),
            "newly_indexed_images": result.get("newly_indexed_images", 0),
            "dork_reports": result.get("dork_reports", []),
            "items": result.get("items", [])
        }
    except Exception as e:
        logger.exception("Error executing dork task %s: %s", self.request.id, e)
        return {
            "task_id": self.request.id,
            "status": "error",
            "query": query,
            "error": str(e)
        }


@celery_app.task(bind=True, name="desertfox.bulk_crawl_urls")
def bulk_crawl_urls_task(
    self,
    urls: List[str],
    max_images_per_url: int = 15
) -> Dict[str, Any]:
    """
    Background worker task to batch crawl hundreds of target URLs or sitemaps.
    """
    logger.info("Starting bulk crawl task %s for %d URLs", self.request.id, len(urls))
    self.update_state(state="PROGRESS", meta={"step": "bulk_crawling", "total": len(urls)})
    try:
        engine = get_worker_engine()
        result = engine.bulk_crawl_urls(urls=urls, max_images_per_url=max_images_per_url)
        return {
            "task_id": self.request.id,
            "status": "success",
            "targets_processed": result.get("targets_processed", 0),
            "total_images_indexed": result.get("total_images_indexed", 0),
            "details": result.get("details", [])
        }
    except Exception as e:
        logger.exception("Error executing bulk crawl task %s: %s", self.request.id, e)
        return {
            "task_id": self.request.id,
            "status": "error",
            "error": str(e)
        }


@celery_app.task(bind=True, name="desertfox.run_scheduled_crawl_sweep")
def run_scheduled_crawl_sweep(self) -> Dict[str, Any]:
    """
    Automated continuous crawl sweep executed by Celery Beat scheduler.
    Inspects persistent schedules, extracts due targets, crawls images,
    extracts faces, and updates index.
    """
    logger.info("Executing scheduled continuous crawl sweep [task %s]", self.request.id)
    self.update_state(state="PROGRESS", meta={"step": "evaluating_due_schedules"})

    registry = SeedScheduleRegistry()
    due_schedules = registry.get_due_schedules()
    logger.info("Found %d schedules due for indexing", len(due_schedules))

    if not due_schedules:
        return {
            "task_id": self.request.id,
            "status": "idle",
            "message": "No continuous schedules currently due for crawling.",
            "total_schedules_checked": len(registry.schedules)
        }

    engine = get_worker_engine()
    sweep_results = []
    total_new_faces = 0

    for item in due_schedules:
        sched_id = item["id"]
        t_type = item.get("target_type", "url")
        target = item["target"]
        max_imgs = item.get("max_images", 15)

        targets_to_crawl = []
        if t_type == "bundle":
            targets_to_crawl = SeedHubManager.get_bundle_urls(target)
        elif t_type == "sitemap":
            targets_to_crawl = SeedHubManager.parse_sitemap_urls(target, max_urls=30)
        else:
            targets_to_crawl = [target]

        try:
            res = engine.bulk_crawl_urls(urls=targets_to_crawl, max_images_per_url=max_imgs)
            indexed_count = res.get("total_images_indexed", 0)
            total_new_faces += indexed_count
            registry.record_run(sched_id, new_faces=indexed_count)
            sweep_results.append({
                "schedule_id": sched_id,
                "name": item.get("name", target),
                "target_type": t_type,
                "targets_crawled": len(targets_to_crawl),
                "faces_indexed": indexed_count,
                "status": "success"
            })
        except Exception as e:
            logger.warning("Failed executing scheduled sweep for %s: %s", sched_id, e)
            sweep_results.append({
                "schedule_id": sched_id,
                "name": item.get("name", target),
                "status": "error",
                "error": str(e)
            })

    return {
        "task_id": self.request.id,
        "status": "complete",
        "schedules_processed": len(due_schedules),
        "total_new_faces_indexed": total_new_faces,
        "details": sweep_results
    }


@celery_app.task(bind=True, name="desertfox.optimize_and_compact_index")
def optimize_and_compact_index(self) -> Dict[str, Any]:
    """
    Periodic index compaction and partition re-building executed by Celery Beat.
    Re-clusters Voronoi cells, syncs FAISS indices, and saves persistent snapshots.
    """
    logger.info("Executing periodic index optimization & compaction [task %s]", self.request.id)
    try:
        engine = get_worker_engine()
        engine.partitioned_index.build_partitions()
        engine.save_index()
        total_vectors = len(engine.vector_store.records)
        return {
            "task_id": self.request.id,
            "status": "success",
            "total_vectors_indexed": total_vectors,
            "faiss_enabled": engine.config.use_faiss,
            "message": "Index partitions optimized and snapshot persisted."
        }
    except Exception as e:
        logger.exception("Error during index optimization: %s", e)
        return {
            "task_id": self.request.id,
            "status": "error",
            "error": str(e)
        }


@celery_app.task(bind=True, name="desertfox.video_harvest_and_index")
def video_harvest_and_index_task(
    self,
    query: Optional[str] = None,
    video_url: Optional[str] = None,
    max_videos: int = 8,
    sample_interval_sec: float = 2.0,
) -> Dict[str, Any]:
    """
    Background worker task to extract keyframes from a video URL or public video search query,
    detect faces across timestamps, and index them into FAISS.
    """
    logger.info("Starting video harvest task %s (query=%s, url=%s)", self.request.id, query, video_url)
    self.update_state(state="PROGRESS", meta={"step": "video_harvesting", "query": query, "video_url": video_url})
    try:
        engine = get_worker_engine()
        if video_url:
            res = engine.crawl_and_index_video_url(video_url, sample_interval_sec=sample_interval_sec)
        elif query:
            res = engine.harvest_and_index_video_query(query, max_videos=max_videos)
        else:
            return {"task_id": self.request.id, "status": "error", "error": "Neither query nor video_url provided"}
        res["task_id"] = self.request.id
        return res
    except Exception as e:
        logger.exception("Error in video harvest task %s: %s", self.request.id, e)
        return {"task_id": self.request.id, "status": "error", "error": str(e)}
