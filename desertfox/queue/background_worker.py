"""
In-Process Autonomous Background Crawler & Indexing Worker for DesertFox.
Decouples slow web/image harvesting from instant (<50ms) FAISS vector lookups:
1. Executes queued crawl, search-harvest, dork-harvest, bulk-crawl, and visual-discover
   tasks in a background daemon thread without requiring external Redis/Celery processes.
2. Provides real-time task status & progress telemetry compatible with `/api/queue/task/{id}`.
3. Periodically sweeps due schedules in SeedScheduleRegistry to pre-warm the face index.
"""

import queue
import threading
import time
import uuid
import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class BackgroundIndexWorker:
    """
    Thread-safe background task runner and periodic seed crawler daemon.
    """

    def __init__(self, engine_getter: Callable[[], Any]):
        self._engine_getter = engine_getter
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Starts the background daemon worker thread if not already running."""
        with self._lock:
            if self._running and self._thread and self._thread.is_alive():
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="DesertFox-BackgroundIndexer",
                daemon=True,
            )
            self._thread.start()

    def enqueue(self, job_type: str, params: Dict[str, Any]) -> str:
        """Enqueues a background indexing job and returns its task_id immediately."""
        self.start()
        task_id = f"bg-{uuid.uuid4().hex[:12]}"
        job = {
            "task_id": task_id,
            "job_type": job_type,
            "params": params,
            "created_at": time.time(),
        }
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "job_type": job_type,
                "status": "PENDING",
                "ready": False,
                "progress": {"step": f"Queued {job_type}"},
                "result": None,
                "error": None,
                "created_at": job["created_at"],
            }
        self._queue.put(job)
        return task_id

    def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Returns current status dictionary for a background task_id, or None if unknown."""
        with self._lock:
            entry = self._tasks.get(task_id)
            return dict(entry) if entry else None

    def list_tasks(self, limit: int = 25) -> Dict[str, Any]:
        """Returns recent background tasks and queue depth."""
        with self._lock:
            recent = sorted(
                self._tasks.values(),
                key=lambda x: x.get("created_at", 0.0),
                reverse=True,
            )[:limit]
        return {
            "worker_alive": bool(self._thread and self._thread.is_alive()),
            "queue_depth": self._queue.qsize(),
            "tasks": recent,
        }

    def _update_task(self, task_id: str, **updates: Any) -> None:
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].update(updates)

    def _worker_loop(self) -> None:
        while self._running:
            try:
                job = self._queue.get(timeout=2.0)
            except queue.Empty:
                continue

            task_id = job["task_id"]
            job_type = job["job_type"]
            params = job["params"]

            self._update_task(
                task_id,
                status="PROGRESS",
                ready=False,
                progress={"step": f"Running {job_type}..."},
            )

            try:
                engine = self._engine_getter()
                if job_type == "crawl":
                    url = params["target_url"]
                    max_pages = int(params.get("max_pages", 10))
                    max_images = int(params.get("max_images", 15))
                    if max_pages > 1:
                        res = engine.audit_site(url, max_pages=max_pages, max_images=max_images)
                    else:
                        res = engine.crawl_and_index_target(url, max_images=max_images)
                elif job_type == "search_harvest":
                    res = engine.harvest_and_index_query(
                        query=params["query"],
                        max_results=int(params.get("max_results", 50)),
                        engines=params.get("engines"),
                    )
                elif job_type == "dork_harvest":
                    res = engine.dork_and_index_query(
                        query=params["query"],
                        categories=params.get("categories"),
                        max_per_dork=int(params.get("max_per_dork", 20)),
                        max_dorks=int(params.get("max_dorks", 12)),
                        engines=params.get("engines"),
                    )
                elif job_type == "bulk_crawl":
                    res = engine.bulk_crawl_urls(
                        urls=params.get("urls", []),
                        max_images_per_url=int(params.get("max_images_per_url", 15)),
                    )
                elif job_type == "visual_discover":
                    res = engine.visual_search_and_index_probe(
                        probe_image=params["probe_image"],
                        query_hint=params.get("query_hint", ""),
                        max_results=int(params.get("max_results", 50)),
                        top_k=int(params.get("top_k", 15)),
                        threshold=float(params.get("threshold", 0.55)),
                    )
                elif job_type == "scheduler_sweep":
                    res = self._run_due_schedules(engine)
                else:
                    raise ValueError(f"Unsupported background job_type: {job_type}")

                self._update_task(
                    task_id,
                    status="SUCCESS",
                    ready=True,
                    progress={"step": "Completed"},
                    result=res,
                )
            except Exception as exc:
                logger.warning("Background task %s (%s) failed: %s", task_id, job_type, exc)
                self._update_task(
                    task_id,
                    status="FAILURE",
                    ready=True,
                    progress={"step": "Failed"},
                    error=str(exc),
                )
            finally:
                self._queue.task_done()

    @staticmethod
    def _run_due_schedules(engine: Any) -> Dict[str, Any]:
        from desertfox.ingestion.seed_manager import SeedHubManager, SeedScheduleRegistry

        reg = SeedScheduleRegistry()
        due = reg.get_due_schedules()
        total_new = 0
        swept = []
        for sched in due:
            sid = sched["id"]
            ttype = sched.get("target_type", "url")
            target = sched.get("target", "")
            max_imgs = int(sched.get("max_images", 15))
            if ttype == "bundle":
                urls = SeedHubManager.get_bundle_urls(target)[:10]
                r = engine.bulk_crawl_urls(urls, max_images_per_url=max_imgs)
                added = int(r.get("total_images_indexed", 0))
            elif ttype == "sitemap":
                urls = SeedHubManager.parse_sitemap_urls(target, max_urls=15)
                r = engine.bulk_crawl_urls(urls, max_images_per_url=max_imgs)
                added = int(r.get("total_images_indexed", 0))
            else:
                r = engine.crawl_and_index_target(target, max_images=max_imgs)
                added = int(r.get("newly_indexed_images", 0))
            reg.record_run(sid, new_faces=added)
            total_new += added
            swept.append({"id": sid, "target": target, "newly_indexed": added})
        return {"status": "success", "schedules_executed": len(swept), "newly_indexed_images": total_new, "swept": swept}
