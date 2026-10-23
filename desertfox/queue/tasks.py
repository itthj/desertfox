"""
Background task execution for DesertFox.
"""

import os
from celery import Celery

from desertfox.pipeline.engine import DesertFoxEngine

celery_app = Celery(
    "desertfox_tasks",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
)


@celery_app.task(bind=True, name="desertfox.crawl_and_index")
def crawl_and_index_task(self, target_url: str, max_images: int = 10):
    """Run a crawl-and-index job in the background."""
    engine = get_worker_engine()
    return engine.crawl_and_index_target(target_url, max_images=max_images, use_async=False, persist=True)


@celery_app.task(bind=True, name="desertfox.reverse_search")
def reverse_search_task(self, image_base64: str, seed_urls=None, search_query=None, max_pages: int = 30, max_images: int = 10, max_sites: int = 10, top_k: int = 10, threshold: float = 0.60):
    """Run a reverse image search job over discovered candidate pages."""
    import base64
    import io
    from PIL import Image

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


def get_worker_engine() -> DesertFoxEngine:
    """Construct a worker-scoped engine instance."""
    return DesertFoxEngine(
        vector_store_path="./data/desertfox.index",
        optout_registry_path="./data/optout_registry.json",
    )
