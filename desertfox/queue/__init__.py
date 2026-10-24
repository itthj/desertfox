"""DesertFox Distributed Queue Package."""
from desertfox.queue.celery_app import celery_app
from desertfox.queue.tasks import crawl_and_index_task, batch_crawl_targets_task, search_and_index_task

__all__ = ["celery_app", "crawl_and_index_task", "batch_crawl_targets_task", "search_and_index_task"]
