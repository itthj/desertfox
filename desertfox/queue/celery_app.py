"""
Celery Application and Distributed Task Queue configuration for DesertFox.
Coordinates asynchronous crawling, biometric indexing, and batch harvesting jobs.
"""

import os
from celery import Celery

# Redis broker and backend defaults
REDIS_HOST = os.environ.get("DESERTFOX_REDIS_HOST", "localhost")
REDIS_PORT = os.environ.get("DESERTFOX_REDIS_PORT", "6379")
REDIS_DB = os.environ.get("DESERTFOX_REDIS_DB", "0")

BROKER_URL = os.environ.get("CELERY_BROKER_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")

celery_app = Celery(
    "desertfox",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=["desertfox.queue.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=1800,  # 30-minute hard cap per harvesting job
    task_soft_time_limit=1500,
    worker_prefetch_multiplier=1,
    worker_concurrency=int(os.environ.get("CELERY_WORKER_CONCURRENCY", "4")),
)
