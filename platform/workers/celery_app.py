import os
from celery import Celery

celery_app = Celery(
    "pwm_platform",
    broker=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)
