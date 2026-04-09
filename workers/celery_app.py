import os
from celery import Celery
from celery.schedules import crontab
from core.config import get_settings

settings = get_settings()

app = Celery(
    "marketingai",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["workers.tasks.agent_tasks", "workers.tasks.content_tasks", "workers.tasks.maintenance_tasks"]
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_publish_retry=True,
)

app.conf.beat_schedule = {
    "execute-due-tasks": {
        "task": "workers.tasks.agent_tasks.execute_due_tasks_batch",
        "schedule": crontab(minute="*/30"),
    },
    "promote-stalled-tasks": {
        "task": "workers.tasks.maintenance_tasks.promote_stalled_tasks",
        "schedule": crontab(minute=0, hour="*/6"),
    },
    "oracle-cloud-keepalive": {
        "task": "workers.tasks.maintenance_tasks.keepalive_ping",
        "schedule": crontab(minute="*/10"),
    },
}
