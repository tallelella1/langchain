import logging

from celery import Celery

from .config import Settings
from .db import make_engine, make_sessions
from .jobs import process_job

settings = Settings()
celery_app = Celery("dataclean", broker=settings.broker_url)
celery_app.conf.update(
    task_serializer="json", accept_content=["json"],
    task_ignore_result=True, task_acks_late=True,
    task_reject_on_worker_lost=True, worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
)


@celery_app.task(name="dataclean.process")
def process(job_id: str):
    # Engine created inside the worker, after Celery forks.
    engine = make_engine(settings)
    try:
        return process_job(job_id, make_sessions(engine), settings)
    finally:
        engine.dispose()
