"""Persist first, publish later: queued rows act as a durable database outbox.

Re-publishing is intentional. A worker claims a row atomically, so delivery is
at least once while visible results are committed by only the current run_token.
"""
import logging
import signal
from threading import Event

from .config import Settings
from .db import make_engine, make_sessions
from .jobs import queued_ids, recover_expired

logger = logging.getLogger(__name__)


def dispatch_once(sessions, publish):
    recover_expired(sessions)
    for job_id in queued_ids(sessions):
        publish(job_id)


def main():
    from .celery_app import celery_app

    settings = Settings()
    if settings.task_backend != "celery":
        raise SystemExit("dispatcher 需要 DATACLEAN_TASK_BACKEND=celery")
    engine = make_engine(settings)
    sessions = make_sessions(engine)
    stop = Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    def publish(job_id):
        # Database polling is the retry policy; do not block with broker publish retries.
        celery_app.send_task("dataclean.process", args=[job_id], retry=False)

    try:
        while not stop.is_set():
            try:
                dispatch_once(sessions, publish)
            except Exception:
                logger.exception("Dispatch failed; queued jobs remain in the database")
            stop.wait(settings.poll_seconds)
    finally:
        engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
