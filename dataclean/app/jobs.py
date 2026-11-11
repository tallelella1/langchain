"""数据库是任务状态的唯一来源；run_token 对每次执行进行隔离。"""
import logging
import shutil
from datetime import timedelta
from pathlib import Path
from threading import Event, Thread
from uuid import uuid4

from sqlalchemy import select, update

from .config import Settings
from .models import Job, utcnow
from .processing import DataValidationError, process_csv

logger = logging.getLogger(__name__)


def recover_expired(sessions) -> int:
    with sessions.begin() as db:
        result = db.execute(
            update(Job).where(Job.status == "running", Job.lease_expires_at < utcnow()).values(
                status="queued", run_token=None, lease_expires_at=None,
                error="执行进程失联，任务已重新排队", started_at=None,
            )
        )
        return result.rowcount


def queued_ids(sessions) -> list[str]:
    with sessions() as db:
        return list(db.scalars(select(Job.id).where(Job.status == "queued").order_by(Job.created_at).limit(100)))


def claim_job(job_id: str, sessions, settings: Settings) -> tuple[str, str] | None:
    token = str(uuid4())
    now = utcnow()
    with sessions.begin() as db:
        changed = db.execute(
            update(Job).where(Job.id == job_id, Job.status == "queued").values(
                status="running", run_token=token, started_at=now, finished_at=None,
                error=None, summary=None, result_path=None,
                lease_expires_at=now + timedelta(seconds=settings.lease_seconds),
                attempt_count=Job.attempt_count + 1,
            )
        ).rowcount
        if changed != 1:
            return None
        return token, db.get(Job, job_id).input_path


def renew_lease(job_id, token, sessions, settings) -> bool:
    with sessions.begin() as db:
        return db.execute(
            update(Job).where(Job.id == job_id, Job.status == "running", Job.run_token == token).values(
                lease_expires_at=utcnow() + timedelta(seconds=settings.lease_seconds)
            )
        ).rowcount == 1


def finish_job(job_id, token, sessions, *, summary=None, result_path=None, error=None) -> bool:
    with sessions.begin() as db:
        return db.execute(
            update(Job).where(Job.id == job_id, Job.status == "running", Job.run_token == token).values(
                status="failed" if error else "succeeded", summary=summary,
                result_path=result_path, error=error, finished_at=utcnow(),
                lease_expires_at=None, run_token=None,
            )
        ).rowcount == 1


def heartbeat(stop, job_id, token, sessions, settings):
    while not stop.wait(settings.lease_seconds / 3):
        try:
            if not renew_lease(job_id, token, sessions, settings):
                return
        except Exception:
            # No false success: if connectivity does not recover the dispatcher expires this lease.
            logger.exception("Lease renewal failed for job %s", job_id)


def process_job(job_id: str, sessions, settings: Settings) -> bool:
    claimed = claim_job(job_id, sessions, settings)
    if claimed is None:
        return False
    token, input_path = claimed
    root = settings.storage_dir.resolve() / job_id
    stage = root / f".{token}.tmp"
    final = root / token
    stop = Event()
    keeper = Thread(target=heartbeat, args=(stop, job_id, token, sessions, settings), daemon=True)
    keeper.start()
    published = False
    try:
        stage.mkdir(parents=True)
        summary = process_csv(settings.storage_dir / input_path, stage)
        stage.rename(final)
        published = finish_job(job_id, token, sessions, summary=summary, result_path=f"{job_id}/{token}")
        return published
    except Exception as exc:
        message = str(exc) if isinstance(exc, DataValidationError) else "处理失败，请重试；若再次失败请联系管理员"
        logger.exception("Processing failed for job %s", job_id)
        finish_job(job_id, token, sessions, error=message)
        return False
    finally:
        stop.set()
        keeper.join()
        # Only remove this execution's private directories; never another attempt's output.
        if stage.exists():
            shutil.rmtree(stage)
        if not published and final.exists():
            # A commit can succeed while its acknowledgement is lost. Query again
            # before deleting files; an unreferenced directory is safer than lost data.
            try:
                with sessions() as db:
                    saved = db.get(Job, job_id)
                    referenced = saved is not None and saved.result_path == f"{job_id}/{token}"
            except Exception:
                logger.exception("Cannot confirm output reference for job %s; retaining files", job_id)
            else:
                if not referenced:
                    shutil.rmtree(final)


def local_loop(stop, sessions, settings):
    while not stop.is_set():
        try:
            recover_expired(sessions)
            for job_id in queued_ids(sessions):
                if stop.is_set():
                    break
                process_job(job_id, sessions, settings)
        except Exception:
            logger.exception("Local task queue iteration failed")
        stop.wait(settings.poll_seconds)
