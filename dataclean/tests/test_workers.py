from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from sqlalchemy import update

from app import jobs
from app.dispatcher import dispatch_once
from app.models import Job, utcnow
from conftest import submit


def test_failure_no_partial_results_and_retry(client, app, auth):
    headers = auth()
    job_id = submit(client, headers, b"wrong,header\nx,y\n").json()["id"]
    assert not jobs.process_job(job_id, app.state.sessions, app.state.settings)
    job = client.get(f"/api/jobs/{job_id}", headers=headers).json()
    assert job["status"] == "failed" and "列名" in job["error"]
    with app.state.sessions() as db:
        saved = db.get(Job, job_id)
        assert saved.result_path is None and saved.summary is None
        assert saved.finished_at is not None
    assert list((app.state.settings.storage_dir / job_id).iterdir()) == [app.state.settings.storage_dir / job_id / "input.csv"]
    assert client.get(f"/api/jobs/{job_id}/download/report", headers=headers).status_code == 409
    assert client.post(f"/api/jobs/{job_id}/retry", headers=headers).status_code == 202
    assert client.post(f"/api/jobs/{job_id}/retry", headers=headers).status_code == 409
    assert not jobs.process_job(job_id, app.state.sessions, app.state.settings)
    assert client.get(f"/api/jobs/{job_id}", headers=headers).json()["attempt_count"] == 2


def test_duplicate_workers_only_one_claim(client, app, auth):
    job_id = submit(client, auth()).json()["id"]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: jobs.process_job(job_id, app.state.sessions, app.state.settings), range(4)))
    assert results.count(True) == 1
    with app.state.sessions() as db:
        job = db.get(Job, job_id)
        assert job.status == "succeeded" and job.attempt_count == 1


def test_expired_lease_requeues_and_fences_old_worker(client, app, auth):
    job_id = submit(client, auth()).json()["id"]
    old_token, _ = jobs.claim_job(job_id, app.state.sessions, app.state.settings)
    assert jobs.recover_expired(app.state.sessions) == 0
    with app.state.sessions.begin() as db:
        db.execute(update(Job).where(Job.id == job_id).values(lease_expires_at=utcnow() - timedelta(seconds=1)))
    assert jobs.recover_expired(app.state.sessions) == 1
    assert not jobs.renew_lease(job_id, old_token, app.state.sessions, app.state.settings)
    assert not jobs.finish_job(job_id, old_token, app.state.sessions, summary={"bad": True}, result_path="bad")
    assert jobs.process_job(job_id, app.state.sessions, app.state.settings)
    assert not jobs.finish_job(job_id, old_token, app.state.sessions, error="stale failure")
    with app.state.sessions() as db:
        job = db.get(Job, job_id)
        assert job.attempt_count == 2 and job.status == "succeeded"
        assert "bad" not in job.summary


def test_disk_write_failure_does_not_expose_partial_outputs(client, app, auth, monkeypatch):
    job_id = submit(client, auth()).json()["id"]

    def fail(source, destination):
        (destination / "cleaned.csv").write_text("partial", encoding="utf-8")
        raise OSError("simulated disk failure with private path")

    monkeypatch.setattr(jobs, "process_csv", fail)
    assert not jobs.process_job(job_id, app.state.sessions, app.state.settings)
    with app.state.sessions() as db:
        job = db.get(Job, job_id)
        assert job.status == "failed" and job.result_path is None
        assert "private path" not in job.error
    assert len(list((app.state.settings.storage_dir / job_id).iterdir())) == 1


def test_broker_outage_does_not_lose_committed_job(client, app, auth):
    job_id = submit(client, auth()).json()["id"]
    def unavailable(_):
        raise ConnectionError("redis unavailable")
    with pytest.raises(ConnectionError):
        dispatch_once(app.state.sessions, unavailable)
    with app.state.sessions() as db:
        assert db.get(Job, job_id).status == "queued"
    published = []
    dispatch_once(app.state.sessions, published.append)
    dispatch_once(app.state.sessions, published.append)
    assert published == [job_id, job_id]
    assert jobs.process_job(job_id, app.state.sessions, app.state.settings)
    assert not jobs.process_job(job_id, app.state.sessions, app.state.settings)


def test_celery_task_executes_same_pipeline(client, app, auth, monkeypatch):
    monkeypatch.setenv("DATACLEAN_JWT_SECRET", app.state.settings.jwt_secret)
    monkeypatch.setenv("DATACLEAN_DATABASE_URL", app.state.settings.database_url)
    monkeypatch.setenv("DATACLEAN_STORAGE_DIR", str(app.state.settings.storage_dir))
    from app import celery_app as module
    monkeypatch.setattr(module, "settings", app.state.settings)
    job_id = submit(client, auth()).json()["id"]
    # Local task invocation validates Celery serialization/handler only, not Redis or Linux workers.
    result = module.process.apply(args=[job_id], throw=True)
    assert result.get() is True


@pytest.mark.parametrize("commit_succeeded", [False, True])
def test_uncertain_completion_commit_preserves_referenced_files(client, app, auth, monkeypatch, commit_succeeded):
    job_id = submit(client, auth()).json()["id"]
    original = jobs.finish_job

    def uncertain(*args, **kwargs):
        if kwargs.get("summary") is not None:
            if commit_succeeded:
                original(*args, **kwargs)
            raise ConnectionError("simulated lost database acknowledgement")
        return original(*args, **kwargs)

    monkeypatch.setattr(jobs, "finish_job", uncertain)
    jobs.process_job(job_id, app.state.sessions, app.state.settings)
    with app.state.sessions() as db:
        job = db.get(Job, job_id)
        if commit_succeeded:
            assert job.status == "succeeded"
            assert (app.state.settings.storage_dir / job.result_path / "report.json").is_file()
        else:
            assert job.status == "failed" and job.result_path is None
            assert len(list((app.state.settings.storage_dir / job_id).iterdir())) == 1
