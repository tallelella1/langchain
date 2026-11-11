from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import jwt
from sqlalchemy import func, select

from app.jobs import process_job
from app.models import Job, User, utcnow
from conftest import CSV, submit


def test_authentication_and_expired_or_tampered_tokens(client, app, auth):
    assert client.get("/api/jobs").status_code == 401
    headers = auth()
    assert client.get("/api/auth/me", headers=headers).json()["username"] == "alice"
    assert client.post("/api/auth/register", json={"username": "Alice", "password": "test-password-123"}).status_code == 409
    assert client.post("/api/auth/token", data={"username": "alice", "password": "wrong"}).status_code == 401
    assert client.post("/api/auth/token", data={"username": "unknown", "password": "wrong"}).status_code == 401
    with app.state.sessions() as db:
        user = db.scalar(select(User))
        assert "test-password" not in user.password_hash
        assert user.password_hash.startswith("$argon2id$")
        claims = {"sub": user.id, "exp": utcnow() - timedelta(seconds=1), "iat": utcnow(), "iss": "dataclean"}
    expired = jwt.encode(claims, app.state.settings.jwt_secret, algorithm="HS256")
    assert client.get("/api/jobs", headers={"Authorization": f"Bearer {expired}"}).status_code == 401
    claims["exp"] = utcnow() + timedelta(hours=1)
    forged = jwt.encode(claims, "other-secret-" * 4, algorithm="HS256")
    assert client.get("/api/jobs", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_complete_flow_and_owner_isolation(client, app, auth):
    alice, bob = auth(), auth("bob")
    response = submit(client, alice)
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "queued"
    assert client.get(f"/api/jobs/{job['id']}/download/cleaned", headers=alice).status_code == 409
    assert client.get(f"/api/jobs/{job['id']}", headers=bob).status_code == 404
    assert client.post(f"/api/jobs/{job['id']}/retry", headers=bob).status_code == 404
    assert client.get(f"/api/jobs/{job['id']}/download/report", headers=bob).status_code == 404
    assert client.get("/api/jobs", headers=bob).json()["total"] == 0
    assert process_job(job["id"], app.state.sessions, app.state.settings)
    result = client.get(f"/api/jobs/{job['id']}", headers=alice).json()
    assert result["status"] == "succeeded"
    assert result["summary"]["total_amount"] == "0.80"
    assert result["attempt_count"] == 1
    assert result["created_at"].endswith("Z")
    assert "input_path" not in result and "run_token" not in result
    for artifact in ["cleaned", "statistics", "report", "rejected"]:
        downloaded = client.get(f"/api/jobs/{job['id']}/download/{artifact}", headers=alice)
        assert downloaded.status_code == 200
        assert "attachment" in downloaded.headers["content-disposition"]
        assert downloaded.headers["cache-control"] == "no-store"
        assert client.get(f"/api/jobs/{job['id']}/download/{artifact}", headers=bob).status_code == 404
    report = client.get(f"/api/jobs/{job['id']}/download/report", headers=alice).json()
    assert report["summary"] == result["summary"]
    assert report["by_product"][0]["total_amount"] == "0.20"
    assert client.post(f"/api/jobs/{job['id']}/retry", headers=alice).status_code == 409
    assert submit(client, bob).status_code == 202


def test_idempotency_and_key_conflict(client, app, auth):
    headers = auth()
    first = submit(client, headers).json()
    replay = submit(client, headers)
    assert replay.status_code == 200
    assert replay.headers["Idempotent-Replayed"] == "true"
    assert replay.json()["id"] == first["id"]
    keyed = {**headers, "Idempotency-Key": "sale-batch-1"}
    assert submit(client, keyed).status_code == 202
    assert submit(client, keyed).status_code == 200
    assert submit(client, keyed, CSV + b"A3,2026-10-03,Mouse,1,1.00\n").status_code == 409
    with app.state.sessions() as db:
        assert db.scalar(select(func.count()).select_from(Job)) == 2
    assert len(list(app.state.settings.storage_dir.iterdir())) == 2


def test_concurrent_identical_uploads_create_one_job(client, app, auth):
    headers = auth()
    with ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(executor.map(lambda _: submit(client, headers), range(4)))
    assert sorted(response.status_code for response in responses) == [200, 200, 200, 202]
    assert len({response.json()["id"] for response in responses}) == 1
    assert len(list(app.state.settings.storage_dir.iterdir())) == 1


def test_pagination_filters_and_validation(client, auth):
    headers = auth()
    for index in range(3):
        assert submit(client, {**headers, "Idempotency-Key": f"batch-{index}"}).status_code == 202
    first = client.get("/api/jobs?page_size=2", headers=headers).json()
    second = client.get("/api/jobs?page=2&page_size=2", headers=headers).json()
    assert first["total"] == 3 and len(first["items"]) == 2 and len(second["items"]) == 1
    assert not {item["id"] for item in first["items"]} & {item["id"] for item in second["items"]}
    assert client.get("/api/jobs?status=failed", headers=headers).json()["total"] == 0
    for query in ["page=0", "page_size=101", "status=oops"]:
        assert client.get(f"/api/jobs?{query}", headers=headers).status_code == 422
    assert client.get("/api/jobs/not-a-uuid", headers=headers).status_code == 422


def test_upload_size_empty_extension_and_no_orphans(client, app, auth):
    headers = auth()
    app.state.settings.max_upload_bytes = len(CSV)
    assert submit(client, headers).status_code == 202  # exact boundary
    assert submit(client, headers, CSV + b"x").status_code == 413
    assert submit(client, headers, b"").status_code == 400
    assert client.post("/api/jobs", headers=headers, files={"file": ("bad.txt", CSV)}).status_code == 400
    assert len(list(app.state.settings.storage_dir.iterdir())) == 1


def test_multipart_request_limit_and_chunked_request_limit(client, auth):
    headers = auth()
    over_limit = b"x" * (10 * 1024 * 1024 + 64 * 1024 + 1)
    assert client.post("/api/jobs", headers=headers, content=over_limit).status_code == 413
    def chunks():
        yield over_limit[:1000]
        yield over_limit[1000:]
    response = client.post("/api/jobs", headers={**headers, "Content-Type": "multipart/form-data; boundary=abc"}, content=chunks())
    assert response.status_code == 413
