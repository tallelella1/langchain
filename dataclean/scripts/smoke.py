"""Run against a real server: python scripts/smoke.py [http://127.0.0.1:8000]."""
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx


def wait_for(client, job_id, expected, headers):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}", headers=headers)
        response.raise_for_status()
        job = response.json()
        if job["status"] in ("succeeded", "failed"):
            assert job["status"] == expected, job
            return job
        time.sleep(0.2)
    raise AssertionError(f"Job {job_id} did not reach {expected} within the 60-second smoke-test budget")


def main():
    root = Path(__file__).resolve().parents[1]
    with httpx.Client(base_url=sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000", timeout=10) as client:
        client.get("/health").raise_for_status()
        accounts = []
        for label in ("owner", "other"):
            username = f"smoke_{label}_{uuid4().hex[:10]}"
            password = uuid4().hex
            assert client.post("/api/auth/register", json={"username": username, "password": password}).status_code == 201
            response = client.post("/api/auth/token", data={"username": username, "password": password})
            response.raise_for_status()
            accounts.append({"Authorization": f"Bearer {response.json()['access_token']}"})
        owner, other = accounts
        content = (root / "examples" / "sales.csv").read_bytes()
        uploaded = client.post("/api/jobs", headers=owner, files={"file": ("sales.csv", content)})
        assert uploaded.status_code == 202, uploaded.text
        job_id = uploaded.json()["id"]
        job = wait_for(client, job_id, "succeeded", owner)
        assert job["summary"]["total_amount"] == "2078.40"
        assert job["summary"]["valid_rows"] == 4
        assert job["summary"]["invalid_rows"] == 3
        replay = client.post("/api/jobs", headers=owner, files={"file": ("sales.csv", content)})
        assert replay.status_code == 200 and replay.json()["id"] == job_id
        for artifact in ("cleaned", "statistics", "report", "rejected"):
            download = client.get(f"/api/jobs/{job_id}/download/{artifact}", headers=owner)
            assert download.status_code == 200 and download.content
            assert client.get(f"/api/jobs/{job_id}/download/{artifact}", headers=other).status_code == 404
        assert client.get("/api/jobs", headers=other).json()["total"] == 0
        invalid = client.post("/api/jobs", headers=owner, files={"file": ("bad.csv", b"bad,columns\nx,y\n")})
        invalid_id = invalid.json()["id"]
        wait_for(client, invalid_id, "failed", owner)
        assert client.post(f"/api/jobs/{invalid_id}/retry", headers=owner).status_code == 202
        assert wait_for(client, invalid_id, "failed", owner)["attempt_count"] == 2
        first = client.get("/api/jobs?page_size=1", headers=owner).json()
        second = client.get("/api/jobs?page=2&page_size=1", headers=owner).json()
        assert first["total"] == 2 and first["items"][0]["id"] != second["items"][0]["id"]
        print("PASS: real HTTP auth, upload, background processing, idempotency, four downloads, ownership, failure/retry, pagination")
        print(f"Sample job: {job_id}; amount: {job['summary']['total_amount']}")


if __name__ == "__main__":
    main()
