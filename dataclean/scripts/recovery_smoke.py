"""Verify Redis interruption and API restart against this project's Compose stack.

Run only on your learning/demo stack. The script briefly stops this stack's Redis
and restarts its API, and restores Redis in a finally block.
"""
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import httpx

from smoke import wait_for

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8000"


def compose(*args):
    subprocess.run(["docker", "compose", *args], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def main():
    username, password = "recovery_" + uuid4().hex[:12], uuid4().hex
    with httpx.Client(base_url=BASE, timeout=10) as client:
        assert client.post("/api/auth/register", json={"username": username, "password": password}).status_code == 201
        token = client.post("/api/auth/token", data={"username": username, "password": password})
        token.raise_for_status()
        headers = {"Authorization": f"Bearer {token.json()['access_token']}"}
        content = (ROOT / "examples" / "sales.csv").read_bytes()
        try:
            compose("stop", "redis")
            uploaded = client.post("/api/jobs", headers=headers, files={"file": ("sales.csv", content)})
            assert uploaded.status_code == 202, uploaded.text
            job_id = uploaded.json()["id"]
            # Cover more than one default dispatcher polling interval during the outage.
            time.sleep(3)
            assert client.get(f"/api/jobs/{job_id}", headers=headers).json()["status"] == "queued"
        finally:
            compose("start", "redis")
        result = wait_for(client, job_id, "succeeded", headers)
        assert result["summary"]["total_amount"] == "2078.40"
    compose("restart", "api")
    deadline = time.monotonic() + 60
    with httpx.Client(base_url=BASE, timeout=5) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get("/health")
                if response.status_code == 200:
                    break
            except httpx.TransportError:
                pass  # The restart is intentionally taking the server offline.
            time.sleep(0.2)
        else:
            raise AssertionError("API did not restart within the smoke-test budget")
        job = client.get(f"/api/jobs/{job_id}", headers=headers)
        job.raise_for_status()
        assert job.json()["status"] == "succeeded"
        assert client.get(f"/api/jobs/{job_id}/download/report", headers=headers).json()["summary"] == result["summary"]
    print("PASS: Redis outage accepted a durable queued job; processing resumed after Redis recovery; job and report survived API restart")


if __name__ == "__main__":
    main()
