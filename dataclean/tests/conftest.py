import os

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy.engine import make_url

from app.config import ROOT, Settings
from app.main import create_app


@pytest.fixture
def app(tmp_path):
    # Set DATACLEAN_TEST_DATABASE_URL to a dedicated, empty PostgreSQL test DB.
    # Tests migrate/downgrade only this isolated database. Never point at real data.
    test_url = os.environ.get("DATACLEAN_TEST_DATABASE_URL")
    if test_url and not (make_url(test_url).database or "").endswith("_test"):
        raise RuntimeError("独立测试数据库的名字必须以 _test 结尾；禁止使用真实业务库")
    settings = Settings(
        _env_file=None,
        jwt_secret="test-secret-" * 4,
        database_url=test_url or f"sqlite:///{tmp_path / 'test.db'}",
        storage_dir=tmp_path / "files", task_backend="local", poll_seconds=0.1,
    )
    migration = Config(str(ROOT / "alembic.ini"))
    migration.attributes["settings"] = settings
    command.upgrade(migration, "head")
    instance = create_app(settings, start_worker=False)
    yield instance
    instance.state.engine.dispose()
    command.downgrade(migration, "base")


@pytest.fixture
def client(app):
    with TestClient(app) as instance:
        yield instance


@pytest.fixture
def auth(client):
    def create(username="alice"):
        response = client.post("/api/auth/register", json={"username": username, "password": "test-password-123"})
        assert response.status_code == 201
        response = client.post("/api/auth/token", data={"username": username, "password": "test-password-123"})
        assert response.status_code == 200
        return {"Authorization": f"Bearer {response.json()['access_token']}"}
    return create


CSV = b"order_id,order_date,product,quantity,unit_price\nA1,2026-10-01,Keyboard,2,0.10\nA2,2026-10-02,Mouse,3,0.20\n"


def submit(client, headers, content=CSV, **kwargs):
    return client.post("/api/jobs", headers=headers, files={"file": ("sales.csv", content, "text/csv")}, **kwargs)
