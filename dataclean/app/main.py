import logging
from contextlib import asynccontextmanager
from threading import Event, Thread

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from .api import router
from .config import ROOT, Settings
from .db import make_engine, make_sessions
from .jobs import local_loop
from .upload_limit import UploadBodyLimit


def create_app(settings: Settings | None = None, *, start_worker: bool = True) -> FastAPI:
    settings = settings or Settings()
    settings.storage_dir = settings.storage_dir.resolve()
    engine = make_engine(settings)
    sessions = make_sessions(engine)

    @asynccontextmanager
    async def lifespan(app):
        settings.storage_dir.mkdir(parents=True, exist_ok=True)
        with engine.connect() as connection:
            connection.execute(text("SELECT id FROM jobs LIMIT 1"))
        stop = Event()
        worker = None
        if settings.task_backend == "local" and start_worker:
            worker = Thread(target=local_loop, args=(stop, sessions, settings), daemon=True)
            worker.start()
        try:
            yield
        finally:
            stop.set()
            if worker:
                worker.join()
            engine.dispose()

    app = FastAPI(title="DataClean · 异步数据处理与报表平台", version="1.0.0", lifespan=lifespan)
    app.add_middleware(UploadBodyLimit, max_file_bytes=settings.max_upload_bytes)
    app.state.settings = settings
    app.state.engine = engine
    app.state.sessions = sessions
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(ROOT / "app" / "static" / "index.html")

    @app.get("/sample.csv", include_in_schema=False)
    def sample():
        return FileResponse(ROOT / "examples" / "sales.csv", filename="sales.csv", media_type="text/csv")

    @app.get("/health", tags=["运维"])
    def health():
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"status": "ok", "task_backend": settings.task_backend}

    return app
