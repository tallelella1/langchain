"""FastAPI 服务入口（P2.4）。

启动：
    uvicorn api.main:app --host 127.0.0.1 --port 8000
    python -m api.main

管理接口使用 ``X-Admin-Token``，令牌来自 ``ASSISTANT_ADMIN_TOKEN``；
未配置时管理接口关闭。聊天接口与终端共用 ``rag`` 核心逻辑。
"""

from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from api import service  # noqa: E402
from api.models import (  # noqa: E402
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestTaskResponse,
)
from rag import answering, conversation, documents, retrieval, sources, vector_store  # noqa: E402

WEB_INDEX = config.PROJECT_DIR / "web" / "index.html"
WEB_ADMIN = config.PROJECT_DIR / "web" / "admin.html"


def _require_admin(
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> None:
    token = config.ADMIN_TOKEN
    if not token:
        raise HTTPException(
            status_code=503,
            detail=f"管理接口未配置 {config.ADMIN_TOKEN_ENV}",
        )
    if not x_admin_token or not secrets.compare_digest(x_admin_token, token):
        raise HTTPException(status_code=401, detail="管理令牌无效")


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ValueError):
        status = 422
    elif isinstance(exc, (conversation.ConversationError, answering.AnswerGenerationError)):
        status = 502
    elif isinstance(
        exc,
        (
            retrieval.RetrievalError,
            vector_store.VectorStoreError,
            documents.ManifestError,
            sources.SourceError,
        ),
    ):
        status = 503
    else:
        status = 500
    return HTTPException(status_code=status, detail=f"{type(exc).__name__}: {exc}")


def _sse(event: dict[str, Any]) -> str:
    name = event.get("event", "message")
    data = json.dumps(event, ensure_ascii=False, default=str)
    return f"event: {name}\ndata: {data}\n\n"


def create_app(
    *,
    ask_fn: Callable[[Any], dict[str, Any]] = service.ask,
    stream_fn: Callable[[Any], Iterator[dict[str, Any]]] = service.stream_events,
    task_manager: Any = None,
    readiness_fn: Callable[[], dict[str, Any]] = service.readiness,
    knowledge_fn: Callable[[], dict[str, Any]] = service.knowledge_status,
) -> FastAPI:
    tasks = task_manager or service.TASKS
    app = FastAPI(title="Atguigu Assistant API", version="0.1.0")

    @app.get("/health/live")
    def health_live() -> dict[str, Any]:
        return {"status": "alive"}

    @app.get("/health/ready")
    def health_ready():
        result = readiness_fn()
        return JSONResponse(
            status_code=200 if result.get("ready") else 503,
            content=result,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> dict[str, Any]:
        try:
            return ask_fn(request)
        except Exception as exc:  # noqa: BLE001
            raise _http_error(exc) from exc

    @app.post("/ask/stream")
    def ask_stream(request: AskRequest) -> StreamingResponse:
        def generate() -> Iterator[str]:
            try:
                for event in stream_fn(request):
                    yield _sse(event)
            except Exception as exc:  # noqa: BLE001
                yield _sse(
                    {
                        "event": "error",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post(
        "/admin/ingest",
        response_model=IngestTaskResponse,
        status_code=202,
        dependencies=[Depends(_require_admin)],
    )
    def start_ingest(body: IngestRequest) -> dict[str, Any]:
        try:
            return tasks.start(force=body.force)
        except service.TaskBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise _http_error(exc) from exc

    @app.get(
        "/admin/ingest/{task_id}",
        response_model=IngestTaskResponse,
        dependencies=[Depends(_require_admin)],
    )
    def ingest_status(task_id: str) -> dict[str, Any]:
        try:
            return tasks.get(task_id)
        except service.TaskNotFoundError as exc:
            raise HTTPException(status_code=404, detail="任务不存在") from exc

    @app.get(
        "/admin/ingest",
        response_model=list[IngestTaskResponse],
        dependencies=[Depends(_require_admin)],
    )
    def ingest_list() -> list[dict[str, Any]]:
        return tasks.list()

    @app.get(
        "/admin/knowledge",
        dependencies=[Depends(_require_admin)],
    )
    def knowledge() -> dict[str, Any]:
        try:
            return knowledge_fn()
        except Exception as exc:  # noqa: BLE001
            raise _http_error(exc) from exc

    @app.get("/admin")
    def admin_page() -> FileResponse:
        if not WEB_ADMIN.is_file():
            raise HTTPException(status_code=404, detail="管理页面文件不存在")
        return FileResponse(WEB_ADMIN, media_type="text/html")

    @app.get("/")
    def index() -> FileResponse:
        if not WEB_INDEX.is_file():
            raise HTTPException(status_code=404, detail="页面文件不存在")
        return FileResponse(WEB_INDEX, media_type="text/html")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000)
