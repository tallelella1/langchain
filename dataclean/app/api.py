import hashlib
import logging
import re
import shutil
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID, uuid4

import jwt
from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .models import Job, User
from .processing import ARTIFACTS
from .schemas import JobPage, JobRead, JobStatus, Registration, Token, UserRead
from .security import DUMMY_HASH, create_token, decode_token, password_hash

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
oauth2 = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


def get_db(request: Request):
    with request.app.state.sessions() as db:
        yield db


DB = Annotated[Session, Depends(get_db)]


def current_user(request: Request, db: DB, token: Annotated[str, Depends(oauth2)]) -> User:
    unauthorized = HTTPException(401, "登录已过期或无效", headers={"WWW-Authenticate": "Bearer"})
    try:
        user_id = decode_token(token, request.app.state.settings)
    except jwt.InvalidTokenError:
        raise unauthorized
    user = db.get(User, user_id)
    if user is None:
        raise unauthorized
    return user


CurrentUser = Annotated[User, Depends(current_user)]


@router.post("/auth/register", response_model=UserRead, status_code=201, tags=["认证"])
def register(body: Registration, db: DB):
    user = User(username=body.username.lower(), password_hash=password_hash.hash(body.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "用户名已存在")
    return user


@router.post("/auth/token", response_model=Token, tags=["认证"])
def login(request: Request, db: DB, form: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = db.scalar(select(User).where(User.username == form.username.lower()))
    stored = user.password_hash if user else DUMMY_HASH
    if len(form.password) > 128 or not password_hash.verify(form.password, stored) or user is None:
        raise HTTPException(401, "用户名或密码错误", headers={"WWW-Authenticate": "Bearer"})
    return Token(access_token=create_token(user, request.app.state.settings))


@router.get("/auth/me", response_model=UserRead, tags=["认证"])
def me(user: CurrentUser):
    return user


def owned_job(job_id: UUID, user: User, db: Session) -> Job:
    job = db.scalar(select(Job).where(Job.id == str(job_id), Job.user_id == user.id))
    if job is None:
        raise HTTPException(404, "任务不存在")
    return job


@router.post("/jobs", response_model=JobRead, status_code=202, tags=["任务"])
def upload(
    request: Request, response: Response, user: CurrentUser, db: DB,
    file: Annotated[UploadFile, File()],
    idempotency_key: Annotated[str | None, Header(min_length=1, max_length=128)] = None,
):
    filename = (file.filename or "").replace("\\", "/").split("/")[-1]
    if not filename.lower().endswith(".csv") or len(filename) > 255 or re.search(r"[\x00-\x1f]", filename):
        raise HTTPException(400, "请上传文件名不超过 255 字符的 CSV 文件")
    settings = request.app.state.settings
    job_id = str(uuid4())
    folder = settings.storage_dir / job_id
    folder.mkdir(parents=True)
    digest = hashlib.sha256()
    size = 0
    committed = False
    try:
        with (folder / "input.part").open("wb") as target:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(413, f"文件超过 {settings.max_upload_bytes} 字节限制")
                digest.update(chunk)
                target.write(chunk)
        if size == 0:
            raise HTTPException(400, "上传文件不能为空")
        checksum = digest.hexdigest()
        key = idempotency_key if idempotency_key is not None else checksum

        def previous():
            return db.scalar(select(Job).where(Job.user_id == user.id, Job.idempotency_key == key))

        def reused(job):
            if job.sha256 != checksum:
                raise HTTPException(409, "Idempotency-Key 已用于不同文件")
            response.status_code = 200
            response.headers["Idempotent-Replayed"] = "true"
            return job

        existing = previous()
        if existing is not None:
            return reused(existing)
        (folder / "input.part").rename(folder / "input.csv")
        job = Job(
            id=job_id, user_id=user.id, idempotency_key=key, filename=filename,
            file_size=size, sha256=checksum, input_path=f"{job_id}/input.csv", status="queued",
        )
        db.add(job)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            existing = previous()
            if existing is None:
                raise
            return reused(existing)
        committed = True
        return job
    finally:
        file.file.close()
        if not committed:
            shutil.rmtree(folder)


@router.get("/jobs", response_model=JobPage, tags=["任务"])
def history(
    user: CurrentUser, db: DB,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 10,
    status: JobStatus | None = None,
):
    filters = [Job.user_id == user.id]
    if status is not None:
        filters.append(Job.status == status)
    total = db.scalar(select(func.count()).select_from(Job).where(*filters))
    items = db.scalars(
        select(Job).where(*filters).order_by(Job.created_at.desc(), Job.id.desc())
        .offset((page - 1) * page_size).limit(page_size)
    ).all()
    return JobPage(items=[JobRead.model_validate(item) for item in items], total=total, page=page, page_size=page_size)


@router.get("/jobs/{job_id}", response_model=JobRead, tags=["任务"])
def detail(job_id: UUID, user: CurrentUser, db: DB):
    return owned_job(job_id, user, db)


@router.post("/jobs/{job_id}/retry", response_model=JobRead, status_code=202, tags=["任务"])
def retry_job(job_id: UUID, user: CurrentUser, db: DB):
    job = owned_job(job_id, user, db)
    changed = db.execute(
        update(Job).where(Job.id == job.id, Job.user_id == user.id, Job.status == "failed").values(
            status="queued", error=None, started_at=None, finished_at=None,
            summary=None, result_path=None, lease_expires_at=None, run_token=None,
        )
    ).rowcount
    if changed != 1:
        db.rollback()
        raise HTTPException(409, "只有失败任务可以重试")
    db.commit()
    db.refresh(job)
    return job


@router.get("/jobs/{job_id}/download/{artifact}", tags=["下载"])
def download(
    request: Request, job_id: UUID, artifact: Literal["cleaned", "report", "statistics", "rejected"],
    user: CurrentUser, db: DB,
):
    job = owned_job(job_id, user, db)
    if job.status != "succeeded" or job.result_path is None:
        raise HTTPException(409, "任务尚未成功，暂无可下载结果")
    name = ARTIFACTS[artifact]
    path = request.app.state.settings.storage_dir / job.result_path / name
    if not path.is_file():
        logger.error("Missing artifact %s for job %s", artifact, job.id)
        raise HTTPException(500, "结果文件缺失，请联系管理员")
    return FileResponse(path, filename=f"{job.id}_{name}", media_type="application/json" if artifact == "report" else "text/csv", headers={"Cache-Control": "no-store"})
