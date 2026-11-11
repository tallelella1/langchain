from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import ROOT, Settings


class Base(DeclarativeBase):
    pass


def make_engine(settings: Settings):
    url = make_url(settings.database_url)
    options = {"pool_pre_ping": True}
    if url.get_backend_name() == "sqlite":
        if url.database and url.database != ":memory:":
            path = Path(url.database)
            if not path.is_absolute():
                path = ROOT / path
            path.parent.mkdir(parents=True, exist_ok=True)
            url = url.set(database=str(path))
        options["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **options)
    if url.get_backend_name() == "sqlite":
        @event.listens_for(engine, "connect")
        def configure_sqlite(connection, _):
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")
    return engine


def make_sessions(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)
