from alembic import context

from app.config import Settings
from app.db import Base, make_engine
from app import models  # noqa: F401 -- register model metadata

target_metadata = Base.metadata
engine = make_engine(context.config.attributes.get("settings") or Settings())
try:
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()
finally:
    engine.dispose()
