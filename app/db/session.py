from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.db.base import Base
from app.db import models  # noqa: F401

settings = get_settings()

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def initialize_database() -> None:
    try:
        with engine.begin() as connection:
            if settings.create_vector_extension:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            Base.metadata.create_all(bind=connection)
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize PostgreSQL/pgvector. "
            "Verify DATABASE_URL and ensure the vector extension is available."
        ) from exc


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

