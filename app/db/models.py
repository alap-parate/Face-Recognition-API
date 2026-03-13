from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.config import get_settings
from app.db.base import Base

settings = get_settings()
EMBEDDING_DIM = settings.face_embedding_dim


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(
        String(128),
        unique=True,
        index=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(VECTOR(EMBEDDING_DIM), nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    last_enrolled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    samples: Mapped[list["FaceSample"]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index(
            "ix_persons_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={
                "m": settings.hnsw_m,
                "ef_construction": settings.hnsw_ef_construction,
            },
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class FaceSample(Base):
    __tablename__ = "face_samples"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    person_id: Mapped[int] = mapped_column(
        ForeignKey("persons.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    sample_index: Mapped[int] = mapped_column(Integer, nullable=False)
    detection_score: Mapped[float] = mapped_column(Float, nullable=False)
    blur_score: Mapped[float] = mapped_column(Float, nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    bbox: Mapped[dict[str, float] | None] = mapped_column(JSONB, nullable=True)
    embedding: Mapped[list[float]] = mapped_column(VECTOR(EMBEDDING_DIM), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    person: Mapped[Person] = relationship(back_populates="samples")

    __table_args__ = (
        UniqueConstraint(
            "person_id",
            "sample_index",
            name="uq_face_samples_person_sample_index",
        ),
    )

