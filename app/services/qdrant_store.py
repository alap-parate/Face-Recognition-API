from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import Settings
from app.services.face_engine import FaceVectorSample
from app.services.vector_types import RankedPerson

logger = logging.getLogger(__name__)

_POINT_NS = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")


def _point_id(org_id: str, external_id: str, sample_index: int) -> str:
    return str(
        uuid.uuid5(_POINT_NS, f"{org_id}:{external_id}:{sample_index}")
    )


class QdrantStore:
    """Qdrant: one point per enrollment sample; payload carries org, identity, and quality."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._collection = settings.qdrant_collection_name
        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )

    @property
    def client(self) -> QdrantClient:
        return self._client

    def health_check(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            logger.exception("Qdrant health check failed")
            return False

    def ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(
                size=self._settings.face_embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        for field in ("org_id", "person_id", "external_id"):
            try:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema="keyword",
                )
            except Exception:
                logger.debug("Payload index for %s may already exist", field)
        try:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="is_active",
                field_schema="bool",
            )
        except Exception:
            logger.debug("Payload index for is_active may already exist")

    def find_existing_person_id(self, org_id: str, external_id: str) -> str | None:
        records, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="org_id", match=MatchValue(value=org_id)),
                    FieldCondition(key="external_id", match=MatchValue(value=external_id)),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        pl = records[0].payload or {}
        pid = pl.get("person_id")
        return str(pid) if pid is not None else None

    def delete_identity_points(self, org_id: str, external_id: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key="org_id", match=MatchValue(value=org_id)),
                    FieldCondition(key="external_id", match=MatchValue(value=external_id)),
                ]
            ),
        )

    def upsert_enrollment(
        self,
        org_id: str,
        external_id: str,
        name: str,
        is_active: bool,
        person_id: str,
        kept_samples: list[FaceVectorSample],
        sample_count: int,
    ) -> None:
        points: list[PointStruct] = []
        for sample in kept_samples:
            emb = np.asarray(sample.embedding, dtype=np.float32)
            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm
            sid = sample.sample_index
            payload: dict[str, Any] = {
                "person_id": person_id,
                "org_id": org_id,
                "external_id": external_id,
                "name": name,
                "is_active": is_active,
                "sample_index": sid,
                "sample_count": sample_count,
                "detection_score": sample.detection_score,
                "blur_score": sample.blur_score,
                "quality_score": sample.quality_score,
                "bbox": sample.bbox,
            }
            points.append(
                PointStruct(
                    id=_point_id(org_id, external_id, sid),
                    vector=emb.tolist(),
                    payload=payload,
                )
            )
        self._client.upsert(collection_name=self._collection, points=points)

    def search_ranked_persons(
        self,
        query_embedding: np.ndarray,
        org_id: str,
        limited_top_k: int,
        probe_k: int,
    ) -> list[RankedPerson]:
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(q))
        if n > 0:
            q = q / n

        resp = self._client.query_points(
            collection_name=self._collection,
            query=q.tolist(),
            query_filter=Filter(
                must=[
                    FieldCondition(key="org_id", match=MatchValue(value=org_id)),
                    FieldCondition(key="is_active", match=MatchValue(value=True)),
                ]
            ),
            limit=max(1, probe_k),
            with_payload=True,
            with_vectors=False,
        )

        best: dict[str, tuple[float, dict[str, Any]]] = {}
        for hit in resp.points:
            pl = hit.payload or {}
            pid = pl.get("person_id")
            if pid is None:
                continue
            pid = str(pid)
            # Cosine similarity score in [~0,1] for normalized vectors
            distance = 1.0 - float(hit.score)
            prev = best.get(pid)
            if prev is None or distance < prev[0]:
                best[pid] = (distance, pl)

        ranked = sorted(best.items(), key=lambda x: x[1][0])[:limited_top_k]
        out: list[RankedPerson] = []
        for _pid, (dist, pl) in ranked:
            out.append(
                RankedPerson(
                    person_id=str(pl.get("person_id", "")),
                    distance=float(dist),
                    external_id=str(pl.get("external_id", "")),
                    name=str(pl.get("name", "")),
                    sample_count=int(pl.get("sample_count", 0)),
                )
            )
        return out

    def scroll_all_vectors_and_payloads(
        self,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Full scan for Faiss rebuild: matrix (N, dim) and parallel payload dicts."""
        rows: list[np.ndarray] = []
        payloads: list[dict[str, Any]] = []
        offset = None
        while True:
            records, offset = self._client.scroll(
                collection_name=self._collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not records:
                break
            for rec in records:
                raw_vec = rec.vector
                if raw_vec is None:
                    continue
                v = np.asarray(raw_vec, dtype=np.float32)
                rows.append(v)
                payloads.append(dict(rec.payload or {}))
            if offset is None:
                break
        if not rows:
            return (
                np.zeros((0, self._settings.face_embedding_dim), dtype=np.float32),
                [],
            )
        return np.stack(rows, axis=0), payloads


class QdrantDirectSearch:
    """Delegates search to Qdrant ANN + payload filters."""

    def __init__(self, store: QdrantStore) -> None:
        self._store = store

    def search_ranked_persons(
        self,
        query_embedding: np.ndarray,
        org_id: str,
        limited_top_k: int,
        probe_k: int,
    ) -> list[RankedPerson]:
        return self._store.search_ranked_persons(
            query_embedding, org_id, limited_top_k, probe_k
        )

    def sync_after_enroll(self) -> None:
        return None
