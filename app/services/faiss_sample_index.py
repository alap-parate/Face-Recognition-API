from __future__ import annotations

import threading

import faiss
import numpy as np

from app.core.config import Settings
from app.services.qdrant_store import QdrantStore
from app.services.vector_types import RankedPerson


class FaissFromQdrantSearch:
    """In-memory Faiss over all Qdrant points; filters org_id and is_active in software."""

    def __init__(self, settings: Settings, store: QdrantStore) -> None:
        self._settings = settings
        self._store = store
        self._dim = settings.face_embedding_dim
        self._lock = threading.Lock()
        self._index: faiss.Index | None = None
        self._payloads: list[dict] = []

    def sync_after_enroll(self) -> None:
        self.rebuild()

    def rebuild(self) -> None:
        matrix, payloads = self._store.scroll_all_vectors_and_payloads()
        with self._lock:
            if matrix.shape[0] == 0:
                self._index = None
                self._payloads = []
                return
            faiss.normalize_L2(matrix)
            index = faiss.IndexFlatIP(self._dim)
            index.add(matrix)
            self._index = index
            self._payloads = payloads

    def search_ranked_persons(
        self,
        query_embedding: np.ndarray,
        org_id: str,
        limited_top_k: int,
        probe_k: int,
    ) -> list[RankedPerson]:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, self._dim)
        faiss.normalize_L2(query)

        with self._lock:
            if self._index is None or self._index.ntotal == 0 or not self._payloads:
                return []
            n = int(self._index.ntotal)
            k = min(max(1, probe_k), n)
            similarities, indices = self._index.search(query, k)

        best: dict[str, tuple[float, dict]] = {}
        for ip, idx in zip(similarities[0], indices[0], strict=True):
            if idx < 0:
                continue
            pl = self._payloads[idx]
            if pl.get("org_id") != org_id:
                continue
            if pl.get("is_active") is not True:
                continue
            pid = pl.get("person_id")
            if pid is None:
                continue
            pid_s = str(pid)
            distance = 1.0 - float(ip)
            prev = best.get(pid_s)
            if prev is None or distance < prev[0]:
                best[pid_s] = (distance, pl)

        ranked = sorted(best.items(), key=lambda x: x[1][0])[:limited_top_k]
        out: list[RankedPerson] = []
        for _pid, (dist, pl) in ranked:
            eid = pl.get("employee_id")
            out.append(
                RankedPerson(
                    person_id=str(pl.get("person_id", "")),
                    distance=float(dist),
                    external_id=str(pl.get("external_id", "")),
                    name=str(pl.get("name", "")),
                    sample_count=int(pl.get("sample_count", 0)),
                    employee_id=str(eid) if eid is not None else None,
                )
            )
        return out
