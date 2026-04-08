from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class RankedPerson:
    person_id: str
    distance: float
    external_id: str
    name: str
    sample_count: int


class VectorSearch(Protocol):
    """Pluggable vector search (Qdrant direct or Faiss rebuilt from Qdrant)."""

    def search_ranked_persons(
        self,
        query_embedding: np.ndarray,
        org_id: str,
        limited_top_k: int,
        probe_k: int,
    ) -> list[RankedPerson]:
        """Return top persons by min cosine distance within org; only active identities."""
        ...

    def sync_after_enroll(self) -> None:
        """Reload secondary index if any (Faiss); no-op for pure Qdrant."""
