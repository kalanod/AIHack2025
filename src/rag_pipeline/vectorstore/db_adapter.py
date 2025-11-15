"""Helpers for interacting with vector store backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional, Sequence

from langchain_core.documents import Document

DEFAULT_MAX_BATCH_SIZE = 5461


def _chunk(iterable: Sequence[Document], size: int) -> Iterator[Sequence[Document]]:
    """Yield ``size`` sized chunks from ``iterable``."""

    for start in range(0, len(iterable), size):
        yield iterable[start : start + size]


@dataclass
class BatchedVectorStore:
    """Wrap a LangChain vector store to respect provider batch limits."""

    db: object
    max_batch_size: Optional[int] = field(default=None)

    def __post_init__(self) -> None:
        if self.max_batch_size is None:
            self.max_batch_size = self._infer_batch_size()

    def add_documents(self, documents: Iterable[Document]) -> None:
        """Persist documents in batches accepted by the backend."""

        docs = list(documents)
        if not docs:
            return

        batch_size = self.max_batch_size or DEFAULT_MAX_BATCH_SIZE
        if batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")

        for batch in _chunk(docs, batch_size):
            self.db.add_documents(documents=list(batch))

    def _infer_batch_size(self) -> int:
        collection = getattr(self.db, "_collection", None)
        client = getattr(collection, "_client", None)
        for attribute in ("max_batch_size", "_max_batch_size"):
            value = getattr(client, attribute, None)
            if isinstance(value, int) and value > 0:
                return value
        return DEFAULT_MAX_BATCH_SIZE
