"""Abstract interfaces for preprocessing raw documents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class BasePreprocessor(ABC):
    """A minimal interface for preprocessing raw knowledge base entries."""

    @abstractmethod
    def process(self) -> List[Document]:
        """Transform the raw source into LangChain ``Document`` objects."""

        raise NotImplementedError


__all__ = ["BasePreprocessor"]
