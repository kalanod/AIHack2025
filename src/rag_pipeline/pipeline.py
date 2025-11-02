from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from langchain_core.documents import Document

from .config import RAGPipelineConfig
from .preprocessing.csv_preprocessor import CSVPreprocessor


@dataclass
class RAGPipeline:
    config: RAGPipelineConfig

    def preprocess(self) -> List[Document]:
        preprocessor = CSVPreprocessor(self.config.preprocessing)
        return preprocessor.process()

    # The methods below will be implemented later on in the project lifecycle.
    def embed(self, documents: Iterable[Document]):  # pragma: no cover - stub
        raise NotImplementedError("Embedding stage is not implemented yet.")

    def index(self, embeddings):  # pragma: no cover - stub
        raise NotImplementedError("Vector store indexing is not implemented yet.")

    def retrieve(self, query: str):  # pragma: no cover - stub
        raise NotImplementedError("Retrieval stage is not implemented yet.")

    def generate(self, query: str):  # pragma: no cover - stub
        raise NotImplementedError("Generation stage is not implemented yet.")


__all__ = ["RAGPipeline"]
