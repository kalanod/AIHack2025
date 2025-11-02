"""High-level orchestration of the Retrieval-Augmented Generation pipeline.

The module intentionally provides only architectural scaffolding for the full
RAG system. The goal is to outline how the preprocessing module integrates with
other LangChain-powered components that will be implemented in subsequent
iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from langchain_core.documents import Document

from .config import RAGPipelineConfig
from .preprocessing.csv_preprocessor import CSVPreprocessor


@dataclass
class RAGPipeline:
    """Container object that wires together the RAG subsystems.

    Only the preprocessing stage is functional at this moment. All other stages
    expose placeholders that will be connected to LangChain components in future
    commits.
    """

    config: RAGPipelineConfig

    def preprocess(self) -> List[Document]:
        """Run the preprocessing pipeline defined in the configuration."""

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
