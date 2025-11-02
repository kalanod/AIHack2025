"""Core configuration dataclasses for the RAG pipeline.

This module defines configuration containers that describe how different
subsystems of the retrieval-augmented generation (RAG) stack should be
constructed. Only the preprocessing related configuration is implemented at
this stage; the rest of the system will leverage these stubs in future
iterations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass()
class PreprocessingConfig:
    """Configuration parameters for textual preprocessing.

    Attributes
    ----------
    csv_path:
        Path to the source CSV file with the raw knowledge base entries.
    text_column:
        Name of the column that contains the raw text to be cleaned and split.
    metadata_columns:
        Additional columns that should be kept as metadata on the resulting
        LangChain ``Document`` objects.
    chunk_size:
        Maximum size (in characters) for each resulting text segment.
    chunk_overlap:
        Number of characters to overlap between consecutive segments to
        preserve context.
    cleaning_pipeline:
        An ordered list of callables that will be applied to the text before
        segmentation. Each callable should accept and return a string.
    encoding:
        Text encoding used when reading the CSV file.
    """

    csv_path: Path
    text_column: str = "text"
    metadata_columns: List[str] = field(default_factory=list)
    chunk_size: int = 1024
    chunk_overlap: int = 128
    cleaning_pipeline: List = field(default_factory=list)
    encoding: str = "utf-8"


@dataclass()
class EmbeddingConfig:
    """Configuration parameters for the embedding stage."""

    model_name: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0


@dataclass()
class VectorStoreConfig:
    """Placeholder configuration for the vector storage backend."""

    backend: Optional[str] = None


@dataclass()
class RetrievalConfig:
    """Placeholder configuration for the retrieval strategy."""

    k: int = 5


@dataclass()
class GenerationConfig:
    """Placeholder configuration for the LLM generation module."""

    model: Optional[str] = None


@dataclass()
class RAGPipelineConfig:
    """Aggregated configuration for the whole RAG pipeline."""

    preprocessing: PreprocessingConfig
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


__all__ = [
    "PreprocessingConfig",
    "EmbeddingConfig",
    "VectorStoreConfig",
    "RetrievalConfig",
    "GenerationConfig",
    "RAGPipelineConfig",
]
