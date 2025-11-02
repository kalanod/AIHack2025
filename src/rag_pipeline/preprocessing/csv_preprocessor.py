"""Implementation of CSV preprocessing for the RAG pipeline."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ..config import PreprocessingConfig
from ..utils.text import DEFAULT_CLEANING_PIPELINE
from .base import BasePreprocessor


@dataclass
class CSVPreprocessor(BasePreprocessor):
    """Preprocess tabular data exported as CSV files."""

    config: PreprocessingConfig

    def __post_init__(self) -> None:
        self._csv_path: Path = self.config.csv_path
        self._text_column: str = self.config.text_column
        self._metadata_columns: Sequence[str] = (
            self.config.metadata_columns
            if self.config.metadata_columns
            else []
        )
        self._cleaning_steps: Sequence[Callable[[str], str]] = (
            self.config.cleaning_pipeline or DEFAULT_CLEANING_PIPELINE
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def process(self) -> List[Document]:
        dataframe = self._load_dataframe()
        documents: List[Document] = []
        for index, row in dataframe.iterrows():
            raw_text = row.get(self._text_column)
            if not isinstance(raw_text, str):
                continue
            cleaned_text = self._apply_cleaning_steps(raw_text)
            metadata = self._build_metadata(index=index, row=row)
            documents.extend(self._split_into_documents(cleaned_text, metadata))
        return documents

    # Internal helpers -------------------------------------------------
    def _load_dataframe(self) -> pd.DataFrame:
        dataframe = pd.read_csv(
            self._csv_path,
            encoding=self.config.encoding,
        )
        if self._text_column not in dataframe.columns:
            raise ValueError(
                f"Text column '{self._text_column}' is missing in {self._csv_path}"
            )
        return dataframe

    def _apply_cleaning_steps(self, text: str) -> str:
        for cleaning_step in self._cleaning_steps:
            text = cleaning_step(text)
        return text

    def _build_metadata(self, index: int, row: pd.Series) -> Dict[str, object]:
        metadata: Dict[str, object] = {
            "row_index": index,
            "source_path": str(self._csv_path),
        }
        candidate_columns: Iterable[str]
        if self._metadata_columns:
            candidate_columns = self._metadata_columns
        else:
            candidate_columns = [
                column
                for column in row.index
                if column != self._text_column
            ]
        for column in candidate_columns:
            if column not in row:
                continue
            value = row[column]
            if pd.isna(value):
                continue
            metadata[column] = self._normalize_metadata_value(value)
        return metadata

    @staticmethod
    def _normalize_metadata_value(value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = ast.literal_eval(stripped)
                    return parsed
                except (ValueError, SyntaxError):
                    return stripped
            return stripped
        return value

    def _split_into_documents(self, text: str, metadata: Dict[str, object]) -> List[Document]:
        chunks = self._splitter.split_text(text)
        documents: List[Document] = []
        for chunk_index, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        **metadata,
                        "chunk_index": chunk_index,
                        "chunk_count": len(chunks),
                    },
                )
            )
        return documents


__all__ = ["CSVPreprocessor"]
