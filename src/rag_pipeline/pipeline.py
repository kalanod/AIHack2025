from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List

from langchain_core.documents import Document

from .config import RAGPipelineConfig
from .preprocessing.csv_preprocessor import CSVPreprocessor

try:  # pragma: no cover - imported for runtime use
    import requests
except ModuleNotFoundError as exc:  # pragma: no cover - handled for clarity
    raise ModuleNotFoundError(
        "The 'requests' package is required for the embedding stage."
    ) from exc


@dataclass
class RAGPipeline:
    config: RAGPipelineConfig

    def preprocess(self) -> List[Document]:
        preprocessor = CSVPreprocessor(self.config.preprocessing)
        return preprocessor.process()

    # The methods below will be implemented later on in the project lifecycle.
    def embed(self, documents: Iterable[Document]) -> List[dict]:
        docs = list(documents)
        if not docs:
            return []

        model_name = (
            self.config.embedding.model_name
            or os.getenv("EMBEDDER_MODEL_NAME")
            or "text-embedding-3-small"
        )
        base_url = (
            self.config.embedding.base_url
            or os.getenv("BASE_URL")
            or "https://openrouter.ai/api/v1"
        ).rstrip("/")
        timeout = self.config.embedding.timeout

        api_key = os.getenv("EMBEDDER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "EMBEDDER_API_KEY environment variable must be set for embedding."
            )

        url = f"{base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_name,
            "input": [doc.page_content for doc in docs],
        }

        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()

        body = response.json()
        if not isinstance(body, dict) or "data" not in body:
            raise ValueError("Unexpected response format from embedding service.")

        embeddings = body["data"]
        if not isinstance(embeddings, list) or len(embeddings) != len(docs):
            raise ValueError("Embedding response does not match document count.")

        results = []
        for document, item in zip(docs, embeddings):
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(embedding, list):
                raise ValueError("Embedding vector missing from response data.")
            results.append({"document": document, "embedding": embedding})

        return results

    def index(self, embeddings):  # pragma: no cover - stub
        raise NotImplementedError("Vector store indexing is not implemented yet.")

    def retrieve(self, query: str):  # pragma: no cover - stub
        raise NotImplementedError("Retrieval stage is not implemented yet.")

    def generate(self, query: str):  # pragma: no cover - stub
        raise NotImplementedError("Generation stage is not implemented yet.")


__all__ = ["RAGPipeline"]
