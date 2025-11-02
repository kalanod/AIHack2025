from pathlib import Path
from typing import Optional

import pytest

from langchain_core.documents import Document

from src.rag_pipeline import (
    EmbeddingConfig,
    PreprocessingConfig,
    RAGPipeline,
    RAGPipelineConfig,
)


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def build_pipeline(base_url: Optional[str] = None) -> RAGPipeline:
    preprocessing = PreprocessingConfig(csv_path=Path("dummy.csv"))
    embedding = EmbeddingConfig(model_name="text-embedding-3-small", base_url=base_url)
    config = RAGPipelineConfig(preprocessing=preprocessing, embedding=embedding)
    return RAGPipeline(config=config)


def test_embed_documents(monkeypatch):
    documents = [Document(page_content="hello"), Document(page_content="world")]
    called = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        called["url"] = url
        called["headers"] = headers
        called["json"] = json
        called["timeout"] = timeout
        return DummyResponse(
            {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]},
                ]
            }
        )

    monkeypatch.setenv("EMBEDDER_API_KEY", "test-key")
    monkeypatch.setenv("BASE_URL", "https://openrouter.ai/api/v1")

    import src.rag_pipeline.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module.requests, "post", fake_post)

    pipeline = build_pipeline()
    result = pipeline.embed(documents)

    assert len(result) == 2
    assert [item["embedding"] for item in result] == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    assert called["url"] == "https://openrouter.ai/api/v1/embeddings"
    assert called["headers"]["Authorization"] == "Bearer test-key"
    assert called["json"] == {
        "model": "text-embedding-3-small",
        "input": ["hello", "world"],
    }


def test_embed_documents_empty_input(monkeypatch):
    pipeline = build_pipeline()
    monkeypatch.setenv("EMBEDDER_API_KEY", "test-key")

    import src.rag_pipeline.pipeline as pipeline_module

    def fail_post(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("requests.post should not be called for empty input")

    monkeypatch.setattr(pipeline_module.requests, "post", fail_post)

    assert pipeline.embed([]) == []


def test_embed_documents_requires_api_key(monkeypatch):
    pipeline = build_pipeline(base_url="https://openrouter.ai/api/v1")
    monkeypatch.delenv("EMBEDDER_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        pipeline.embed([Document(page_content="hello")])
