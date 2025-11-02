from pathlib import Path

import pandas as pd
import pytest
from langchain_core.documents import Document

from src.rag_pipeline import RAGPipelineConfig, PreprocessingConfig, RAGPipeline


def _create_sample_csv(tmp_path: Path) -> Path:
    data = pd.DataFrame(
        [
            {
                "id": "doc_001",
                "text": "Первый абзац.\n\nВторой абзац.",
                "tags": "['A', 'B']",
            }
        ]
    )
    csv_path = tmp_path / "sample.csv"
    data.to_csv(csv_path, index=False)
    return csv_path


def test_preprocessing_produces_documents(tmp_path: Path):
    csv_path = _create_sample_csv(tmp_path)
    config = RAGPipelineConfig(
        preprocessing=PreprocessingConfig(
            csv_path=csv_path,
            text_column="text",
            metadata_columns=["id", "tags"],
            chunk_size=20,
            chunk_overlap=0,
        )
    )

    pipeline = RAGPipeline(config)
    documents = pipeline.preprocess()

    assert all(isinstance(doc, Document) for doc in documents)
    assert documents, "Preprocessing should yield at least one document"

    metadata = documents[0].metadata
    assert metadata["id"] == "doc_001"
    assert metadata["tags"] == ["A", "B"]
    assert metadata["chunk_index"] == 0


@pytest.mark.parametrize(
    "text,expected_chunks",
    [
        ("abc" * 10, 2),
        ("\n\n".join(["paragraph"] * 3), 3),
    ],
)
def test_chunking_respects_configuration(tmp_path: Path, text: str, expected_chunks: int):
    csv_path = _create_sample_csv(tmp_path)
    df = pd.read_csv(csv_path)
    df.loc[0, "text"] = text
    df.to_csv(csv_path, index=False)

    config = RAGPipelineConfig(
        preprocessing=PreprocessingConfig(
            csv_path=csv_path,
            text_column="text",
            chunk_size=15,
            chunk_overlap=0,
        )
    )

    pipeline = RAGPipeline(config)
    documents = pipeline.preprocess()

    assert len(documents) == expected_chunks

@pytest.mark.parametrize(
    "path,expected_chunks",
    [
        (Path("../res/train_data.csv"), 5),
    ],
)
def test_on_csv(path: Path, expected_chunks: int):
    config = RAGPipelineConfig(
        preprocessing=PreprocessingConfig(
            csv_path=path,
            text_column="text",
            chunk_size=1000,
            chunk_overlap=100,
        )
    )

    pipeline = RAGPipeline(config)
    documents = pipeline.preprocess()

    print(documents[0])