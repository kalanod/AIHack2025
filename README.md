# AIHack2025 RAG Platform

Этот репозиторий содержит прототип Retrieval-Augmented Generation (RAG)
платформы, построенной на базе компонентов LangChain. На текущем этапе
реализован модуль предобработки текстовых данных, который готовит исходные CSV
файлы для последующей индексации и поиска.

## Архитектура решения

Проект организован по модульному принципу, чтобы упростить расширение
функциональности и замену компонентов:

```
src/
└── rag_pipeline/
    ├── config.py              # Конфигурация всех подсистем пайплайна
    ├── pipeline.py            # Высокоуровневый оркестратор RAG-процесса
    ├── preprocessing/         # Предобработка и очистка данных
    │   ├── base.py            # Абстрактный интерфейс предобработки
    │   └── csv_preprocessor.py# Реализация для CSV-источников
    └── utils/                 # Вспомогательные утилиты
        └── text.py            # Нормализация и очистка текста
```

Следующие подсистемы запланированы и будут реализованы позже:

- **Embedding** — вычисление векторных представлений документов.
- **Vector Store** — хранение и управление индексацией.
- **Retriever** — подбор релевантных документов к пользовательскому запросу.
- **Generator** — LLM-компонент, формирующий ответ на базе найденного контента.

Все подсистемы будут конфигурироваться через `RAGPipelineConfig` и работать в
рамках `RAGPipeline`.

## Модуль предобработки CSV

`CSVPreprocessor` реализует поток обработки:

1. Загрузка таблицы (`pandas.read_csv`).
2. Очистка текста (удаление неразрывных пробелов, лишних переводов строк,
   BOM-символов и т. п.).
3. Разбиение на сегменты с помощью `RecursiveCharacterTextSplitter` из
   LangChain.
4. Сбор метаданных (ID документа, теги и другие столбцы).

Результатом работы являются списки `Document` из LangChain с очищенными кусками
текста и подробными метаданными.

### Пример использования

```python
from pathlib import Path

from rag_pipeline import PreprocessingConfig, RAGPipeline, RAGPipelineConfig

config = RAGPipelineConfig(
    preprocessing=PreprocessingConfig(
        csv_path=Path("data/articles.csv"),
        text_column="text",
        metadata_columns=["id", "tags"],
        chunk_size=800,
        chunk_overlap=120,
    )
)

pipeline = RAGPipeline(config)
documents = pipeline.preprocess()
print(documents[0].metadata)
print(documents[0].page_content)
```

## Требования

- Python 3.10+
- [LangChain](https://python.langchain.com/)
- pandas

Список зависимостей находится в `requirements.txt`.

## Тестирование

```bash
pytest
```
