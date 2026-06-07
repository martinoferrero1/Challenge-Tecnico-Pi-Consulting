from types import SimpleNamespace

from app.infrastructure.pipelines import indexing_pipeline


class FakeDocumentLoader:
    pass


class FakeTextChunker:
    def __init__(self, config) -> None:
        self.config = config


class FakeEmbeddingModel:
    pass


class FakeVectorStore:
    pass


def test_index_document_with_indexing_dependencies(monkeypatch) -> None:
    document_loader = FakeDocumentLoader()
    embedding_model = FakeEmbeddingModel()
    vector_store = FakeVectorStore()

    monkeypatch.setattr(
        indexing_pipeline,
        "DocxDocumentLoader",
        lambda: document_loader,
    )
    monkeypatch.setattr(indexing_pipeline, "TextChunker", FakeTextChunker)
    monkeypatch.setattr(
        indexing_pipeline,
        "create_embedding_model",
        lambda settings: embedding_model,
    )
    monkeypatch.setattr(
        indexing_pipeline,
        "ChromaVectorStore",
        lambda persist_dir, collection_name: vector_store,
    )

    use_case = indexing_pipeline.create_index_document_use_case(
        build_settings(text_chunk_size=500, text_chunk_overlap=75)
    )

    ingest_use_case = use_case.ingest_document_use_case
    assert ingest_use_case.document_loader is document_loader
    assert ingest_use_case.document_chunker.config.chunk_size == 500
    assert ingest_use_case.document_chunker.config.chunk_overlap == 75
    assert use_case.embedding_model is embedding_model
    assert use_case.vector_store is vector_store


def build_settings(**overrides: object) -> SimpleNamespace:
    defaults = {
        "text_chunk_size": 800,
        "text_chunk_overlap": 120,
        "embedding_provider": "openai",
        "openai_api_key": "openai-key",
        "openai_embedding_model": "text-embedding-3-small",
        "cohere_api_key": None,
        "cohere_embedding_model": "embed-v4.0",
        "cohere_embedding_input_type": "search_document",
        "gemini_api_key": None,
        "gemini_embedding_model": "gemini-embedding-2",
        "chroma_persist_dir": ".chroma",
        "chroma_collection_name": "documents",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)
