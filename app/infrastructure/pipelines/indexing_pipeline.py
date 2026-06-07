from typing import Protocol

from app.application.use_cases.index_document import IndexDocumentUseCase
from app.application.use_cases.ingest_document import IngestDocumentUseCase
from app.infrastructure.document_loaders.docx_document_loader import DocxDocumentLoader
from app.infrastructure.embedding_models.embedding_model_factory import (
    create_embedding_model,
)
from app.infrastructure.text_splitter.text_chunker import (
    DEFAULT_DOCUMENT_SECTION_CHUNK_OVERLAP,
    DEFAULT_DOCUMENT_SECTION_CHUNK_SIZE,
    DEFAULT_DOCUMENT_SECTIONS_CHUNKING_STRATEGY,
    TextChunker,
    TextSplitterConfig,
)
from app.infrastructure.vector_store.chroma_vector_store import ChromaVectorStore


class IndexDocumentSettings(Protocol):
    source_document_is_default: bool
    text_chunk_size: int
    text_chunk_overlap: int
    embedding_provider: str
    openai_api_key: str | None
    openai_embedding_model: str
    cohere_api_key: str | None
    cohere_embedding_model: str
    cohere_embedding_input_type: str
    gemini_api_key: str | None
    gemini_embedding_model: str
    chroma_persist_dir: str
    chroma_collection_name: str


def create_index_document_use_case(
    settings: IndexDocumentSettings,
) -> IndexDocumentUseCase:
    document_loader = DocxDocumentLoader()
    document_chunker = _create_document_chunker(settings)
    ingest_document_use_case = IngestDocumentUseCase(
        document_loader=document_loader,
        document_chunker=document_chunker,
    )

    return IndexDocumentUseCase(
        ingest_document_use_case=ingest_document_use_case,
        embedding_model=create_embedding_model(settings),
        vector_store=ChromaVectorStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        ),
    )


def _create_document_chunker(settings: IndexDocumentSettings) -> TextChunker:
    if settings.source_document_is_default:
        return TextChunker(
            config=TextSplitterConfig(
                chunk_size=DEFAULT_DOCUMENT_SECTION_CHUNK_SIZE,
                chunk_overlap=DEFAULT_DOCUMENT_SECTION_CHUNK_OVERLAP,
                strategy=DEFAULT_DOCUMENT_SECTIONS_CHUNKING_STRATEGY,
            )
        )

    return TextChunker(
        config=TextSplitterConfig(
            chunk_size=settings.text_chunk_size,
            chunk_overlap=settings.text_chunk_overlap,
        )
    )
