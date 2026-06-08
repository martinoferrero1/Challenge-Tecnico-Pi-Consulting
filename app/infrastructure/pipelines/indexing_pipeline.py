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
    """Subset de settings requerido para indexar documentos.

    Atributos:
        source_document_is_default: Indica si se usa la estrategia especial del
            documento base.
        text_chunk_size: Tamaño de chunk para documentos genéricos.
        text_chunk_overlap: Overlap para documentos genéricos.
        embedding_provider: Provider de embeddings seleccionado.
        openai_api_key: API key de OpenAI.
        openai_embedding_model: Modelo de embeddings de OpenAI.
        cohere_api_key: API key de Cohere.
        cohere_embedding_model: Modelo de embeddings de Cohere.
        cohere_embedding_input_type: Tipo de input de embeddings de Cohere.
        gemini_api_key: API key de Gemini.
        gemini_embedding_model: Modelo de embeddings de Gemini.
        chroma_persist_dir: Carpeta de persistencia de Chroma.
        chroma_collection_name: Colección Chroma usada para indexar.
    """

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
    """Arma el caso de uso de indexación con infraestructura concreta.

    Args:
        settings: Configuración requerida para loader, chunker, embeddings y
            Chroma.

    Returns:
        Caso de uso listo para indexar el documento configurado.
    """
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
    """Elige estrategia de chunking según si el documento es el default.

    Args:
        settings: Configuración con ``source_document_is_default`` y parámetros
            genéricos de chunking.

    Returns:
        ``TextChunker`` configurado para documento default o estrategia
        recursiva.
    """
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
