from pathlib import Path
from typing import Any, Sequence

from app.domain.entities.document_chunk import DocumentChunk
from app.domain.entities.retrieval import RetrievedChunk


class ChromaVectorStore:
    """Adaptador de Chroma para persistencia y búsqueda vectorial."""

    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        vector_store: Any | None = None,
    ) -> None:
        """Inicializa o recibe una instancia de Chroma.

        Args:
            persist_dir: Carpeta donde Chroma persiste la colección.
            collection_name: Nombre de la colección vectorial.
            vector_store: Instancia inyectada para tests o configuración manual.
        """
        self.vector_store = vector_store or self._build_vector_store(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )

    async def add_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Inserta o actualiza chunks y embeddings en Chroma.

        Args:
            chunks: Chunks de dominio que se van a persistir.
            embeddings: Vectores correspondientes a ``chunks`` en el mismo
                orden.

        Raises:
            ValueError: Si ``chunks`` y ``embeddings`` tienen distinta cantidad.
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length")

        self.vector_store._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            metadatas=[self._build_metadata(chunk) for chunk in chunks],
            embeddings=[list(embedding) for embedding in embeddings],
        )

    async def search(
        self,
        query_embedding: Sequence[float],
        limit: int,
    ) -> list[RetrievedChunk]:
        """Busca los chunks más similares a un embedding.

        Args:
            query_embedding: Vector de la pregunta de retrieval.
            limit: Cantidad máxima de chunks a recuperar.

        Returns:
            Lista de ``RetrievedChunk`` ordenada por similitud.

        Raises:
            ValueError: Si ``limit`` no es mayor que cero.
        """
        if limit <= 0:
            raise ValueError("Search limit must be greater than zero")

        docs_and_scores = self.vector_store.similarity_search_by_vector_with_relevance_scores(
            embedding=list(query_embedding),
            k=limit,
        )

        return [
            RetrievedChunk(
                chunk=DocumentChunk(
                    id=str(document.metadata.get("chunk_id", "")),
                    content=str(document.page_content),
                    metadata=self._string_metadata(document.metadata),
                ),
                similarity_score=float(score),
            )
            for document, score in docs_and_scores
        ]

    def _build_vector_store(self, persist_dir: str, collection_name: str) -> Any:
        """Construye la instancia persistente de Chroma.

        Args:
            persist_dir: Carpeta donde se guarda el índice.
            collection_name: Nombre de la colección Chroma.

        Returns:
            Instancia de ``langchain_chroma.Chroma``.
        """
        from langchain_chroma import Chroma

        path = Path(persist_dir)
        path.mkdir(parents=True, exist_ok=True)

        return Chroma(
            collection_name=collection_name,
            persist_directory=str(path),
        )

    def _build_metadata(self, chunk: DocumentChunk) -> dict[str, str]:
        """Prepara la metadata almacenada junto al chunk.

        Args:
            chunk: Chunk cuya metadata se va a persistir.

        Returns:
            Metadata del chunk más ``chunk_id``.
        """
        return {
            **chunk.metadata,
            "chunk_id": chunk.id,
        }

    def _string_metadata(self, metadata: dict[str, Any]) -> dict[str, str]:
        """Convierte metadata heterogénea a strings.

        Args:
            metadata: Diccionario de metadata con valores arbitrarios.

        Returns:
            Diccionario con claves y valores convertidos a ``str``.
        """
        return {
            str(key): str(value)
            for key, value in metadata.items()
            if value is not None
        }
