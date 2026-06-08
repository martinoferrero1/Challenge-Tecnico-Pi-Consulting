from dataclasses import dataclass

from app.domain.entities.document_chunk import DocumentChunk


@dataclass(frozen=True)
class RetrievedChunk:
    """Chunk devuelto por búsqueda vectorial junto con su score.

    Atributos:
        chunk: Fragmento de documento recuperado.
        similarity_score: Score de similitud devuelto por el vector store.
    """

    chunk: DocumentChunk
    similarity_score: float

    def __post_init__(self) -> None:
        """Valida que ``similarity_score`` no sea negativo.

        Raises:
            ValueError: Si ``similarity_score`` es menor que cero.
        """
        if self.similarity_score < 0:
            raise ValueError("The similarity score cannot be negative")
