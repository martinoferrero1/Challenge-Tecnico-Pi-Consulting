from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentChunk:
    """Fragmento indexable de un documento.

    Atributos:
        id: Identificador determinístico del chunk.
        content: Texto del fragmento enviado a embeddings y retrieval.
        metadata: Metadata serializable usada por el vector store.
    """

    id: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Valida que ``id`` y ``content`` no sean vacíos.

        Raises:
            ValueError: Si ``id`` o ``content`` quedan vacíos luego de hacer
                ``strip``.
        """
        if not self.id.strip():
            raise ValueError("The id of the document chunk cannot be empty")
        if not self.content.strip():
            raise ValueError("The content of the document chunk cannot be empty")
