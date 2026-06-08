from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """Documento fuente cargado antes del chunking.

    Atributos:
        id: Identificador estable del documento.
        content: Texto plano extraído desde la fuente original.
        metadata: Datos adicionales asociados al documento, como ruta o tipo.
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
            raise ValueError("The id of the document cannot be empty")
        if not self.content.strip():
            raise ValueError("The content of the document cannot be empty")
