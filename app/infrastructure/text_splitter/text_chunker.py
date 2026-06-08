from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Any

from app.domain.entities.document import Document
from app.domain.entities.document_chunk import DocumentChunk


RECURSIVE_CHUNKING_STRATEGY = "recursive"
DEFAULT_DOCUMENT_SECTIONS_CHUNKING_STRATEGY = "default_document_sections"
DEFAULT_DOCUMENT_SECTION_CHUNK_SIZE = 2000
DEFAULT_DOCUMENT_SECTION_CHUNK_OVERLAP = 0
_SUPPORTED_CHUNKING_STRATEGIES = {
    RECURSIVE_CHUNKING_STRATEGY,
    DEFAULT_DOCUMENT_SECTIONS_CHUNKING_STRATEGY,
}
_SECTION_TITLE_PATTERN = re.compile(r"^(?P<title>[^:\n]{1,120}):\s+")


@dataclass(frozen=True)
class TextSplitterConfig:
    """Configuración de estrategia y tamaño de chunking.

    Atributos:
        chunk_size: Tamaño máximo del chunk en la estrategia recursiva.
        chunk_overlap: Overlap entre chunks en la estrategia recursiva.
        strategy: Estrategia de chunking seleccionada.
    """

    chunk_size: int = 800
    chunk_overlap: int = 120
    strategy: str = RECURSIVE_CHUNKING_STRATEGY

    def __post_init__(self) -> None:
        """Valida ``strategy``, ``chunk_size`` y ``chunk_overlap``.

        Raises:
            ValueError: Si ``strategy`` no está soportada, si ``chunk_size`` no
                es positivo, o si ``chunk_overlap`` es inválido.
        """
        if self.strategy not in _SUPPORTED_CHUNKING_STRATEGIES:
            raise ValueError(f"Unsupported chunking strategy: {self.strategy}")
        if self.chunk_size <= 0:
            raise ValueError("The chunk size must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("The chunk overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("The chunk overlap must be lower than the chunk size")


class TextChunker:
    """Chunker de documentos con estrategia genérica o por secciones."""

    def __init__(
        self,
        config: TextSplitterConfig | None = None,
        splitter: Any | None = None,
    ) -> None:
        """Inicializa el splitter con configuración o dependencia inyectada.

        Args:
            config: Configuración de chunking. Si es ``None``, usa valores por
                defecto.
            splitter: Splitter compatible con LangChain, opcional para tests.
        """
        self.config = config or TextSplitterConfig()
        self.splitter = splitter or self._build_splitter()

    def chunk(self, document: Document) -> tuple[DocumentChunk, ...]:
        """Divide un documento según la estrategia configurada.

        Args:
            document: Documento de dominio a dividir.

        Returns:
            Tupla de chunks generados desde ``document``.
        """
        if self.config.strategy == DEFAULT_DOCUMENT_SECTIONS_CHUNKING_STRATEGY:
            return self._chunk_default_document_sections(document)

        return self._chunk_with_splitter(document)

    def _chunk_with_splitter(self, document: Document) -> tuple[DocumentChunk, ...]:
        """Aplica RecursiveCharacterTextSplitter y mapea chunks de dominio.

        Args:
            document: Documento que se divide con el splitter genérico.

        Returns:
            Chunks de dominio con metadata normalizada.
        """
        source_document = self._to_source_document(document)
        split_documents = self.splitter.split_documents([source_document])
        chunks: list[DocumentChunk] = []

        for chunk_index, split_document in enumerate(split_documents):
            content = str(split_document.page_content).strip()
            if not content:
                continue

            metadata = self._string_metadata(
                {
                    **document.metadata,
                    **dict(getattr(split_document, "metadata", {}) or {}),
                    "document_id": document.id,
                    "chunk_index": str(chunk_index),
                }
            )

            chunks.append(
                DocumentChunk(
                    id=self._build_chunk_id(document.id, chunk_index, content),
                    content=content,
                    metadata=metadata,
                )
            )

        return tuple(chunks)

    def _chunk_default_document_sections(
        self,
        document: Document,
    ) -> tuple[DocumentChunk, ...]:
        """Divide el documento default en secciones tituladas sin overlap.

        Args:
            document: Documento default del challenge.

        Returns:
            Un chunk por sección detectada, o fallback al splitter genérico si
            no se detectan secciones.
        """
        sections = self._split_default_document_sections(document.content)
        if not sections:
            return self._chunk_with_splitter(document)

        chunks: list[DocumentChunk] = []

        for section_index, section in enumerate(sections):
            content = section["content"]
            metadata = {
                **document.metadata,
                "document_id": document.id,
                "chunk_index": str(section_index),
                "chunk_strategy": self.config.strategy,
                "section_index": str(section_index),
            }
            if section["title"]:
                metadata["section_title"] = section["title"]

            chunks.append(
                DocumentChunk(
                    id=self._build_chunk_id(document.id, section_index, content),
                    content=content,
                    metadata=self._string_metadata(metadata),
                )
            )

        return tuple(chunks)

    def _split_default_document_sections(
        self,
        content: str,
    ) -> tuple[dict[str, str], ...]:
        """Extrae secciones por párrafos y títulos terminados en dos puntos.

        Args:
            content: Texto completo del documento default.

        Returns:
            Tupla de diccionarios con ``content`` y ``title`` por sección.
        """
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n+", content.strip())
            if paragraph.strip()
        ]

        sections: list[dict[str, str]] = []
        for paragraph in paragraphs:
            title_match = _SECTION_TITLE_PATTERN.match(paragraph)
            sections.append(
                {
                    "content": paragraph,
                    "title": title_match.group("title").strip()
                    if title_match
                    else "",
                }
            )

        return tuple(sections)

    def _build_splitter(self) -> Any:
        """Construye el splitter recursivo usado en documentos genéricos.

        Returns:
            Instancia de ``RecursiveCharacterTextSplitter`` configurada.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            add_start_index=True,
        )

    def _to_source_document(self, document: Document) -> Any:
        """Convierte la entidad Document al formato esperado por LangChain.

        Args:
            document: Documento de dominio.

        Returns:
            Documento compatible con LangChain.
        """
        from langchain_core.documents import Document as SourceDocument

        return SourceDocument(
            page_content=document.content,
            metadata={
                **document.metadata,
                "document_id": document.id,
            },
        )

    def _build_chunk_id(
        self,
        document_id: str,
        chunk_index: int,
        content: str,
    ) -> str:
        """Genera un identificador determinístico para un chunk.

        Args:
            document_id: Identificador del documento fuente.
            chunk_index: Posición del chunk dentro del documento.
            content: Contenido del chunk usado para el hash.

        Returns:
            Identificador estable del chunk.
        """
        content_hash = sha256(content.encode("utf-8")).hexdigest()[:12]
        return f"{document_id}:chunk:{chunk_index}:{content_hash}"

    def _string_metadata(self, metadata: dict[str, Any]) -> dict[str, str]:
        """Convierte metadata a strings para persistencia vectorial.

        Args:
            metadata: Metadata heterogénea generada durante el chunking.

        Returns:
            Metadata con claves y valores convertidos a ``str``.
        """
        return {
            str(key): str(value)
            for key, value in metadata.items()
            if value is not None
        }
