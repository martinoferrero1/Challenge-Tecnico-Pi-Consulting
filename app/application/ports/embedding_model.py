from typing import Protocol, Sequence


class EmbeddingModelPort(Protocol):
    """Contrato para generar embeddings de texto."""

    async def embed_text(self, text: str) -> list[float]:
        """Genera un embedding para una consulta o texto individual.

        Args:
            text: Texto que se quiere transformar en vector.

        Returns:
            Vector de floats generado por el proveedor de embeddings.
        """
        raise NotImplementedError

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Genera embeddings para una lista de textos.

        Args:
            texts: Secuencia de textos a vectorizar.

        Returns:
            Lista de embeddings en el mismo orden que ``texts``.
        """
        raise NotImplementedError
