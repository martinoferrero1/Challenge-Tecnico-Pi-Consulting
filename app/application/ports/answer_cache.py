from typing import Protocol

from app.domain.entities.answer import Answer
from app.domain.entities.cache_key import AnswerCacheKey


class AnswerCachePort(Protocol):
    """Contrato de cache de respuestas usado por la capa de aplicación."""

    async def get(self, key: AnswerCacheKey) -> Answer | None:
        """Obtiene una respuesta cacheada por clave exacta.

        Args:
            key: Clave compuesta por pregunta normalizada y hash de contexto.

        Returns:
            Respuesta cacheada si existe; ``None`` en caso contrario.
        """
        raise NotImplementedError

    async def set(self, key: AnswerCacheKey, answer: Answer) -> None:
        """Guarda una respuesta bajo una clave de cache.

        Args:
            key: Clave bajo la cual se guarda ``answer``.
            answer: Respuesta de dominio que se quiere reutilizar.
        """
        raise NotImplementedError

    async def list_by_question(self, question: str) -> list[Answer]:
        """Lista respuestas cacheadas por contenido normalizado de pregunta.

        Args:
            question: Pregunta normalizada usada para buscar candidatos.

        Returns:
            Lista de respuestas asociadas a ``question``.
        """
        raise NotImplementedError
