from app.domain.entities.answer import Answer
from app.domain.entities.cache_key import AnswerCacheKey


class InMemoryAnswerCache:
    """Cache de respuestas en memoria para la vida del proceso."""

    def __init__(self) -> None:
        """Inicializa ``_answers`` como almacenamiento interno de respuestas."""
        self._answers: dict[AnswerCacheKey, Answer] = {}

    async def get(self, key: AnswerCacheKey) -> Answer | None:
        """Obtiene una respuesta cacheada por clave.

        Args:
            key: Clave exacta de cache.

        Returns:
            Respuesta guardada para ``key`` o ``None``.
        """
        return self._answers.get(key)

    async def set(self, key: AnswerCacheKey, answer: Answer) -> None:
        """Guarda una respuesta cacheada.

        Args:
            key: Clave bajo la que se guarda ``answer``.
            answer: Respuesta que se quiere cachear.
        """
        self._answers[key] = answer

    async def list_by_question(self, question: str) -> list[Answer]:
        """Lista respuestas cacheadas para una pregunta normalizada.

        Args:
            question: Pregunta que se normaliza antes de buscar candidatos.

        Returns:
            Respuestas cuyo ``AnswerCacheKey.question`` coincide con
            ``question`` normalizada.
        """
        normalized_question = " ".join(question.strip().split()).casefold()

        return [
            answer
            for key, answer in self._answers.items()
            if key.question == normalized_question
        ]

    def clear(self) -> None:
        """Limpia todo el cache en memoria."""
        self._answers.clear()
