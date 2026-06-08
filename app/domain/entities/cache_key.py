from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerCacheKey:
    """Clave normalizada para buscar o guardar respuestas cacheadas.

    Atributos:
        question: Pregunta normalizada usada como primer componente de cache.
        context_hash: Hash que representa el contexto relevante según el modo de
            cache configurado.
    """

    question: str
    context_hash: str

    def __post_init__(self) -> None:
        """Valida que ``question`` y ``context_hash`` no sean vacíos.

        Raises:
            ValueError: Si ``question`` o ``context_hash`` quedan vacíos.
        """
        if not self.question.strip():
            raise ValueError("The question of the cache key cannot be empty")
        if not self.context_hash.strip():
            raise ValueError("The context hash of the cache key cannot be empty")
