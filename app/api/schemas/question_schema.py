from pydantic import BaseModel, Field, field_validator


class QuestionRequest(BaseModel):
    """Payload público para enviar una pregunta a la API.

    Atributos:
        user_name: Identificador del usuario o sesión conversacional.
        question: Pregunta actual que se quiere responder con el documento.
    """

    user_name: str = Field(..., min_length=1, examples=["John Doe"])
    question: str = Field(..., min_length=1, examples=["Quien es Zara?"])

    @field_validator("user_name", "question")
    @classmethod
    def validate_not_blank(cls, value: str) -> str:
        """Normaliza campos de texto y rechaza valores vacíos.

        Args:
            value: Valor crudo de ``user_name`` o ``question``.

        Returns:
            El valor sin espacios externos.

        Raises:
            ValueError: Si ``value`` queda vacío luego de hacer ``strip``.
        """
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be empty")

        return value


class RetrievedChunkResponse(BaseModel):
    """Chunk recuperado expuesto por el endpoint de evaluación.

    Atributos:
        id: Identificador del chunk recuperado.
        content: Texto del chunk recuperado.
        similarity_score: Score de similitud reportado por el vector store.
        metadata: Metadata serializable asociada al chunk.
    """

    id: str
    content: str
    similarity_score: float
    metadata: dict[str, str] = Field(default_factory=dict)


class AnswerDiagnosticsResponse(BaseModel):
    """Diagnóstico interno usado por el script de métricas RAG.

    Atributos:
        conversation_context_mode: Modo de contexto conversacional activo.
        answer_cache_mode: Modo de cache activo.
        cache_hit: Indica si la respuesta vino desde cache.
        cache_hit_source: Fuente del cache hit cuando existe.
        resolved_query: Pregunta reescrita cuando el rewrite produjo una query
            distinta.
        stage_latencies_ms: Latencias por etapa en milisegundos.
        retrieved_chunks: Chunks recuperados o reutilizados para la respuesta.
    """

    conversation_context_mode: str
    answer_cache_mode: str
    cache_hit: bool
    cache_hit_source: str | None = None
    resolved_query: str | None = None
    stage_latencies_ms: dict[str, float] = Field(default_factory=dict)
    retrieved_chunks: list[RetrievedChunkResponse] = Field(default_factory=list)


class AnswerResponse(BaseModel):
    """Respuesta pública del endpoint de preguntas.

    Atributos:
        user_name: Usuario o sesión que realizó la pregunta.
        question: Pregunta normalizada recibida por la API.
        answer: Respuesta final del asistente.
    """

    user_name: str
    question: str
    answer: str


class AnswerEvaluationResponse(AnswerResponse):
    """Respuesta extendida con diagnóstico para evaluaciones offline.

    Atributos:
        diagnostics: Datos internos usados por scripts de evaluación y métricas.
    """

    diagnostics: AnswerDiagnosticsResponse
