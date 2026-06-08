import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

from app.domain.entities.answer import Answer
from app.domain.entities.answer import AnswerDiagnostics


TraceResult = TypeVar("TraceResult")


@dataclass
class RagTrace:
    """Acumula tiempos y arma diagnostics para una respuesta RAG.

    Atributos:
        conversation_context_mode: Modo de contexto conversacional usado.
        answer_cache_mode: Modo de cache usado.
        started_at: Momento inicial de la request según ``time.perf_counter``.
        stage_latencies_ms: Latencias acumuladas por nombre de etapa.
    """

    conversation_context_mode: str
    answer_cache_mode: str
    started_at: float = field(default_factory=time.perf_counter)
    stage_latencies_ms: dict[str, float] = field(default_factory=dict)

    @classmethod
    def start(
        cls,
        conversation_context_mode: str,
        answer_cache_mode: str,
    ) -> "RagTrace":
        """Crea un trace para una ejecución del caso de uso.

        Args:
            conversation_context_mode: Modo de contexto conversacional activo.
            answer_cache_mode: Modo de cache de respuestas activo.

        Returns:
            Nueva instancia de ``RagTrace``.
        """
        return cls(
            conversation_context_mode=conversation_context_mode,
            answer_cache_mode=answer_cache_mode,
        )

    async def measure_async(
        self,
        stage: str,
        call: Callable[[], Awaitable[TraceResult]],
    ) -> TraceResult:
        """Mide una operación asincrónica y acumula su latencia.

        Args:
            stage: Nombre de la etapa a medir.
            call: Callable asincrónico que ejecuta la operación real.

        Returns:
            Resultado devuelto por ``call``.
        """
        started_at = time.perf_counter()
        try:
            return await call()
        finally:
            self.add_stage(stage, (time.perf_counter() - started_at) * 1000)

    def measure_sync(
        self,
        stage: str,
        call: Callable[[], TraceResult],
    ) -> TraceResult:
        """Mide una operación sincrónica y acumula su latencia.

        Args:
            stage: Nombre de la etapa a medir.
            call: Callable sincrónico que ejecuta la operación real.

        Returns:
            Resultado devuelto por ``call``.
        """
        started_at = time.perf_counter()
        try:
            return call()
        finally:
            self.add_stage(stage, (time.perf_counter() - started_at) * 1000)

    async def finish_answer(
        self,
        answer: Answer,
        remember_answer: Callable[[Answer], Awaitable[Answer]],
        cache_hit: bool = False,
        cache_hit_source: str | None = None,
    ) -> Answer:
        """Guarda conversación y devuelve la respuesta con diagnostics.

        Args:
            answer: Respuesta de dominio antes de adjuntar diagnostics.
            remember_answer: Callable que persiste conversación y devuelve
                ``answer``.
            cache_hit: Indica si ``answer`` fue reutilizada desde cache.
            cache_hit_source: Fuente del cache hit cuando existe.

        Returns:
            Nueva respuesta con ``diagnostics`` cargado.
        """
        remembered_answer = await self.measure_async(
            "conversation_save",
            lambda: remember_answer(answer),
        )

        return self.with_diagnostics(
            answer=remembered_answer,
            cache_hit=cache_hit,
            cache_hit_source=cache_hit_source,
        )

    def with_diagnostics(
        self,
        answer: Answer,
        cache_hit: bool = False,
        cache_hit_source: str | None = None,
    ) -> Answer:
        """Adjunta el diagnóstico acumulado a una respuesta.

        Args:
            answer: Respuesta de dominio a enriquecer.
            cache_hit: Indica si la respuesta fue servida desde cache.
            cache_hit_source: Fuente del cache hit cuando aplica.

        Returns:
            Nueva ``Answer`` con ``AnswerDiagnostics``.
        """
        self.add_stage("total", (time.perf_counter() - self.started_at) * 1000)

        return Answer(
            question=answer.question,
            content=answer.content,
            context=answer.context,
            resolved_query=answer.resolved_query,
            diagnostics=AnswerDiagnostics(
                conversation_context_mode=self.conversation_context_mode,
                answer_cache_mode=self.answer_cache_mode,
                cache_hit=cache_hit,
                cache_hit_source=cache_hit_source,
                stage_latencies_ms=dict(self.stage_latencies_ms),
            ),
        )

    def add_stage(self, stage: str, elapsed_ms: float) -> None:
        """Acumula una latencia medida para una etapa.

        Args:
            stage: Nombre de la etapa medida.
            elapsed_ms: Latencia de la etapa en milisegundos.
        """
        current_elapsed_ms = self.stage_latencies_ms.get(stage, 0.0)
        self.stage_latencies_ms[stage] = round(current_elapsed_ms + elapsed_ms, 2)
