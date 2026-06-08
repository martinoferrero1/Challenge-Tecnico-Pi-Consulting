from functools import lru_cache

from app.application.use_cases.answer_question import AnswerQuestionUseCase
from app.core.config import settings
from app.infrastructure.pipelines.question_answering_pipeline import (
    create_answer_question_use_case,
)


@lru_cache(maxsize=1)
def get_answer_question_use_case() -> AnswerQuestionUseCase:
    """Construye y cachea el caso de uso de preguntas para FastAPI.

    Returns:
        Instancia singleton de ``AnswerQuestionUseCase`` configurada con
        infraestructura real.
    """
    return create_answer_question_use_case(settings)
