from functools import lru_cache

from app.application.use_cases.answer_question import AnswerQuestionUseCase
from app.core.config import settings
from app.infrastructure.pipelines.question_answering_pipeline import (
    create_answer_question_use_case,
)


@lru_cache(maxsize=1)
def get_answer_question_use_case() -> AnswerQuestionUseCase: # lo implemento así para desacoplar la creación del caso de uso de la lógica de negocio (y desde la api no interactuo con infraestructura), y para aprovechar el caching que ofrece lru_cache y no crear múltiples instancias del caso de uso
    return create_answer_question_use_case(settings)
