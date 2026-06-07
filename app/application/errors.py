from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalServiceError(Exception):
    cause: str
    message: str = "The question could not be processed."

    @property
    def code(self) -> str:
        return "question_processing_error"

    def to_detail(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "cause": self.cause,
        }
