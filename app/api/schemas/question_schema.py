from pydantic import BaseModel, Field, field_validator


class QuestionRequest(BaseModel):
    user_name: str = Field(..., min_length=1, examples=["John Doe"])
    question: str = Field(..., min_length=1, examples=["Quien es Zara?"])

    @field_validator("user_name", "question")
    @classmethod
    def validate_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be empty")

        return value


class AnswerResponse(BaseModel):
    user_name: str
    question: str
    answer: str
