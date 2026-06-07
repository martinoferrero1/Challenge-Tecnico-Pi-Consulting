from typing import Protocol, TypeVar

from pydantic import BaseModel


StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


class LLMPort(Protocol):
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError

    async def generate_structured(
        self,
        prompt: str,
        output_schema: type[StructuredOutput],
    ) -> StructuredOutput:
        raise NotImplementedError
