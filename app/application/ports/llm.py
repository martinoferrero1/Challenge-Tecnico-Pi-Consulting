from typing import Protocol


class LLMPort(Protocol):
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError
