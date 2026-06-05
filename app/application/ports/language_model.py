from typing import Protocol


class LanguageModelPort(Protocol):
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError
