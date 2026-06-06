from typing import Any


class OpenAILLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        client: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.client = client

    async def generate(self, prompt: str) -> str:
        response = await self._get_client().responses.create(
            model=self.model,
            input=prompt,
        )

        return str(response.output_text).strip()

    def _get_client(self) -> Any:
        if self.client is None:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(api_key=self.api_key)

        return self.client
