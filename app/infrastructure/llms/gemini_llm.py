import asyncio
from typing import Any


class GeminiLLM:
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
        response = await asyncio.to_thread(
            self._get_client().models.generate_content,
            model=self.model,
            contents=prompt,
        )

        return str(response.text).strip()

    def _get_client(self) -> Any:
        if self.client is None:
            from google import genai

            self.client = genai.Client(api_key=self.api_key)

        return self.client
