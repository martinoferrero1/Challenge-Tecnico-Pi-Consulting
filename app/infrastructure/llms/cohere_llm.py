import asyncio
from typing import Any


class CohereLLM:
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
            self._get_client().chat,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        return self._extract_text(response).strip()

    def _get_client(self) -> Any:
        if self.client is None:
            import cohere

            self.client = cohere.ClientV2(api_key=self.api_key)

        return self.client

    def _extract_text(self, response: Any) -> str:
        message = getattr(response, "message", None)
        content = getattr(message, "content", None)

        if content and hasattr(content[0], "text"):
            return str(content[0].text)
        if content and isinstance(content[0], dict):
            return str(content[0].get("text", ""))

        raise ValueError("Cohere response does not contain text output")
