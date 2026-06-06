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
        client = self._get_client()

        if hasattr(client, "ainvoke"):
            response = await client.ainvoke(prompt)
        else:
            response = await asyncio.to_thread(client.invoke, prompt)

        return self._content_to_text(response.content).strip()

    def _get_client(self) -> Any:
        if self.client is None:
            from langchain_cohere import ChatCohere

            self.client = ChatCohere(
                cohere_api_key=self.api_key,
                model=self.model,
                temperature=0,
            )

        return self.client

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(item.get("text", item)) if isinstance(item, dict) else str(item)
                for item in content
            )

        return str(content)
