import asyncio
from typing import Any, TypeVar

from pydantic import BaseModel

from app.infrastructure.llms.structured_output import invoke_structured


StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


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
        client = self._get_client()

        if hasattr(client, "ainvoke"):
            response = await client.ainvoke(prompt)
        else:
            response = await asyncio.to_thread(client.invoke, prompt)

        return self._content_to_text(response.content).strip()

    async def generate_structured(
        self,
        prompt: str,
        output_schema: type[StructuredOutput],
    ) -> StructuredOutput:
        return await invoke_structured(
            client=self._get_client(),
            prompt=prompt,
            output_schema=output_schema,
        )

    def _get_client(self) -> Any:
        if self.client is None:
            from langchain_openai import ChatOpenAI

            self.client = ChatOpenAI(
                api_key=self.api_key,
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
