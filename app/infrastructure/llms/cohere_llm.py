import asyncio
from typing import Any, TypeVar

from pydantic import BaseModel

from app.infrastructure.llms.structured_output import invoke_structured


StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


class CohereLLM:
    """Adaptador de ChatCohere para el puerto LLM."""

    def __init__(
        self,
        api_key: str,
        model: str,
        client: Any | None = None,
    ) -> None:
        """Guarda configuración y permite inyectar cliente para tests.

        Args:
            api_key: API key de Cohere.
            model: Nombre del modelo de chat.
            client: Cliente compatible con LangChain, opcional para tests.
        """
        self.api_key = api_key
        self.model = model
        self.client = client

    async def generate(self, prompt: str) -> str:
        """Genera texto libre con temperatura determinística.

        Args:
            prompt: Prompt completo enviado al modelo.

        Returns:
            Texto generado por el modelo.
        """
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
        """Genera una salida estructurada validada por Pydantic.

        Args:
            prompt: Prompt completo enviado al modelo.
            output_schema: Schema Pydantic esperado para la salida.

        Returns:
            Instancia validada de ``output_schema``.
        """
        return await invoke_structured(
            client=self._get_client(),
            prompt=prompt,
            output_schema=output_schema,
        )

    def _get_client(self) -> Any:
        """Construye perezosamente el cliente ChatCohere.

        Returns:
            Cliente de chat cacheado en ``self.client``.
        """
        if self.client is None:
            from langchain_cohere import ChatCohere

            self.client = ChatCohere(
                cohere_api_key=self.api_key,
                model=self.model,
                temperature=0,
            )

        return self.client

    def _content_to_text(self, content: Any) -> str:
        """Convierte contenido del provider a texto plano.

        Args:
            content: Contenido crudo devuelto por LangChain/provider.

        Returns:
            Texto plano concatenado.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(item.get("text", item)) if isinstance(item, dict) else str(item)
                for item in content
            )

        return str(content)
