import json
from logging import getLogger
from typing import Any

import openai
from pydantic import BaseModel

from app.exceptions import ProviderAuthError, ProviderRateLimitError, ProviderError
from app.services.ai.providers.base import LLMProvider, ModelInfo

logger = getLogger(__name__)


def _schema_to_json_instruction(schema: type[BaseModel]) -> str:
    return (
        "\n\nYou MUST respond with valid JSON matching this schema exactly:\n"
        f"```json\n{json.dumps(schema.model_json_schema(), indent=2)}\n```\n"
        "Respond ONLY with the JSON object, no other text."
    )


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, api_key: str, model_id: str, base_url: str):
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = base_url
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def list_models(self) -> list[ModelInfo]:
        try:
            result = await self.client.models.list()
            return sorted(
                [ModelInfo(id=m.id, name=m.id, supports_structured_output=False) for m in result.data],
                key=lambda x: x.id,
            )
        except Exception as e:
            logger.warning(f"Failed to list models from {self.base_url}: {e}")
            return []

    async def generate(
        self,
        system_prompt: str,
        inputs: list[str | dict[str, Any]],
        *,
        structured_output_schema: type[BaseModel] | None = None,
        temperature: float = 0.1,
        timeout: int | None = None,
    ) -> str:
        effective_prompt = system_prompt
        if structured_output_schema:
            effective_prompt += _schema_to_json_instruction(structured_output_schema)

        user_parts = []
        for inp in inputs:
            if isinstance(inp, str):
                user_parts.append(inp)
            elif isinstance(inp, dict):
                user_parts.append("[Image content not supported for this provider]")

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": effective_prompt},
                {"role": "user", "content": "\n".join(user_parts)},
            ],
            "temperature": temperature,
        }
        if timeout:
            kwargs["timeout"] = timeout

        try:
            response = await self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except openai.AuthenticationError as e:
            raise ProviderAuthError(f"Authentication failed: {e}") from e
        except openai.RateLimitError as e:
            raise ProviderRateLimitError(f"Rate limit: {e}") from e
        except Exception as e:
            raise ProviderError(f"Provider error: {e}") from e
