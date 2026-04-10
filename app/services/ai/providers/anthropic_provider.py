import json
from logging import getLogger
from typing import Any

import anthropic
from pydantic import BaseModel

from app.exceptions import ProviderAuthError, ProviderRateLimitError, ProviderModelError, ProviderError
from app.services.ai.providers.base import LLMProvider, ModelInfo

logger = getLogger(__name__)

ANTHROPIC_MODELS = [
    ModelInfo(id="claude-opus-4-6", name="Claude Opus 4.6", supports_structured_output=False),
    ModelInfo(id="claude-sonnet-4-6", name="Claude Sonnet 4.6", supports_structured_output=False),
    ModelInfo(id="claude-haiku-4-5", name="Claude Haiku 4.5", supports_structured_output=False),
]


def _schema_to_json_instruction(schema: type[BaseModel]) -> str:
    return (
        "\n\nYou MUST respond with valid JSON matching this schema exactly:\n"
        f"```json\n{json.dumps(schema.model_json_schema(), indent=2)}\n```\n"
        "Respond ONLY with the JSON object, no other text."
    )


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model_id: str):
        self.api_key = api_key
        self.model_id = model_id
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def list_models(self) -> list[ModelInfo]:
        return list(ANTHROPIC_MODELS)

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
            "max_tokens": 8192,
            "system": effective_prompt,
            "messages": [{"role": "user", "content": "\n".join(user_parts)}],
            "temperature": temperature,
        }
        if timeout:
            kwargs["timeout"] = timeout

        try:
            response = await self.client.messages.create(**kwargs)
            return response.content[0].text.strip()
        except anthropic.AuthenticationError as e:
            raise ProviderAuthError(f"Anthropic authentication failed: {e}") from e
        except anthropic.RateLimitError as e:
            raise ProviderRateLimitError(f"Anthropic rate limit: {e}") from e
        except anthropic.NotFoundError as e:
            raise ProviderModelError(f"Anthropic model not found: {e}") from e
        except Exception as e:
            raise ProviderError(f"Anthropic error: {e}") from e
