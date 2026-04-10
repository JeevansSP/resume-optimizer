import asyncio
from logging import getLogger
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.exceptions import ProviderAuthError, ProviderRateLimitError, ProviderModelError, ProviderError
from app.services.ai.providers.base import LLMProvider, ModelInfo

logger = getLogger(__name__)


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model_id: str):
        self.api_key = api_key
        self.model_id = model_id
        self.client = genai.Client(api_key=api_key)

    async def list_models(self) -> list[ModelInfo]:
        try:
            raw = await asyncio.to_thread(self.client.models.list)
            models = []
            for m in raw:
                methods = getattr(m, "supported_generation_methods", [])
                if "generateContent" not in methods:
                    continue
                name = getattr(m, "name", "")
                model_id = name.replace("models/", "") if name.startswith("models/") else name
                display = getattr(m, "display_name", model_id)
                models.append(ModelInfo(id=model_id, name=display, supports_structured_output=True))
            return models
        except Exception as e:
            raise ProviderError(f"Failed to list Gemini models: {e}") from e

    async def generate(
        self,
        system_prompt: str,
        inputs: list[str | dict[str, Any]],
        *,
        structured_output_schema: type[BaseModel] | None = None,
        temperature: float = 0.1,
        timeout: int | None = None,
    ) -> str:
        config_params: dict[str, Any] = {
            "system_instruction": system_prompt,
            "temperature": temperature,
            "thinking_config": types.ThinkingConfig(thinking_level="LOW"),
        }

        if structured_output_schema:
            config_params["response_mime_type"] = "application/json"
            if isinstance(structured_output_schema, type) and issubclass(
                structured_output_schema, BaseModel
            ):
                config_params["response_schema"] = structured_output_schema

        try:
            coro = self.client.aio.models.generate_content(
                model=self.model_id,
                config=types.GenerateContentConfig(**config_params),
                contents=inputs,
            )
            if timeout:
                response = await asyncio.wait_for(coro, timeout=timeout)
            else:
                response = await coro

            return response.text.strip()

        except asyncio.TimeoutError:
            raise
        except Exception as e:
            error_str = str(e).lower()
            if "api key" in error_str or "unauthorized" in error_str or "403" in error_str:
                raise ProviderAuthError(f"Gemini authentication failed: {e}") from e
            if "429" in error_str or "resource exhausted" in error_str:
                raise ProviderRateLimitError(f"Gemini rate limit: {e}") from e
            if "not found" in error_str and "model" in error_str:
                raise ProviderModelError(f"Gemini model not found: {e}") from e
            raise ProviderError(f"Gemini error: {e}") from e
