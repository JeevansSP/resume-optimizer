import asyncio
import json
import time
from logging import getLogger
from typing import Type
from pydantic import BaseModel, ValidationError
from app.config import get_settings

# If primary model exceeds this, immediately fall back to FALLBACK_MODEL.
PRIMARY_TIMEOUT_SECONDS = 60
FALLBACK_MODEL = "gemini-2.5-pro"

logger = getLogger(__name__)


async def _log_request(
    model_name: str,
    user_id: str | None,
    purpose: str | None,
    reference_id: str | None,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cached_tokens: int,
    response_time_ms: int,
    success: bool,
    error_message: str | None,
) -> None:
    """Persist an LLMRequest row using an independent DB session.

    Wrapped in try/except so a DB failure never breaks the main flow.
    """
    if not purpose:
        return
    try:
        from app.database.session import async_session_factory
        from app.models.token_usage import LLMRequest

        async with async_session_factory() as db:
            row = LLMRequest(
                user_id=user_id,
                purpose=purpose,
                reference_id=reference_id,
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                response_time_ms=response_time_ms,
                success=success,
                error_message=error_message,
            )
            db.add(row)
            await db.commit()
    except Exception:
        logger.debug("Failed to persist LLMRequest row", exc_info=True)


class InferenceService:
    """Orchestrates LLM inference with retry, schema validation, and logging.

    When constructed with a provider, delegates generate() to it.
    When constructed without one (or with just a model_name), falls back to
    the legacy GeminiProvider path for backward compatibility.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        provider: "LLMProvider | None" = None,
        allow_fallback: bool = False,
    ):
        from app.services.ai.providers.base import LLMProvider as _LLM

        self.allow_fallback = allow_fallback

        if provider:
            self.provider: _LLM = provider
            self.model = getattr(provider, "model_id", model_name or "unknown")
        else:
            # Legacy path: construct a GeminiProvider from settings
            self.allow_fallback = True  # platform default always gets fallback
            settings = get_settings()
            model = model_name or settings.GEMINI_FLASH_MODEL
            from app.services.ai.providers.gemini import GeminiProvider
            self.provider = GeminiProvider(api_key=settings.GEMINI_API_KEY, model_id=model)
            self.model = model

    def parse_output(
        self,
        raw_content: str,
        structured_output_schema: Type[BaseModel] | None,
        is_list: bool = False,
    ) -> dict | list[dict] | str:
        json_str = raw_content.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        elif json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()

        if not structured_output_schema:
            return json_str

        parsed_content = json.loads(json_str)
        if is_list:
            if isinstance(parsed_content, dict):
                parsed_content = [parsed_content]
            return [
                structured_output_schema.model_validate(o).model_dump()
                for o in parsed_content
            ]
        return structured_output_schema.model_validate(parsed_content).model_dump()

    async def run_inference(
        self,
        system_prompt: str,
        inputs: list | None = None,
        structured_output_schema: Type[BaseModel] | None = None,
        is_structured_output_list: bool = False,
        temperature: float = 0.1,
        *,
        user_id: str | None = None,
        purpose: str | None = None,
        reference_id: str | None = None,
        thinking_level: str = "LOW",
        fallback_model: str | None = FALLBACK_MODEL,
        primary_timeout: int | None = PRIMARY_TIMEOUT_SECONDS,
    ) -> str | dict | list:
        call_kwargs = dict(
            system_prompt=system_prompt,
            inputs=inputs or [],
            structured_output_schema=structured_output_schema,
            temperature=temperature,
            timeout=primary_timeout,
        )

        t0 = time.monotonic()

        async def _generate_with_fallback() -> str:
            can_fallback = (
                self.allow_fallback
                and fallback_model
                and fallback_model != self.model
            )
            try:
                return await self.provider.generate(**call_kwargs)
            except asyncio.TimeoutError:
                if not can_fallback:
                    raise
                logger.warning(
                    f"Primary model {self.model} timed out, falling back to {fallback_model}"
                )
                settings = get_settings()
                from app.services.ai.providers.gemini import GeminiProvider
                fb_provider = GeminiProvider(api_key=settings.GEMINI_API_KEY, model_id=fallback_model)
                fb_kwargs = {**call_kwargs, "timeout": None}
                return await fb_provider.generate(**fb_kwargs)
            except Exception:
                if not can_fallback:
                    raise
                logger.warning(
                    f"Primary model {self.model} failed, falling back to {fallback_model}"
                )
                settings = get_settings()
                from app.services.ai.providers.gemini import GeminiProvider
                fb_provider = GeminiProvider(api_key=settings.GEMINI_API_KEY, model_id=fallback_model)
                fb_kwargs = {**call_kwargs, "timeout": None}
                return await fb_provider.generate(**fb_kwargs)

        max_validation_retries = 2
        for attempt in range(1 + max_validation_retries):
            try:
                response_str = await _generate_with_fallback()
            except Exception as exc:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                await _log_request(
                    model_name=self.model, user_id=user_id, purpose=purpose,
                    reference_id=reference_id, input_tokens=0, output_tokens=0,
                    total_tokens=0, cached_tokens=0, response_time_ms=elapsed_ms,
                    success=False, error_message=str(exc)[:500],
                )
                raise

            elapsed_ms = int((time.monotonic() - t0) * 1000)
            await _log_request(
                model_name=self.model, user_id=user_id, purpose=purpose,
                reference_id=reference_id, input_tokens=0, output_tokens=0,
                total_tokens=0, cached_tokens=0, response_time_ms=elapsed_ms,
                success=True, error_message=None,
            )

            if not structured_output_schema:
                return response_str

            try:
                return self.parse_output(
                    response_str, structured_output_schema, is_structured_output_list
                )
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt >= max_validation_retries:
                    logger.error(
                        f"Schema validation failed after {attempt + 1} attempts: {e}"
                    )
                    raise
                logger.warning(
                    f"Schema validation failed (attempt {attempt + 1}), retrying: {e}"
                )


# Backward compatibility alias
GeminiInference = InferenceService
