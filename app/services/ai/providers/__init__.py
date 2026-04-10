from app.services.ai.providers.base import LLMProvider, ModelInfo
from app.config import get_settings

# Purposes that should use the pro model when no user config is set
_PRO_PURPOSES = {"resume_tailoring"}


def get_provider(config, purpose: str = "") -> LLMProvider:
    """Build the right LLMProvider based on user AI config.

    Args:
        config: UserAIConfig ORM instance, or None for platform default.
        purpose: e.g. 'profile_structuring', 'resume_tailoring' — used to
                 select flash vs pro when no user config exists.
    """
    settings = get_settings()

    # No config → platform default with per-purpose model selection
    if config is None:
        from app.services.ai.providers.gemini import GeminiProvider

        model = settings.GEMINI_PRO_MODEL if purpose in _PRO_PURPOSES else settings.GEMINI_FLASH_MODEL
        return GeminiProvider(api_key=settings.GEMINI_API_KEY, model_id=model)

    provider_name = config.provider

    if provider_name == "PLATFORM_GEMINI":
        from app.services.ai.providers.gemini import GeminiProvider
        return GeminiProvider(api_key=settings.GEMINI_API_KEY, model_id=config.model_id)

    # All other providers require a user-supplied API key
    api_key = config.decrypted_api_key

    if provider_name == "GEMINI":
        from app.services.ai.providers.gemini import GeminiProvider
        return GeminiProvider(api_key=api_key, model_id=config.model_id)

    if provider_name == "OPENAI":
        from app.services.ai.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(api_key=api_key, model_id=config.model_id)

    if provider_name == "ANTHROPIC":
        from app.services.ai.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key=api_key, model_id=config.model_id)

    if provider_name == "CUSTOM_OPENAI_COMPATIBLE":
        from app.services.ai.providers.openai_compatible import OpenAICompatibleProvider
        return OpenAICompatibleProvider(
            api_key=api_key, model_id=config.model_id, base_url=config.api_host,
        )

    raise ValueError(f"Unknown provider: {provider_name}")


__all__ = ["LLMProvider", "ModelInfo", "get_provider"]
