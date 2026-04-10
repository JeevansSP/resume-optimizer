from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.dependencies import get_current_user
from app.models.ai_config import UserAIConfig, AIProvider
from app.models.user import User
from app.schemas.ai_config import AIConfigResponse, AIConfigUpdate, FetchModelsRequest, ModelInfoResponse
from app.services.crypto import encrypt
from app.config import get_settings

router = APIRouter(prefix="/settings/ai", tags=["ai-settings"])


def _config_to_response(config: UserAIConfig) -> AIConfigResponse:
    key_hint = None
    if config.api_key_encrypted:
        try:
            decrypted = config.decrypted_api_key
            key_hint = f"••••{decrypted[-4:]}" if decrypted and len(decrypted) >= 4 else "••••"
        except Exception:
            key_hint = "••••"
    return AIConfigResponse(
        provider=config.provider,
        model_id=config.model_id,
        api_host=config.api_host,
        key_configured=config.api_key_encrypted is not None,
        key_hint=key_hint,
    )


async def _fetch_models_for_provider(
    provider: AIProvider,
    api_key: str | None = None,
    api_host: str | None = None,
) -> list[dict]:
    settings = get_settings()

    if provider in (AIProvider.PLATFORM_GEMINI, AIProvider.GEMINI):
        from app.services.ai.providers.gemini import GeminiProvider
        key = settings.GEMINI_API_KEY if provider == AIProvider.PLATFORM_GEMINI else api_key
        p = GeminiProvider(api_key=key, model_id="")
    elif provider == AIProvider.OPENAI:
        from app.services.ai.providers.openai_provider import OpenAIProvider
        p = OpenAIProvider(api_key=api_key, model_id="")
    elif provider == AIProvider.ANTHROPIC:
        from app.services.ai.providers.anthropic_provider import AnthropicProvider
        p = AnthropicProvider(api_key=api_key or "", model_id="")
    elif provider == AIProvider.CUSTOM_OPENAI_COMPATIBLE:
        from app.services.ai.providers.openai_compatible import OpenAICompatibleProvider
        p = OpenAICompatibleProvider(api_key=api_key or "", model_id="", base_url=api_host or "")
    else:
        return []

    models = await p.list_models()
    return [{"id": m.id, "name": m.name, "supports_structured_output": m.supports_structured_output} for m in models]


@router.get("/")
async def get_ai_config(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AIConfigResponse | None:
    result = await db.execute(
        select(UserAIConfig).where(UserAIConfig.user_id == current_user.id)
    )
    config = result.scalar_one_or_none()
    if not config:
        return None
    return _config_to_response(config)


@router.put("/")
async def save_ai_config(
    payload: AIConfigUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AIConfigResponse:
    result = await db.execute(
        select(UserAIConfig).where(UserAIConfig.user_id == current_user.id)
    )
    config = result.scalar_one_or_none()

    encrypted_key = None
    if payload.api_key and payload.provider != AIProvider.PLATFORM_GEMINI:
        encrypted_key = encrypt(payload.api_key)

    if config:
        config.provider = payload.provider
        config.model_id = payload.model_id
        config.api_host = payload.api_host
        if encrypted_key is not None:
            config.api_key_encrypted = encrypted_key
        elif payload.provider == AIProvider.PLATFORM_GEMINI:
            config.api_key_encrypted = None
    else:
        config = UserAIConfig(
            user_id=current_user.id,
            provider=payload.provider,
            api_key_encrypted=encrypted_key,
            api_host=payload.api_host,
            model_id=payload.model_id,
        )
        db.add(config)

    await db.commit()
    await db.refresh(config)
    return _config_to_response(config)


@router.delete("/")
async def delete_ai_config(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(UserAIConfig).where(UserAIConfig.user_id == current_user.id)
    )
    config = result.scalar_one_or_none()
    if config:
        await db.delete(config)
        await db.commit()
    return {"detail": "AI config deleted"}


@router.post("/models")
async def fetch_models(
    payload: FetchModelsRequest,
    current_user: User = Depends(get_current_user),
) -> list[ModelInfoResponse]:
    models = await _fetch_models_for_provider(
        provider=payload.provider,
        api_key=payload.api_key,
        api_host=payload.api_host,
    )
    return [ModelInfoResponse(**m) for m in models]
