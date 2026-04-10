import pytest
from unittest.mock import patch, AsyncMock
from app.models.ai_config import UserAIConfig, AIProvider


@pytest.mark.asyncio
async def test_get_ai_config_empty(client):
    resp = await client.get("/settings/ai/")
    assert resp.status_code == 200
    assert resp.json() is None


@pytest.mark.asyncio
async def test_save_platform_gemini(client, db_session, test_user):
    with patch("app.api.ai_settings._fetch_models_for_provider", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = []
        resp = await client.put("/settings/ai/", json={
            "provider": "PLATFORM_GEMINI",
            "model_id": "gemini-3-flash-preview",
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "PLATFORM_GEMINI"
    assert data["model_id"] == "gemini-3-flash-preview"
    assert data["key_configured"] is False


@pytest.mark.asyncio
async def test_save_openai_requires_api_key(client):
    resp = await client.put("/settings/ai/", json={
        "provider": "OPENAI",
        "model_id": "gpt-4o",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_delete_ai_config(client, db_session, test_user):
    config = UserAIConfig(
        user_id=test_user.id,
        provider=AIProvider.PLATFORM_GEMINI,
        model_id="gemini-3-flash-preview",
    )
    db_session.add(config)
    await db_session.commit()

    resp = await client.delete("/settings/ai/")
    assert resp.status_code == 200
    assert resp.json()["detail"] == "AI config deleted"


@pytest.mark.asyncio
async def test_fetch_models(client):
    with patch("app.api.ai_settings._fetch_models_for_provider", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = [
            {"id": "gpt-4o", "name": "GPT-4o", "supports_structured_output": True},
        ]
        resp = await client.post("/settings/ai/models", json={
            "provider": "OPENAI",
            "api_key": "sk-test",
        })
    assert resp.status_code == 200
    assert len(resp.json()) >= 1
