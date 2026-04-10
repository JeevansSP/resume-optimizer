import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.ai.providers.base import ModelInfo


@pytest.mark.asyncio
async def test_gemini_provider_generate():
    with patch("app.services.ai.providers.gemini.genai") as mock_genai:
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = '{"name": "John"}'
        mock_response.usage_metadata = None
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        from app.services.ai.providers.gemini import GeminiProvider

        provider = GeminiProvider(api_key="test-key", model_id="gemini-3-flash-preview")
        result = await provider.generate(
            system_prompt="Extract info",
            inputs=["some text"],
        )
        assert result == '{"name": "John"}'
        mock_client.aio.models.generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_gemini_provider_list_models():
    with patch("app.services.ai.providers.gemini.genai") as mock_genai:
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_model = MagicMock()
        mock_model.name = "models/gemini-3-flash-preview"
        mock_model.display_name = "Gemini 3 Flash Preview"
        mock_model.supported_generation_methods = ["generateContent"]
        mock_client.models.list.return_value = [mock_model]

        from app.services.ai.providers.gemini import GeminiProvider

        provider = GeminiProvider(api_key="test-key", model_id="gemini-3-flash-preview")
        models = await provider.list_models()
        assert len(models) >= 1
        assert isinstance(models[0], ModelInfo)


@pytest.mark.asyncio
async def test_openai_provider_generate():
    with patch("app.services.ai.providers.openai_provider.openai") as mock_openai:
        mock_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = '{"name": "Jane"}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        from app.services.ai.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="sk-test", model_id="gpt-4o")
        result = await provider.generate(
            system_prompt="Extract info",
            inputs=["some text"],
        )
        assert result == '{"name": "Jane"}'


@pytest.mark.asyncio
async def test_openai_provider_list_models():
    with patch("app.services.ai.providers.openai_provider.openai") as mock_openai:
        mock_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        mock_model = MagicMock()
        mock_model.id = "gpt-4o"
        mock_list = MagicMock()
        mock_list.data = [mock_model]
        mock_client.models.list = AsyncMock(return_value=mock_list)

        from app.services.ai.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="sk-test", model_id="gpt-4o")
        models = await provider.list_models()
        assert len(models) >= 1
        assert models[0].id == "gpt-4o"
