# Multi-Provider AI Configuration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow users to configure their own LLM provider (Google Gemini, OpenAI, Anthropic, or any OpenAI-compatible endpoint) and model, while preserving the platform's default Gemini behavior for unconfigured users.

**Architecture:** Provider Strategy Pattern — an abstract `LLMProvider` base with concrete implementations per provider. A factory function selects the right provider based on the user's stored config. The existing `GeminiInference` class becomes `InferenceService`, which delegates `generate()` calls to the selected provider while keeping orchestration (retry, schema validation, logging) centralized.

**Tech Stack:** FastAPI, SQLAlchemy async, Fernet encryption (cryptography), google-genai SDK, openai SDK, anthropic SDK, Vue 3 (CDN)

---

## File Map

### New files

| File | Responsibility |
|------|---------------|
| `app/services/crypto.py` | Fernet encrypt/decrypt utility |
| `app/services/ai/providers/__init__.py` | Exports `get_provider()` factory |
| `app/services/ai/providers/base.py` | Abstract `LLMProvider`, `ModelInfo` dataclass, provider exceptions |
| `app/services/ai/providers/gemini.py` | `GeminiProvider` — google-genai SDK |
| `app/services/ai/providers/openai_provider.py` | `OpenAIProvider` — openai SDK |
| `app/services/ai/providers/anthropic_provider.py` | `AnthropicProvider` — anthropic SDK |
| `app/services/ai/providers/openai_compatible.py` | `OpenAICompatibleProvider` — openai SDK + custom base_url |
| `app/models/ai_config.py` | `UserAIConfig` SQLAlchemy model |
| `app/schemas/ai_config.py` | Pydantic schemas for AI config API |
| `app/api/ai_settings.py` | `/settings/ai` router |
| `tests/test_services/test_crypto.py` | Crypto utility tests |
| `tests/test_services/test_providers.py` | Provider unit tests |
| `tests/test_models/test_ai_config.py` | AI config model tests |
| `tests/test_api/test_ai_settings.py` | AI settings API tests |

### Modified files

| File | Changes |
|------|---------|
| `pyproject.toml` | Add `openai`, `anthropic` dependencies |
| `app/config.py` | Add `ENCRYPTION_KEY` setting |
| `app/models/__init__.py` | Import `UserAIConfig` |
| `app/exceptions.py` | Add provider exception classes |
| `app/services/ai/inference.py` | Rename class to `InferenceService`, accept provider, delegate `generate()` |
| `app/services/profile/service.py` | Accept optional `UserAIConfig`, pass to `InferenceService` |
| `app/services/job/service.py` | Accept optional `UserAIConfig`, pass to `InferenceService` |
| `app/services/roast/service.py` | Accept optional `UserAIConfig`, pass to `InferenceService` |
| `app/services/ocr/extractor.py` | Accept optional `UserAIConfig`, use text-only path for non-Gemini |
| `app/api/profiles.py` | Query `UserAIConfig`, pass to service |
| `app/api/jobs.py` | Query `UserAIConfig`, pass to service |
| `app/api/roasts.py` | Query `UserAIConfig`, pass to service |
| `app/main.py` | Register `ai_settings` router + provider exception handlers |
| `alembic/env.py` | Import `ai_config` model |
| `frontend/static/js/app.js` | Add AI Settings page, sidebar link, route |

---

### Task 1: Add dependencies and config

**Files:**
- Modify: `pyproject.toml:6-37`
- Modify: `app/config.py:7-44`

- [ ] **Step 1: Add openai and anthropic to pyproject.toml**

In `pyproject.toml`, add after the `google-genai` line (line 16):

```python
    # AI — multi-provider
    "openai>=1.40.0",
    "anthropic>=0.34.0",
```

- [ ] **Step 2: Add ENCRYPTION_KEY to Settings**

In `app/config.py`, add after line 15 (`GEMINI_PRO_MODEL`):

```python
    ENCRYPTION_KEY: str = ""  # Fernet key for encrypting user API keys
```

- [ ] **Step 3: Install dependencies**

Run: `uv sync`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml app/config.py uv.lock
git commit -m "feat: add openai, anthropic deps and ENCRYPTION_KEY config"
```

---

### Task 2: Crypto utility

**Files:**
- Create: `app/services/crypto.py`
- Create: `tests/test_services/test_crypto.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_services/test_crypto.py`:

```python
import pytest
from unittest.mock import patch


@pytest.fixture
def fernet_key():
    """A valid Fernet key for testing."""
    from cryptography.fernet import Fernet
    return Fernet.generate_key().decode()


def test_encrypt_decrypt_roundtrip(fernet_key):
    with patch("app.services.crypto.get_settings") as mock:
        mock.return_value.ENCRYPTION_KEY = fernet_key
        from importlib import reload
        import app.services.crypto as mod
        reload(mod)
        from app.services.crypto import encrypt, decrypt

        plaintext = "sk-test-key-12345"
        encrypted = encrypt(plaintext)
        assert encrypted != plaintext
        assert decrypt(encrypted) == plaintext


def test_encrypt_produces_different_ciphertexts(fernet_key):
    with patch("app.services.crypto.get_settings") as mock:
        mock.return_value.ENCRYPTION_KEY = fernet_key
        from importlib import reload
        import app.services.crypto as mod
        reload(mod)
        from app.services.crypto import encrypt

        a = encrypt("same-key")
        b = encrypt("same-key")
        # Fernet includes a timestamp, so ciphertexts differ
        assert a != b


def test_decrypt_wrong_key_fails(fernet_key):
    from cryptography.fernet import Fernet

    with patch("app.services.crypto.get_settings") as mock:
        mock.return_value.ENCRYPTION_KEY = fernet_key
        from importlib import reload
        import app.services.crypto as mod
        reload(mod)
        from app.services.crypto import encrypt

        encrypted = encrypt("my-secret")

    other_key = Fernet.generate_key().decode()
    with patch("app.services.crypto.get_settings") as mock:
        mock.return_value.ENCRYPTION_KEY = other_key
        from importlib import reload
        import app.services.crypto as mod
        reload(mod)
        from app.services.crypto import decrypt

        with pytest.raises(Exception):
            decrypt(encrypted)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_services/test_crypto.py -v`
Expected: FAIL — module `app.services.crypto` does not exist

- [ ] **Step 3: Write the implementation**

Create `app/services/crypto.py`:

```python
from cryptography.fernet import Fernet
from app.config import get_settings


def _get_fernet() -> Fernet:
    key = get_settings().ENCRYPTION_KEY
    if not key:
        raise RuntimeError(
            "ENCRYPTION_KEY not set. Generate one with: "
            "python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
        )
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt(plaintext: str) -> str:
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    return _get_fernet().decrypt(ciphertext.encode()).decode()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_services/test_crypto.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/crypto.py tests/test_services/test_crypto.py
git commit -m "feat: add Fernet encrypt/decrypt utility"
```

---

### Task 3: Provider exceptions

**Files:**
- Modify: `app/exceptions.py`

- [ ] **Step 1: Add provider exceptions**

Append to end of `app/exceptions.py`:

```python


class ProviderError(Exception):
    """Base exception for LLM provider failures."""
    pass


class ProviderAuthError(ProviderError):
    """Invalid or expired API key."""
    pass


class ProviderRateLimitError(ProviderError):
    """Provider rate limit or quota exceeded."""
    pass


class ProviderModelError(ProviderError):
    """Model not found or unavailable."""
    pass
```

- [ ] **Step 2: Commit**

```bash
git add app/exceptions.py
git commit -m "feat: add provider exception hierarchy"
```

---

### Task 4: Provider base class and ModelInfo

**Files:**
- Create: `app/services/ai/providers/__init__.py`
- Create: `app/services/ai/providers/base.py`

- [ ] **Step 1: Create the base module**

Create `app/services/ai/providers/__init__.py`:

```python
from app.services.ai.providers.base import LLMProvider, ModelInfo

__all__ = ["LLMProvider", "ModelInfo", "get_provider"]
```

Note: `get_provider` will be added in a later task once all providers exist.

- [ ] **Step 2: Create the abstract base class**

Create `app/services/ai/providers/base.py`:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass
class ModelInfo:
    id: str
    name: str
    supports_structured_output: bool = False


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Return available models from this provider."""
        ...

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        inputs: list[str | dict[str, Any]],
        *,
        structured_output_schema: type[BaseModel] | None = None,
        temperature: float = 0.1,
        timeout: int | None = None,
    ) -> str:
        """Generate a response. Returns raw text.

        If structured_output_schema is provided and the provider supports native
        JSON mode, the provider should use it. Otherwise, the provider should
        inject the schema into the prompt as instructions.
        """
        ...
```

- [ ] **Step 3: Commit**

```bash
git add app/services/ai/providers/
git commit -m "feat: add LLMProvider base class and ModelInfo"
```

---

### Task 5: GeminiProvider

**Files:**
- Create: `app/services/ai/providers/gemini.py`
- Create: `tests/test_services/test_providers.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_services/test_providers.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_services/test_providers.py -v`
Expected: FAIL — `app.services.ai.providers.gemini` does not exist

- [ ] **Step 3: Write the implementation**

Create `app/services/ai/providers/gemini.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_services/test_providers.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/ai/providers/gemini.py tests/test_services/test_providers.py
git commit -m "feat: add GeminiProvider implementation"
```

---

### Task 6: OpenAIProvider

**Files:**
- Create: `app/services/ai/providers/openai_provider.py`
- Modify: `tests/test_services/test_providers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_services/test_providers.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_services/test_providers.py::test_openai_provider_generate -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the implementation**

Create `app/services/ai/providers/openai_provider.py`:

```python
import json
from logging import getLogger
from typing import Any

import openai
from pydantic import BaseModel

from app.exceptions import ProviderAuthError, ProviderRateLimitError, ProviderModelError, ProviderError
from app.services.ai.providers.base import LLMProvider, ModelInfo

logger = getLogger(__name__)


def _schema_to_json_instruction(schema: type[BaseModel]) -> str:
    """Convert a Pydantic model to a JSON schema instruction string."""
    return (
        "\n\nYou MUST respond with valid JSON matching this schema exactly:\n"
        f"```json\n{json.dumps(schema.model_json_schema(), indent=2)}\n```\n"
        "Respond ONLY with the JSON object, no other text."
    )


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model_id: str, base_url: str | None = None):
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = base_url
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = openai.AsyncOpenAI(**kwargs)

    async def list_models(self) -> list[ModelInfo]:
        try:
            result = await self.client.models.list()
            models = []
            for m in result.data:
                models.append(ModelInfo(
                    id=m.id,
                    name=m.id,
                    supports_structured_output="gpt" in m.id.lower() or "o1" in m.id.lower() or "o3" in m.id.lower(),
                ))
            return sorted(models, key=lambda x: x.id)
        except openai.AuthenticationError as e:
            raise ProviderAuthError(f"OpenAI authentication failed: {e}") from e
        except Exception as e:
            raise ProviderError(f"Failed to list OpenAI models: {e}") from e

    async def generate(
        self,
        system_prompt: str,
        inputs: list[str | dict[str, Any]],
        *,
        structured_output_schema: type[BaseModel] | None = None,
        temperature: float = 0.1,
        timeout: int | None = None,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]

        # Flatten inputs into a single user message
        user_parts = []
        for inp in inputs:
            if isinstance(inp, str):
                user_parts.append(inp)
            elif isinstance(inp, dict):
                # Skip vision inputs — OpenAI has different format
                user_parts.append("[Image content not supported for this provider]")
        messages.append({"role": "user", "content": "\n".join(user_parts)})

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
        }
        if timeout:
            kwargs["timeout"] = timeout

        # Use native JSON mode for structured output
        if structured_output_schema:
            try:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": structured_output_schema.__name__,
                        "schema": structured_output_schema.model_json_schema(),
                        "strict": False,
                    },
                }
            except Exception:
                # Fallback: inject schema into prompt
                messages[0]["content"] += _schema_to_json_instruction(structured_output_schema)

        try:
            response = await self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except openai.AuthenticationError as e:
            raise ProviderAuthError(f"OpenAI authentication failed: {e}") from e
        except openai.RateLimitError as e:
            raise ProviderRateLimitError(f"OpenAI rate limit: {e}") from e
        except openai.NotFoundError as e:
            raise ProviderModelError(f"OpenAI model not found: {e}") from e
        except Exception as e:
            raise ProviderError(f"OpenAI error: {e}") from e
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_services/test_providers.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/ai/providers/openai_provider.py tests/test_services/test_providers.py
git commit -m "feat: add OpenAIProvider implementation"
```

---

### Task 7: AnthropicProvider

**Files:**
- Create: `app/services/ai/providers/anthropic_provider.py`
- Modify: `tests/test_services/test_providers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_services/test_providers.py`:

```python
@pytest.mark.asyncio
async def test_anthropic_provider_generate():
    with patch("app.services.ai.providers.anthropic_provider.anthropic") as mock_anthropic:
        mock_client = AsyncMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        mock_block = MagicMock()
        mock_block.text = '{"name": "Alice"}'
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        from app.services.ai.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(api_key="sk-ant-test", model_id="claude-sonnet-4-6")
        result = await provider.generate(
            system_prompt="Extract info",
            inputs=["some text"],
        )
        assert result == '{"name": "Alice"}'


@pytest.mark.asyncio
async def test_anthropic_provider_list_models_returns_curated():
    with patch("app.services.ai.providers.anthropic_provider.anthropic"):
        from app.services.ai.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(api_key="sk-ant-test", model_id="claude-sonnet-4-6")
        models = await provider.list_models()
        assert len(models) >= 3
        ids = [m.id for m in models]
        assert "claude-sonnet-4-6" in ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_services/test_providers.py::test_anthropic_provider_generate -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the implementation**

Create `app/services/ai/providers/anthropic_provider.py`:

```python
import json
from logging import getLogger
from typing import Any

import anthropic
from pydantic import BaseModel

from app.exceptions import ProviderAuthError, ProviderRateLimitError, ProviderModelError, ProviderError
from app.services.ai.providers.base import LLMProvider, ModelInfo

logger = getLogger(__name__)

# Anthropic has no list-models endpoint — maintain a curated list
ANTHROPIC_MODELS = [
    ModelInfo(id="claude-opus-4-6", name="Claude Opus 4.6", supports_structured_output=False),
    ModelInfo(id="claude-sonnet-4-6", name="Claude Sonnet 4.6", supports_structured_output=False),
    ModelInfo(id="claude-haiku-4-5", name="Claude Haiku 4.5", supports_structured_output=False),
]


def _schema_to_json_instruction(schema: type[BaseModel]) -> str:
    """Convert a Pydantic model to a JSON schema instruction string for prompt injection."""
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
        # Inject schema into system prompt (Anthropic has no native JSON mode)
        effective_prompt = system_prompt
        if structured_output_schema:
            effective_prompt += _schema_to_json_instruction(structured_output_schema)

        # Flatten inputs into user message
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_services/test_providers.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/ai/providers/anthropic_provider.py tests/test_services/test_providers.py
git commit -m "feat: add AnthropicProvider implementation"
```

---

### Task 8: OpenAICompatibleProvider

**Files:**
- Create: `app/services/ai/providers/openai_compatible.py`
- Modify: `tests/test_services/test_providers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_services/test_providers.py`:

```python
@pytest.mark.asyncio
async def test_openai_compatible_generate():
    with patch("app.services.ai.providers.openai_compatible.openai") as mock_openai:
        mock_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = '{"name": "Bob"}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        from app.services.ai.providers.openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key="test-key",
            model_id="mixtral-8x7b",
            base_url="https://api.groq.com/openai/v1",
        )
        result = await provider.generate(
            system_prompt="Extract info",
            inputs=["some text"],
        )
        assert result == '{"name": "Bob"}'
        # Verify base_url was passed
        mock_openai.AsyncOpenAI.assert_called_once_with(
            api_key="test-key", base_url="https://api.groq.com/openai/v1"
        )


@pytest.mark.asyncio
async def test_openai_compatible_list_models_failure_returns_empty():
    with patch("app.services.ai.providers.openai_compatible.openai") as mock_openai:
        mock_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_client
        mock_client.models.list = AsyncMock(side_effect=Exception("Connection refused"))

        from app.services.ai.providers.openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key="test-key",
            model_id="local-model",
            base_url="http://localhost:11434/v1",
        )
        models = await provider.list_models()
        assert models == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_services/test_providers.py::test_openai_compatible_generate -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the implementation**

Create `app/services/ai/providers/openai_compatible.py`:

```python
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
        """Try to fetch models; return empty list on failure (frontend falls back to manual input)."""
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
            # Custom providers may not support native JSON mode — always use prompt injection
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_services/test_providers.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/ai/providers/openai_compatible.py tests/test_services/test_providers.py
git commit -m "feat: add OpenAICompatibleProvider implementation"
```

---

### Task 9: Provider factory function

**Files:**
- Modify: `app/services/ai/providers/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_services/test_providers.py`:

```python
from app.services.ai.providers.base import LLMProvider


def test_get_provider_returns_gemini_for_none():
    with patch("app.services.ai.providers.get_settings") as mock:
        mock.return_value.GEMINI_API_KEY = "platform-key"
        mock.return_value.GEMINI_FLASH_MODEL = "gemini-3-flash-preview"
        mock.return_value.GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"

        from app.services.ai.providers import get_provider
        from app.services.ai.providers.gemini import GeminiProvider

        provider = get_provider(config=None, purpose="profile_structuring")
        assert isinstance(provider, GeminiProvider)
        assert provider.model_id == "gemini-3-flash-preview"


def test_get_provider_returns_gemini_pro_for_tailoring():
    with patch("app.services.ai.providers.get_settings") as mock:
        mock.return_value.GEMINI_API_KEY = "platform-key"
        mock.return_value.GEMINI_FLASH_MODEL = "gemini-3-flash-preview"
        mock.return_value.GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"

        from app.services.ai.providers import get_provider
        provider = get_provider(config=None, purpose="resume_tailoring")
        assert provider.model_id == "gemini-3.1-pro-preview"


def test_get_provider_platform_gemini_uses_user_model():
    with patch("app.services.ai.providers.get_settings") as mock:
        mock.return_value.GEMINI_API_KEY = "platform-key"

        from app.services.ai.providers import get_provider
        from app.services.ai.providers.gemini import GeminiProvider

        config = MagicMock()
        config.provider = "PLATFORM_GEMINI"
        config.model_id = "gemini-2.5-pro"
        config.api_host = None

        provider = get_provider(config=config, purpose="anything")
        assert isinstance(provider, GeminiProvider)
        assert provider.model_id == "gemini-2.5-pro"
        assert provider.api_key == "platform-key"


def test_get_provider_openai():
    from app.services.ai.providers import get_provider
    from app.services.ai.providers.openai_provider import OpenAIProvider

    config = MagicMock()
    config.provider = "OPENAI"
    config.model_id = "gpt-4o"
    config.decrypted_api_key = "sk-user-key"
    config.api_host = None

    with patch("app.services.ai.providers.get_settings"):
        provider = get_provider(config=config, purpose="anything")
    assert isinstance(provider, OpenAIProvider)
    assert provider.model_id == "gpt-4o"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_services/test_providers.py::test_get_provider_returns_gemini_for_none -v`
Expected: FAIL — `get_provider` not yet implemented

- [ ] **Step 3: Write the factory implementation**

Replace `app/services/ai/providers/__init__.py`:

```python
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
```

- [ ] **Step 4: Run all provider tests**

Run: `uv run pytest tests/test_services/test_providers.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/ai/providers/__init__.py tests/test_services/test_providers.py
git commit -m "feat: add get_provider factory function"
```

---

### Task 10: UserAIConfig model and migration

**Files:**
- Create: `app/models/ai_config.py`
- Modify: `app/models/__init__.py:1-21`
- Modify: `alembic/env.py:11`
- Create: `tests/test_models/test_ai_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_models/test_ai_config.py`:

```python
import pytest
from app.models.ai_config import UserAIConfig, AIProvider


@pytest.mark.asyncio
async def test_create_ai_config(db_session, test_user):
    config = UserAIConfig(
        user_id=test_user.id,
        provider=AIProvider.OPENAI,
        api_key_encrypted="encrypted-data",
        model_id="gpt-4o",
    )
    db_session.add(config)
    await db_session.commit()
    await db_session.refresh(config)

    assert config.id is not None
    assert config.user_id == test_user.id
    assert config.provider == AIProvider.OPENAI
    assert config.model_id == "gpt-4o"


@pytest.mark.asyncio
async def test_ai_config_unique_per_user(db_session, test_user):
    config1 = UserAIConfig(
        user_id=test_user.id,
        provider=AIProvider.OPENAI,
        api_key_encrypted="encrypted-1",
        model_id="gpt-4o",
    )
    db_session.add(config1)
    await db_session.commit()

    config2 = UserAIConfig(
        user_id=test_user.id,
        provider=AIProvider.ANTHROPIC,
        api_key_encrypted="encrypted-2",
        model_id="claude-sonnet-4-6",
    )
    db_session.add(config2)
    with pytest.raises(Exception):  # IntegrityError — unique constraint
        await db_session.commit()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models/test_ai_config.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Create the model**

Create `app/models/ai_config.py`:

```python
import enum
from sqlalchemy import Integer, String, Text, Enum, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.models.base import Base, TimestampMixin


class AIProvider(str, enum.Enum):
    PLATFORM_GEMINI = "PLATFORM_GEMINI"
    GEMINI = "GEMINI"
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    CUSTOM_OPENAI_COMPATIBLE = "CUSTOM_OPENAI_COMPATIBLE"


class UserAIConfig(TimestampMixin, Base):
    __tablename__ = "user_ai_configs"
    __table_args__ = (
        UniqueConstraint("user_id", name="uq_user_ai_config_user_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[AIProvider] = mapped_column(Enum(AIProvider), nullable=False)
    api_key_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    api_host: Mapped[str | None] = mapped_column(String, nullable=True)
    model_id: Mapped[str] = mapped_column(String, nullable=False)

    user = relationship("User", backref="ai_config", uselist=False)

    @property
    def decrypted_api_key(self) -> str | None:
        if not self.api_key_encrypted:
            return None
        from app.services.crypto import decrypt
        return decrypt(self.api_key_encrypted)
```

- [ ] **Step 4: Register in models __init__**

In `app/models/__init__.py`, add the import. After line 8 (`from app.models.roast_view import RoastView`), add:

```python
from app.models.ai_config import UserAIConfig, AIProvider
```

And update `__all__` to include `"UserAIConfig", "AIProvider"`.

- [ ] **Step 5: Register in alembic env.py**

In `alembic/env.py` line 11, add `ai_config` to the import:

```python
from app.models import user, profile, job, token_usage, tenant, roast, credit, ai_config  # noqa: F401
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_models/test_ai_config.py -v`
Expected: 2 passed

- [ ] **Step 7: Generate Alembic migration**

Run: `uv run alembic revision --autogenerate -m "add user_ai_configs table"`

Verify the generated migration creates the `user_ai_configs` table with the correct columns and unique constraint.

- [ ] **Step 8: Commit**

```bash
git add app/models/ai_config.py app/models/__init__.py alembic/env.py tests/test_models/test_ai_config.py alembic/versions/
git commit -m "feat: add UserAIConfig model and migration"
```

---

### Task 11: AI config schemas

**Files:**
- Create: `app/schemas/ai_config.py`

- [ ] **Step 1: Create the schemas**

Create `app/schemas/ai_config.py`:

```python
from pydantic import BaseModel, field_validator
from app.models.ai_config import AIProvider


class AIConfigResponse(BaseModel):
    provider: AIProvider
    model_id: str
    api_host: str | None = None
    key_configured: bool = False
    key_hint: str | None = None  # last 4 chars, e.g. "••••abcd"

    model_config = {"from_attributes": True}


class AIConfigUpdate(BaseModel):
    provider: AIProvider
    api_key: str | None = None
    api_host: str | None = None
    model_id: str

    @field_validator("api_host")
    @classmethod
    def validate_api_host(cls, v, info):
        if info.data.get("provider") == AIProvider.CUSTOM_OPENAI_COMPATIBLE and not v:
            raise ValueError("api_host is required for CUSTOM_OPENAI_COMPATIBLE provider")
        return v

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v, info):
        provider = info.data.get("provider")
        if provider and provider != AIProvider.PLATFORM_GEMINI and not v:
            raise ValueError("api_key is required for non-platform providers")
        return v


class FetchModelsRequest(BaseModel):
    provider: AIProvider
    api_key: str | None = None
    api_host: str | None = None


class ModelInfoResponse(BaseModel):
    id: str
    name: str
    supports_structured_output: bool = False
```

- [ ] **Step 2: Commit**

```bash
git add app/schemas/ai_config.py
git commit -m "feat: add AI config Pydantic schemas"
```

---

### Task 12: AI settings API router

**Files:**
- Create: `app/api/ai_settings.py`
- Modify: `app/main.py:237-258`
- Create: `tests/test_api/test_ai_settings.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api/test_ai_settings.py`:

```python
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.models.ai_config import UserAIConfig, AIProvider


@pytest.mark.asyncio
async def test_get_ai_config_empty(client):
    resp = await client.get("/settings/ai")
    assert resp.status_code == 200
    assert resp.json() is None


@pytest.mark.asyncio
async def test_save_platform_gemini(client, db_session, test_user):
    with patch("app.api.ai_settings._fetch_models_for_provider", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = []
        resp = await client.put("/settings/ai", json={
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
    resp = await client.put("/settings/ai", json={
        "provider": "OPENAI",
        "model_id": "gpt-4o",
    })
    assert resp.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_delete_ai_config(client, db_session, test_user):
    # First create a config
    config = UserAIConfig(
        user_id=test_user.id,
        provider=AIProvider.PLATFORM_GEMINI,
        model_id="gemini-3-flash-preview",
    )
    db_session.add(config)
    await db_session.commit()

    resp = await client.delete("/settings/ai")
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api/test_ai_settings.py -v`
Expected: FAIL — module does not exist / no route

- [ ] **Step 3: Write the router**

Create `app/api/ai_settings.py`:

```python
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
    """Fetch models from the selected provider. Returns list of dicts."""
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
```

- [ ] **Step 4: Register the router in main.py**

In `app/main.py`, after line 246 (`from app.api.chat import router as chat_router`), add:

```python
    from app.api.ai_settings import router as ai_settings_router
```

After line 258 (`app.include_router(profile_chat_router)`), add:

```python
    app.include_router(ai_settings_router)
```

- [ ] **Step 5: Add provider exception handlers to main.py**

In `app/main.py`, add to the imports (after `ConflictError` on line 23):

```python
from app.exceptions import ProviderError, ProviderAuthError, ProviderRateLimitError, ProviderModelError
```

After the `ConflictError` handler (line 192), add:

```python
    @app.exception_handler(ProviderAuthError)
    async def provider_auth_handler(request: Request, exc: ProviderAuthError):
        return JSONResponse(status_code=401, content={"detail": str(exc)})

    @app.exception_handler(ProviderRateLimitError)
    async def provider_rate_limit_handler(request: Request, exc: ProviderRateLimitError):
        return JSONResponse(status_code=429, content={"detail": str(exc)})

    @app.exception_handler(ProviderModelError)
    async def provider_model_handler(request: Request, exc: ProviderModelError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ProviderError)
    async def provider_error_handler(request: Request, exc: ProviderError):
        return JSONResponse(status_code=502, content={"detail": str(exc)})
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_api/test_ai_settings.py -v`
Expected: 5 passed

- [ ] **Step 7: Run full test suite to check for regressions**

Run: `uv run pytest tests/ -v --ignore=tests/integration`
Expected: All existing tests pass + 5 new tests pass

- [ ] **Step 8: Commit**

```bash
git add app/api/ai_settings.py app/main.py tests/test_api/test_ai_settings.py
git commit -m "feat: add /settings/ai API endpoints"
```

---

### Task 13: Refactor InferenceService to use providers

**Files:**
- Modify: `app/services/ai/inference.py`

This is the core refactoring step. `GeminiInference` becomes `InferenceService` and accepts an optional provider.

- [ ] **Step 1: Refactor inference.py**

Replace the entire `GeminiInference` class in `app/services/ai/inference.py`. Keep the `_log_request` function and constants unchanged. Replace from line 62 onward:

```python
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
    ):
        from app.services.ai.providers.base import LLMProvider as _LLM

        if provider:
            self.provider: _LLM = provider
            self.model = getattr(provider, "model_id", model_name or "unknown")
        else:
            # Legacy path: construct a GeminiProvider from settings
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
            try:
                return await self.provider.generate(**call_kwargs)
            except asyncio.TimeoutError:
                # For non-Gemini providers or same-model fallback, just re-raise
                if not fallback_model or fallback_model == self.model:
                    raise
                logger.warning(
                    f"Primary model {self.model} timed out, falling back to {fallback_model}"
                )
                # Fallback only for platform Gemini (construct a fresh Gemini provider)
                from app.services.ai.providers.gemini import GeminiProvider
                if not isinstance(self.provider, GeminiProvider):
                    raise  # BYOK — no cross-provider fallback
                settings = get_settings()
                fb_provider = GeminiProvider(api_key=settings.GEMINI_API_KEY, model_id=fallback_model)
                fb_kwargs = {**call_kwargs, "timeout": None}
                return await fb_provider.generate(**fb_kwargs)
            except Exception:
                if not fallback_model or fallback_model == self.model:
                    raise
                from app.services.ai.providers.gemini import GeminiProvider
                if not isinstance(self.provider, GeminiProvider):
                    raise  # BYOK — no cross-provider fallback
                logger.warning(
                    f"Primary model {self.model} failed, falling back to {fallback_model}"
                )
                settings = get_settings()
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
```

- [ ] **Step 2: Run existing tests to verify backward compatibility**

Run: `uv run pytest tests/ -v --ignore=tests/integration`
Expected: All existing tests still pass — `GeminiInference` alias ensures no import breaks

- [ ] **Step 3: Commit**

```bash
git add app/services/ai/inference.py
git commit -m "refactor: rename GeminiInference to InferenceService, delegate to providers"
```

---

### Task 14: Wire user AI config into services

**Files:**
- Modify: `app/services/profile/service.py`
- Modify: `app/services/job/service.py`
- Modify: `app/services/roast/service.py`
- Modify: `app/services/ocr/extractor.py`

- [ ] **Step 1: Update ProfileService**

In `app/services/profile/service.py`:

Replace the import on line 9:
```python
from app.services.ai.inference import GeminiInference
```
with:
```python
from app.services.ai.inference import InferenceService
from app.services.ai.providers import get_provider
```

In `process_profile()` method (line 43), add `ai_config=None` parameter:

```python
    async def process_profile(
        self, db: AsyncSession, profile_id: int, pdf_bytes: bytes, extracted_text: str = "", ai_config=None
    ) -> None:
```

Replace the text path LLM call (lines 79-87):
```python
                llm = InferenceService(provider=get_provider(ai_config, purpose="profile_structuring"))
                result = await llm.run_inference(
                    system_prompt=STRUCTURED_RESUME_SYSTEM_PROMPT,
                    inputs=[text],
                    structured_output_schema=ResumeInfo,
                    user_id=profile.user_id,
                    purpose="profile_structuring",
                    reference_id=str(profile_id),
                )
```

For the vision path (lines 70-76), check if non-Gemini provider and fall back to text:
```python
            if needs_vision:
                # Vision path is Gemini-only; non-Gemini BYOK falls back to text
                from app.services.ai.providers.gemini import GeminiProvider
                provider = get_provider(ai_config, purpose="profile_structuring")
                if isinstance(provider, GeminiProvider):
                    logger.info(f"[profile:{profile_id}] Using vision path")
                    result = await self.extractor.extract_and_structure_via_vision(
                        pdf_bytes,
                        user_id=profile.user_id,
                        reference_id=str(profile_id),
                    )
                else:
                    logger.info(f"[profile:{profile_id}] Non-Gemini provider, using text path for vision-needed PDF")
                    # Force text path even for poor quality text
                    if not text.strip():
                        text = "(No text could be extracted from this PDF)"
                    llm = InferenceService(provider=provider)
                    result = await llm.run_inference(
                        system_prompt=STRUCTURED_RESUME_SYSTEM_PROMPT,
                        inputs=[text],
                        structured_output_schema=ResumeInfo,
                        user_id=profile.user_id,
                        purpose="profile_structuring",
                        reference_id=str(profile_id),
                    )
```

In `enhance_profile()` (line 146), add `ai_config=None` parameter and use it:

```python
    async def enhance_profile(self, db: AsyncSession, profile_id: int, user_id: str, ai_config=None) -> Profile:
        profile = await self.get_profile(db, profile_id, user_id)
        llm = InferenceService(provider=get_provider(ai_config, purpose="profile_enhancement"))
        result = await llm.run_inference(
```

- [ ] **Step 2: Update JobService**

In `app/services/job/service.py`:

Replace the import on line 11:
```python
from app.services.ai.inference import GeminiInference
```
with:
```python
from app.services.ai.inference import InferenceService
from app.services.ai.providers import get_provider
```

In `generate_custom_resume()` (line 98), add `ai_config=None`:
```python
    async def generate_custom_resume(self, db: AsyncSession, job_id: int, user_id: str, ai_config=None) -> None:
```

Replace the LLM call (lines 118-126):
```python
            llm = InferenceService(provider=get_provider(ai_config, purpose="resume_tailoring"))
            result = await llm.run_inference(
```

- [ ] **Step 3: Update RoastService**

In `app/services/roast/service.py`:

Replace the import on line 18:
```python
from app.services.ai.inference import GeminiInference
```
with:
```python
from app.services.ai.inference import InferenceService
from app.services.ai.providers import get_provider
```

In `process_roast()` (line 78), add `ai_config=None`:
```python
    async def process_roast(
        self, db: AsyncSession, roast_id: int, pdf_bytes: bytes, extracted_text: str = "", ai_config=None
    ) -> None:
```

Replace the LLM instantiation (line 100):
```python
            llm = InferenceService(provider=get_provider(ai_config, purpose="resume_roast"))
```

- [ ] **Step 4: Run existing tests**

Run: `uv run pytest tests/ -v --ignore=tests/integration`
Expected: All existing tests still pass — `ai_config=None` defaults preserve existing behavior

- [ ] **Step 5: Commit**

```bash
git add app/services/profile/service.py app/services/job/service.py app/services/roast/service.py
git commit -m "feat: wire user AI config into profile, job, and roast services"
```

---

### Task 15: Wire AI config into API routes

**Files:**
- Modify: `app/api/profiles.py`
- Modify: `app/api/jobs.py`
- Modify: `app/api/roasts.py` (if it exists — check and follow the same pattern)

The pattern: query `UserAIConfig` for the current user, pass it into background tasks.

- [ ] **Step 1: Update profiles router**

In `app/api/profiles.py`, add import:

```python
from app.models.ai_config import UserAIConfig
```

In the upload endpoint, after getting `current_user`, query the config:

```python
    # Query user's AI config
    ai_config_result = await db.execute(
        select(UserAIConfig).where(UserAIConfig.user_id == current_user.id)
    )
    ai_config = ai_config_result.scalar_one_or_none()
```

Then pass `ai_config=ai_config` into the background `service.process_profile(...)` call.

Similarly in the enhance endpoint, query the config and pass `ai_config=ai_config` to `service.enhance_profile(...)`.

- [ ] **Step 2: Update jobs router**

In `app/api/jobs.py`, add import:

```python
from app.models.ai_config import UserAIConfig
```

In the generate-resume endpoint, query the config and pass `ai_config=ai_config` into the background `service.generate_custom_resume(...)` call.

- [ ] **Step 3: Update roasts router**

Follow the same pattern: query `UserAIConfig`, pass to `service.process_roast(...)`.

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -v --ignore=tests/integration`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add app/api/profiles.py app/api/jobs.py app/api/roasts.py
git commit -m "feat: query user AI config in API routes and pass to services"
```

---

### Task 16: Frontend — AI Settings page

**Files:**
- Modify: `frontend/static/js/app.js`

- [ ] **Step 1: Add the AISettingsPage component**

Add the component before the `AppLayout` definition (before line 4076). Insert:

```javascript
// ================================================================
// PAGES — AI Settings
// ================================================================
const AISettingsPage = {
  template: `
    <div>
      <TopHeader>
        <template #left>
          <div class="text-sm font-mono">
            <span class="font-bold text-white">AI SETTINGS</span>
            <p class="text-[10px] font-mono mt-0.5 hidden md:block" style="color:var(--text-dim)">Configure your AI provider and model</p>
          </div>
        </template>
      </TopHeader>
      <div class="flex-1 overflow-y-auto p-4 md:p-6 page-scroll">
        <div class="max-w-lg mx-auto">
          <div class="card p-6">
            <!-- Provider -->
            <label class="block text-[10px] font-mono font-bold tracking-widest uppercase mb-2" style="color:var(--text-dim)">Provider</label>
            <select v-model="form.provider" @change="onProviderChange" class="input-field w-full mb-4">
              <option value="PLATFORM_GEMINI">Platform Default (Gemini)</option>
              <option value="GEMINI">Google Gemini (Own Key)</option>
              <option value="OPENAI">OpenAI</option>
              <option value="ANTHROPIC">Anthropic</option>
              <option value="CUSTOM_OPENAI_COMPATIBLE">Custom (OpenAI-compatible)</option>
            </select>

            <!-- API Key -->
            <div v-if="form.provider !== 'PLATFORM_GEMINI'" class="mb-4">
              <label class="block text-[10px] font-mono font-bold tracking-widest uppercase mb-2" style="color:var(--text-dim)">API Key</label>
              <input v-model="form.api_key" type="password" :placeholder="keyHint || 'Enter your API key'" class="input-field w-full">
            </div>

            <!-- API Host -->
            <div v-if="form.provider === 'CUSTOM_OPENAI_COMPATIBLE'" class="mb-4">
              <label class="block text-[10px] font-mono font-bold tracking-widest uppercase mb-2" style="color:var(--text-dim)">API Host</label>
              <input v-model="form.api_host" type="text" placeholder="https://api.groq.com/openai/v1" class="input-field w-full">
            </div>

            <!-- Fetch Models -->
            <button @click="fetchModels" :disabled="fetchingModels" class="btn-secondary w-full mb-4">
              {{ fetchingModels ? 'Fetching...' : 'Fetch Available Models' }}
            </button>

            <!-- Model Select -->
            <div class="mb-6">
              <label class="block text-[10px] font-mono font-bold tracking-widest uppercase mb-2" style="color:var(--text-dim)">Model</label>
              <select v-if="models.length > 0" v-model="form.model_id" class="input-field w-full">
                <option v-for="m in models" :key="m.id" :value="m.id">{{ m.name || m.id }}</option>
              </select>
              <input v-else v-model="form.model_id" type="text" placeholder="Enter model ID" class="input-field w-full">
            </div>

            <!-- Actions -->
            <div class="flex gap-3">
              <button @click="save" :disabled="saving" class="btn-primary flex-1 justify-center">
                {{ saving ? 'Saving...' : 'Save' }}
              </button>
              <button @click="reset" :disabled="saving" class="btn-secondary">
                Reset to Default
              </button>
            </div>

            <p v-if="error" class="text-red-400 text-xs mt-3 font-mono">{{ error }}</p>
            <p v-if="success" class="text-green-400 text-xs mt-3 font-mono">{{ success }}</p>

            <p class="text-[10px] font-mono mt-4" style="color:var(--text-dim)">
              Your API key is encrypted at rest. Platform default uses our Gemini API.
            </p>
          </div>
        </div>
      </div>
    </div>
  `,
  setup() {
    const form = ref({
      provider: 'PLATFORM_GEMINI',
      api_key: '',
      api_host: '',
      model_id: '',
    });
    const models = ref([]);
    const fetchingModels = ref(false);
    const saving = ref(false);
    const error = ref('');
    const success = ref('');
    const keyHint = ref('');

    const load = async () => {
      try {
        const resp = await api('/settings/ai');
        if (resp) {
          form.value.provider = resp.provider;
          form.value.model_id = resp.model_id;
          form.value.api_host = resp.api_host || '';
          form.value.api_key = '';
          keyHint.value = resp.key_hint || '';
        }
      } catch {}
    };

    const onProviderChange = () => {
      form.value.api_key = '';
      form.value.api_host = '';
      form.value.model_id = '';
      models.value = [];
      keyHint.value = '';
      error.value = '';
      success.value = '';
    };

    const fetchModels = async () => {
      fetchingModels.value = true;
      error.value = '';
      try {
        const body = {
          provider: form.value.provider,
          api_key: form.value.api_key || undefined,
          api_host: form.value.api_host || undefined,
        };
        const resp = await api('/settings/ai/models', { method: 'POST', body: JSON.stringify(body) });
        models.value = resp || [];
        if (models.value.length === 0) {
          error.value = 'No models returned. You can type a model ID manually.';
        }
      } catch (e) {
        error.value = e.message || 'Failed to fetch models';
        models.value = [];
      } finally {
        fetchingModels.value = false;
      }
    };

    const save = async () => {
      saving.value = true;
      error.value = '';
      success.value = '';
      try {
        const body = {
          provider: form.value.provider,
          model_id: form.value.model_id,
          api_key: form.value.api_key || undefined,
          api_host: form.value.api_host || undefined,
        };
        await api('/settings/ai', { method: 'PUT', body: JSON.stringify(body) });
        success.value = 'Settings saved!';
        form.value.api_key = '';
        await load();
      } catch (e) {
        error.value = e.message || 'Failed to save';
      } finally {
        saving.value = false;
      }
    };

    const reset = async () => {
      saving.value = true;
      error.value = '';
      success.value = '';
      try {
        await api('/settings/ai', { method: 'DELETE' });
        form.value = { provider: 'PLATFORM_GEMINI', api_key: '', api_host: '', model_id: '' };
        models.value = [];
        keyHint.value = '';
        success.value = 'Reset to platform default.';
      } catch (e) {
        error.value = e.message || 'Failed to reset';
      } finally {
        saving.value = false;
      }
    };

    onMounted(load);

    return { form, models, fetchingModels, saving, error, success, keyHint, onProviderChange, fetchModels, save, reset };
  },
};
```

- [ ] **Step 2: Add route to router**

In the routes array (around line 5293), add before the `admin` route:

```javascript
      { path: 'settings/ai', component: AISettingsPage },
```

- [ ] **Step 3: Add sidebar link**

In the `AppSidebar` template, after the credits link (around line 952) and before the admin section (line 954), add a new section:

```html
        <div class="pt-4 mt-4 border-t border-white/5">
          <router-link to="/settings/ai" class="sidebar-link" active-class="active">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12.22 2h-.44a2 2 0 00-2 2v.18a2 2 0 01-1 1.73l-.43.25a2 2 0 01-2 0l-.15-.08a2 2 0 00-2.73.73l-.22.38a2 2 0 00.73 2.73l.15.1a2 2 0 011 1.72v.51a2 2 0 01-1 1.74l-.15.09a2 2 0 00-.73 2.73l.22.38a2 2 0 002.73.73l.15-.08a2 2 0 012 0l.43.25a2 2 0 011 1.73V20a2 2 0 002 2h.44a2 2 0 002-2v-.18a2 2 0 011-1.73l.43-.25a2 2 0 012 0l.15.08a2 2 0 002.73-.73l.22-.39a2 2 0 00-.73-2.73l-.15-.08a2 2 0 01-1-1.74v-.5a2 2 0 011-1.74l.15-.09a2 2 0 00.73-2.73l-.22-.38a2 2 0 00-2.73-.73l-.15.08a2 2 0 01-2 0l-.43-.25a2 2 0 01-1-1.73V4a2 2 0 00-2-2z"/><circle cx="12" cy="12" r="3"/></svg>
            AI Settings
          </router-link>
        </div>
```

- [ ] **Step 4: Verify the app loads**

Start the dev server and navigate to `/#/settings/ai` to confirm the page renders, provider dropdown works, and the API calls succeed.

Run: `uv run python -m app.main`
Then visit: http://localhost:8000/#/settings/ai

- [ ] **Step 5: Commit**

```bash
git add frontend/static/js/app.js
git commit -m "feat: add AI Settings page to frontend"
```

---

### Task 17: Update .env.example and final verification

**Files:**
- Modify: `.env.example` (if it exists)

- [ ] **Step 1: Add ENCRYPTION_KEY to .env.example**

Add under the AI section:

```
# Encryption (for user API key storage)
ENCRYPTION_KEY=           # Generate with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -v --ignore=tests/integration`
Expected: All tests pass (existing + new)

- [ ] **Step 3: Commit**

```bash
git add .env.example
git commit -m "docs: add ENCRYPTION_KEY to .env.example"
```
