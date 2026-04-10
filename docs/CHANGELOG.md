# Changelog

## [Unreleased] - 2026-04-10

### Multi-Provider AI Configuration

Users can now choose their own LLM provider and model instead of being locked to the platform's default Gemini models.

#### New Features

- **AI Settings page** (`/#/settings/ai`) — new frontend page accessible from the sidebar where users configure their AI provider
- **Provider support** — Google Gemini, OpenAI, Anthropic, and any OpenAI-compatible endpoint (Groq, Together, Ollama, vLLM, etc.)
- **Bring Your Own Key (BYOK)** — users can supply their own API key for any supported provider; keys are encrypted at rest with Fernet
- **Model discovery** — fetch available models from the selected provider's API; falls back to manual entry for custom endpoints that don't support `/v1/models`
- **Platform default preserved** — users who don't configure anything continue using the platform's Gemini key with flash/pro model selection per task type
- **API endpoints**:
  - `GET /settings/ai` — retrieve current config (API key never returned, only a masked hint)
  - `PUT /settings/ai` — save provider, model, and encrypted API key
  - `DELETE /settings/ai` — reset to platform default
  - `POST /settings/ai/models` — fetch available models from a provider (supports pre-save browsing)

#### Architecture

- **Provider Strategy Pattern** — abstract `LLMProvider` base class with four concrete implementations (`GeminiProvider`, `OpenAIProvider`, `AnthropicProvider`, `OpenAICompatibleProvider`)
- **Factory function** (`get_provider`) — selects the right provider based on user config and task purpose
- **`InferenceService`** — renamed from `GeminiInference`, now delegates `generate()` to the selected provider while keeping retry, schema validation, and logging centralized
- **Structured output** — native JSON schema mode for Gemini and OpenAI; prompt-based JSON extraction with validation retry for Anthropic and custom providers
- **Fallback** — platform Gemini users retain the existing primary-model-timeout-then-fallback behavior; BYOK users get no cross-provider fallback (errors surface directly)

#### Data Model

- New `user_ai_configs` table — one row per user, stores provider enum, encrypted API key, optional API host (for custom endpoints), and selected model ID
- `AIProvider` enum: `PLATFORM_GEMINI`, `GEMINI`, `OPENAI`, `ANTHROPIC`, `CUSTOM_OPENAI_COMPATIBLE`

#### Security

- API keys encrypted at rest using Fernet symmetric encryption (`ENCRYPTION_KEY` env var)
- API key never returned by the API — only `key_configured: bool` and a masked hint (last 4 chars)
- Key validated on save via a lightweight `list_models()` call

#### Dependencies Added

- `openai>=1.40.0`
- `anthropic>=0.34.0`

#### Configuration

- New env var: `ENCRYPTION_KEY` — required for BYOK; generate with `python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'`

#### Files Changed

| Category | Files |
|----------|-------|
| New — providers | `app/services/ai/providers/__init__.py`, `base.py`, `gemini.py`, `openai_provider.py`, `anthropic_provider.py`, `openai_compatible.py` |
| New — model/schema | `app/models/ai_config.py`, `app/schemas/ai_config.py` |
| New — API | `app/api/ai_settings.py` |
| New — utility | `app/services/crypto.py` |
| New — tests | `tests/test_services/test_crypto.py`, `tests/test_services/test_providers.py`, `tests/test_models/test_ai_config.py`, `tests/test_api/test_ai_settings.py` |
| Modified | `app/services/ai/inference.py` (refactored to `InferenceService`) |
| Modified | `app/services/profile/service.py`, `app/services/job/service.py`, `app/services/roast/service.py` (accept `ai_config`) |
| Modified | `app/api/profiles.py`, `app/api/jobs.py`, `app/api/roasts.py` (query and pass user config) |
| Modified | `app/main.py` (router + exception handlers) |
| Modified | `app/exceptions.py` (provider exception hierarchy) |
| Modified | `app/config.py` (`ENCRYPTION_KEY` setting) |
| Modified | `app/models/__init__.py`, `alembic/env.py` (model registration) |
| Modified | `pyproject.toml` (new deps) |
| Modified | `frontend/static/js/app.js` (AI Settings page, sidebar link, route) |
| Modified | `.env.example` (`ENCRYPTION_KEY`) |

#### Migration Required

Run `alembic revision --autogenerate -m "add user_ai_configs table"` then `alembic upgrade head` to create the new table.
