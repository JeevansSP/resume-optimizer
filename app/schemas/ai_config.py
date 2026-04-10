from pydantic import BaseModel, field_validator, model_validator
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

    @model_validator(mode="after")
    def validate_api_key(self):
        if self.provider and self.provider != AIProvider.PLATFORM_GEMINI and not self.api_key:
            raise ValueError("api_key is required for non-platform providers")
        return self


class FetchModelsRequest(BaseModel):
    provider: AIProvider
    api_key: str | None = None
    api_host: str | None = None


class ModelInfoResponse(BaseModel):
    id: str
    name: str
    supports_structured_output: bool = False
