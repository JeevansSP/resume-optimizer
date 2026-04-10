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
    provider: Mapped[AIProvider] = mapped_column(Enum(AIProvider, create_type=False), nullable=False)
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
