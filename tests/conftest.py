import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.models.base import Base
from app.models import user, profile, job, token_usage, tenant, roast, roast_view, credit, ai_config  # noqa: F401  # imports ensure all models registered
from app.main import create_app
from app.database.session import get_db
from app.dependencies import get_current_user, get_super_admin
from app.models.user import User


@pytest_asyncio.fixture
async def async_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(async_engine):
    session_factory = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def test_user(db_session):
    user_obj = User(
        id="test-user-id",
        google_id="google-123",
        email="test@example.com",
        name="Test User",
        consent_accepted=True,
    )
    db_session.add(user_obj)
    await db_session.commit()
    return user_obj


@pytest_asyncio.fixture
async def client(async_engine, test_user):
    app = create_app()

    session_factory = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

    async def _override_get_db():
        async with session_factory() as session:
            yield session

    async def _override_get_current_user():
        return test_user

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_current_user] = _override_get_current_user

    # Override async_session_factory in chat modules so background tasks use test DB
    import app.api.chat as chat_mod
    import app.api.profile_chat as pchat_mod
    orig_chat_factory = getattr(chat_mod, 'async_session_factory', None)
    orig_pchat_factory = getattr(pchat_mod, 'async_session_factory', None)
    chat_mod.async_session_factory = session_factory
    pchat_mod.async_session_factory = session_factory

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

    # Restore
    if orig_chat_factory is not None:
        chat_mod.async_session_factory = orig_chat_factory
    if orig_pchat_factory is not None:
        pchat_mod.async_session_factory = orig_pchat_factory


@pytest_asyncio.fixture
async def super_admin_user(db_session):
    user_obj = User(
        id="admin-user-id",
        google_id="google-admin-456",
        email="admin@example.com",
        name="Admin User",
        consent_accepted=True,
        is_super_admin=True,
    )
    db_session.add(user_obj)
    await db_session.commit()
    return user_obj


@pytest_asyncio.fixture
async def admin_client(async_engine, super_admin_user):
    app = create_app()

    session_factory = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

    async def _override_get_db():
        async with session_factory() as session:
            yield session

    async def _override_get_current_user():
        return super_admin_user

    async def _override_get_super_admin():
        return super_admin_user

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_current_user] = _override_get_current_user
    app.dependency_overrides[get_super_admin] = _override_get_super_admin

    import app.api.chat as chat_mod
    import app.api.profile_chat as pchat_mod
    orig_chat_factory = getattr(chat_mod, 'async_session_factory', None)
    orig_pchat_factory = getattr(pchat_mod, 'async_session_factory', None)
    chat_mod.async_session_factory = session_factory
    pchat_mod.async_session_factory = session_factory

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

    if orig_chat_factory is not None:
        chat_mod.async_session_factory = orig_chat_factory
    if orig_pchat_factory is not None:
        pchat_mod.async_session_factory = orig_pchat_factory
