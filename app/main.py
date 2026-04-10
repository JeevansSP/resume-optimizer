import asyncio
import html as html_mod
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.config import get_settings
from app.exceptions import (
    ProfileNotFoundError,
    JobNotFoundError,
    LaTeXCompilationError,
    AIInferenceError,
    UsageLimitExceeded,
    AuthenticationError,
    ForbiddenError,
    RoastNotFoundError,
    NotFoundError,
    BadRequestError,
    ConflictError,
    ProviderError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderModelError,
)

logger = logging.getLogger(__name__)

# ── Background task tracking ────────────────────────────────────────────────
_active_tasks: set[asyncio.Task] = set()


def create_tracked_task(coro) -> asyncio.Task:
    """Schedule a coroutine as a tracked asyncio Task.

    Tracked tasks are awaited during graceful shutdown so in-flight
    AI generation / PDF compilation can finish before the process exits.
    """

    async def _safe_wrapper():
        try:
            await coro
        except asyncio.CancelledError:
            raise  # let cancellation propagate for graceful shutdown
        except Exception:
            logger.exception("Unhandled exception in background task")

    task = asyncio.create_task(_safe_wrapper())
    _active_tasks.add(task)
    task.add_done_callback(_active_tasks.discard)
    return task


# ── Startup: recover rows stuck by a previous hard kill ─────────────────────
async def _recover_stuck_items():
    from sqlalchemy import update
    from app.database.session import async_session_factory
    from app.models.profile import Profile, ProfileStatus
    from app.models.job import Job, JobStatus
    from app.models.roast import Roast, RoastStatus

    async with async_session_factory() as db:
        r1 = await db.execute(
            update(Profile)
            .where(Profile.status == ProfileStatus.PROCESSING)
            .values(status=ProfileStatus.FAILED)
        )
        r2 = await db.execute(
            update(Job)
            .where(Job.status.in_([JobStatus.GENERATING_RESUME, JobStatus.GENERATING_PDF]))
            .values(status=JobStatus.FAILED)
        )
        r3 = await db.execute(
            update(Roast)
            .where(Roast.status == RoastStatus.PROCESSING)
            .values(status=RoastStatus.FAILED)
        )
        await db.commit()

    total = r1.rowcount + r2.rowcount + r3.rowcount
    if total:
        logger.warning(
            "Recovered %d stuck item(s) from previous shutdown: "
            "%d profile(s), %d job(s), %d roast(s)",
            total, r1.rowcount, r2.rowcount, r3.rowcount,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    logger.info("Starting up")
    try:
        await _recover_stuck_items()
    except Exception:
        logger.exception("Failed to recover stuck items on startup (DB may not be ready yet)")

    yield

    # ── Shutdown ──
    # Cloud Run sends SIGTERM then SIGKILL after 10s.
    # Leave 2s buffer for engine disposal → wait up to 8s for tasks.
    if _active_tasks:
        logger.info("Shutting down — waiting for %d background task(s)", len(_active_tasks))
        _, pending = await asyncio.wait(_active_tasks, timeout=8)
        if pending:
            logger.warning("Force-cancelling %d task(s) that didn't finish", len(pending))
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    else:
        logger.info("Shutting down — no active background tasks")

    from app.database.session import engine
    await engine.dispose()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="ATS Beater API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.FRONTEND_URL],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Domain redirect: send old Cloud Run URLs to the custom domain
    custom_domain = settings.FRONTEND_URL.replace("https://", "").replace("http://", "") if settings.FRONTEND_URL else ""

    @app.middleware("http")
    async def redirect_to_custom_domain(request: Request, call_next):
        host = request.headers.get("host", "")
        # Only redirect Cloud Run URLs (*.run.app) to custom domain
        if host.endswith(".run.app") and request.url.path != "/health":
            target = f"https://{custom_domain}{request.url.path}"
            if request.url.query:
                target += f"?{request.url.query}"
            return RedirectResponse(url=target, status_code=301)
        return await call_next(request)

    # Exception handlers
    @app.exception_handler(ProfileNotFoundError)
    async def profile_not_found_handler(request: Request, exc: ProfileNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(JobNotFoundError)
    async def job_not_found_handler(request: Request, exc: JobNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(LaTeXCompilationError)
    async def latex_error_handler(request: Request, exc: LaTeXCompilationError):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.exception_handler(AIInferenceError)
    async def ai_error_handler(request: Request, exc: AIInferenceError):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(UsageLimitExceeded)
    async def usage_limit_handler(request: Request, exc: UsageLimitExceeded):
        return JSONResponse(status_code=429, content={"detail": str(exc)})

    @app.exception_handler(AuthenticationError)
    async def auth_error_handler(request: Request, exc: AuthenticationError):
        return JSONResponse(status_code=401, content={"detail": str(exc)})

    @app.exception_handler(ForbiddenError)
    async def forbidden_error_handler(request: Request, exc: ForbiddenError):
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    @app.exception_handler(RoastNotFoundError)
    async def roast_not_found_handler(request: Request, exc: RoastNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(NotFoundError)
    async def not_found_handler(request: Request, exc: NotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(BadRequestError)
    async def bad_request_handler(request: Request, exc: BadRequestError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ConflictError)
    async def conflict_error_handler(request: Request, exc: ConflictError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

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

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        # Return JSON for API requests, styled HTML for browser requests
        accept = request.headers.get("accept", "")
        if "text/html" not in accept:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        return HTMLResponse(status_code=exc.status_code, content=f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{exc.status_code} — ATS Beater</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0f1e;color:#e2e8f0;font-family:'Inter',sans-serif;min-height:100vh;display:flex;align-items:center;justify-content:center;text-align:center;padding:24px}}
.container{{max-width:480px}}
.code{{font-family:'JetBrains Mono',monospace;font-size:clamp(80px,15vw,140px);font-weight:800;background:linear-gradient(135deg,#f97316,#14b8a6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1}}
.message{{font-size:18px;color:#94a3b8;margin:16px 0 32px;line-height:1.6}}
.btn{{display:inline-block;padding:12px 32px;background:linear-gradient(135deg,#f97316,#fb923c);color:#0a0f1e;font-family:'JetBrains Mono',monospace;font-weight:700;font-size:13px;text-transform:uppercase;letter-spacing:1px;text-decoration:none;border-radius:6px;transition:transform 0.2s}}
.btn:hover{{transform:translateY(-2px)}}
.brand{{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:2px;color:#475569;margin-top:48px}}
</style>
</head>
<body>
<div class="container">
<div class="code">{exc.status_code}</div>
<p class="message">{"This page doesn't exist. It might have been moved or you may have mistyped the URL." if exc.status_code == 404 else html_mod.escape(str(exc.detail))}</p>
<a href="/" class="btn">Go Home</a>
<p class="brand">ATS BEATER</p>
</div>
</body>
</html>""")

    # Health check
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # Mount routers
    from app.api.auth import router as auth_router
    from app.api.profiles import router as profiles_router
    from app.api.jobs import router as jobs_router
    from app.api.credits import router as credits_router
    from app.api.payments import router as payments_router
    from app.api.progress import router as progress_router
    from app.api.admin import router as admin_router
    from app.api.roasts import router as roasts_router
    from app.api.chat import router as chat_router
    from app.api.profile_chat import router as profile_chat_router
    from app.api.ai_settings import router as ai_settings_router

    app.include_router(auth_router)
    app.include_router(profiles_router)
    app.include_router(jobs_router)
    app.include_router(credits_router)
    app.include_router(payments_router)
    app.include_router(progress_router)
    app.include_router(admin_router)
    app.include_router(roasts_router)
    app.include_router(chat_router)
    app.include_router(profile_chat_router)
    app.include_router(ai_settings_router)

    # Landing page at /landing (clean URL)
    frontend_path = Path(__file__).parent.parent / "frontend"

    @app.get("/landing")
    async def landing_page():
        return FileResponse(str(frontend_path / "landing.html"))

    # Shared roast page (server-rendered for OG tags)
    @app.get("/roast/{share_id}")
    async def shared_roast_page(share_id: str, request: Request):
        from app.database.session import async_session_factory
        from app.services.roast.service import RoastService
        import json as json_mod

        svc = RoastService()
        try:
            async with async_session_factory() as db:
                roast = await svc.get_roast_by_share_id(db, share_id)
                # Extract all data inside session scope (ORM object detaches after)
                roast_data = roast.roast_data
                roast_share_id = roast.share_id
                roast_created_at = roast.created_at.isoformat()
                roast_id = roast.id
        except Exception:
            raise StarletteHTTPException(status_code=404, detail="Roast not found")

        roast_json = json_mod.dumps({
            "share_id": roast_share_id,
            "roast_data": roast_data,
            "created_at": roast_created_at,
        })
        # Escape </ to prevent </script> XSS breakout
        roast_json_safe = roast_json.replace("<", "\\u003c")

        rd = roast_data or {}
        score = rd.get("score", "?")
        headline = html_mod.escape(rd.get("headline", "Resume Roast"))
        base_url = str(request.base_url).rstrip("/")
        og_url = f"{base_url}/roast/{html_mod.escape(share_id)}"
        og_image = f"{base_url}/static/og-roast.png"
        og_title = f"My Resume Got Roasted: {score}/10"

        # Fire-and-forget analytics capture
        _ua = request.headers.get("user-agent")
        _referer = request.headers.get("referer")
        _ip = request.client.host if request.client else None

        async def _record_view():
            from app.models.roast_view import RoastView
            from app.services.roast.ua_parser import parse_user_agent

            parsed = parse_user_agent(_ua)
            async with async_session_factory() as view_db:
                view_db.add(RoastView(
                    roast_id=roast_id,
                    share_id=share_id,
                    ip_address=_ip,
                    user_agent=_ua,
                    referer=_referer,
                    platform=parsed["platform"],
                    os=parsed["os"],
                    browser=parsed["browser"],
                ))
                await view_db.commit()

        create_tracked_task(_record_view())

        template_path = frontend_path / "roast-share.html"
        template = template_path.read_text()
        rendered = (
            template
            .replace("{{OG_TITLE}}", og_title)
            .replace("{{OG_DESCRIPTION}}", headline)
            .replace("{{OG_IMAGE}}", og_image)
            .replace("{{OG_URL}}", og_url)
            .replace("{{ROAST_JSON}}", roast_json_safe)
            .replace("{{BASE_URL}}", base_url)
        )
        return HTMLResponse(content=rendered)

    # Static files for frontend
    if frontend_path.exists():
        app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")

    return app


app = create_app()
