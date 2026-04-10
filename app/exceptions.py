class ProfileNotFoundError(Exception):
    pass


class JobNotFoundError(Exception):
    pass


class LaTeXCompilationError(Exception):
    pass


class AIInferenceError(Exception):
    pass


class UsageLimitExceeded(Exception):
    pass


class AuthenticationError(Exception):
    pass


class ForbiddenError(Exception):
    pass


class RoastNotFoundError(Exception):
    pass


class NotFoundError(Exception):
    """Generic 404 for admin routes — use specific errors (JobNotFoundError, etc.) elsewhere."""
    pass


class BadRequestError(Exception):
    """Explicit 400 for validation errors in route handlers."""
    pass


class ConflictError(Exception):
    """409 — e.g. concurrent chat request on the same session."""
    pass


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
