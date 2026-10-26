from collections.abc import Iterator
from typing import Any


_QUOTA_MARKERS = (
    "resource_exhausted",
    "quota exceeded",
    "quota_exceeded",
    "insufficient_quota",
    "insufficient quota",
    "exceeded your current quota",
    "quota limit exceeded",
    "credit exhausted",
    "credits exhausted",
    "out of credits",
    "billing limit",
    "billing hard limit",
)


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _status_code(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def classify_provider_error(error: BaseException) -> tuple[str, str]:
    """Return a safe user-facing category and message for provider failures."""
    errors = list(_exception_chain(error))
    error_text = " ".join(
        f"{type(item).__name__} {item} {getattr(item, 'code', '')}"
        for item in errors
    ).lower()

    if any(marker in error_text for marker in _QUOTA_MARKERS) or any(
        _status_code(item) == 402 for item in errors
    ):
        return (
            "credits_exhausted",
            "API credits or quota are exhausted. Please wait for the quota to reset/refill, or check your provider account.",
        )

    if any(_status_code(item) == 429 for item in errors):
        return (
            "rate_limited",
            "The provider is receiving too many requests. Please wait briefly and try again.",
        )

    return "provider_error", "The request could not be completed. Please try again later."


def provider_error_payload(error: BaseException) -> dict[str, str]:
    code, message = classify_provider_error(error)
    return {"code": code, "detail": message}


def provider_result_error(error: BaseException) -> dict[str, str]:
    payload = provider_error_payload(error)
    return {"error_code": payload["code"], "error": payload["detail"]}


def provider_http_detail(error: BaseException) -> dict[str, str]:
    return provider_error_payload(error)
