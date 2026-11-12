"""HTTP API application.

Importing this module does not read an API key, open a database, or send a
question. ``GET /health``, ``GET /docs``, ``GET /``, and ``GET /assets/*`` stay
local. ``POST /ask`` and ``POST /assist`` can retrieve or call Luna only after
the body is valid and the Host/Origin checks pass. ``/assist`` does not call a
model while a required field is still missing.
"""

import json
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.responses import Response

from customer_support_rag import __version__
from customer_support_rag.ask import (
    AskConfig,
    AskFailure,
    AskResponse,
    CitedSource,
    UsageCounts,
    answer_ask,
    config_from_environment,
)
from customer_support_rag.assist import (
    AssistError,
    AssistView,
    SlotCoordinator,
    handoff_summary,
)
from customer_support_rag.local_http import (
    forwarded_host_present,
    host_is_trusted,
    origin_is_same,
)
from customer_support_rag.web_pages import (
    INDEX_CONTENT_TYPE,
    NOSNIFF,
    asset_file,
    default_web_root,
    index_file,
)

APPLICATION_NAME = "customer-support-rag"
MAX_BODY_BYTES = 16 * 1024
_MAX_QUESTION_CHARS = 1000


class _FromEnvironment:
    """Marker for ``create_app()`` so an explicit ``None`` stays unconfigured."""


_FROM_ENVIRONMENT = _FromEnvironment()


class HealthResponse(BaseModel):
    status: Literal["ok"]
    application: str
    version: str


class AskError(BaseModel):
    """A fixed failure code. The body does not echo the question or a path."""

    model_config = ConfigDict(extra="forbid")

    code: str
    request_id: str


class _AskInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    product_id: Literal["hp-a", "hp-b"]
    question: str


class _AssistInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["start", "continue", "reset"]
    slot_token: str | None = None
    product_id: Literal["hp-a", "hp-b"] | None = None
    question: str | None = None
    steps_tried: str | None = None


class AssistResponse(BaseModel):
    """One clarify, answer, or handoff step. User fields are not verified."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: Literal["clarify", "answer", "handoff"]
    slot_token: str
    attempt: int
    missing: list[str]
    product_id: Literal["hp-a", "hp-b"] | None
    question: str | None
    steps_tried: str | None
    summary: str | None
    user_statements_unverified: Literal[True]
    answer: str | None
    reason: str | None
    sources: list[CitedSource]
    usage: UsageCounts | None
    usage_code: str | None
    usage_fields: list[str]


def create_app(
    config: AskConfig | None | _FromEnvironment = _FROM_ENVIRONMENT,
    *,
    web_root: Path | None = None,
    extra_hosts: tuple[str, ...] = (),
) -> FastAPI:
    """Build the app. The default reads process config and does not open it.

    Production Host checks accept loopback only. ``extra_hosts`` is for an
    explicit testserver configuration. ``web_root`` is process config; a
    request cannot replace it.
    """
    selected = (
        config_from_environment() if isinstance(config, _FromEnvironment) else config
    )
    pages = default_web_root() if web_root is None else web_root
    application = FastAPI(title=APPLICATION_NAME)
    application.state.ask_config = selected
    application.state.web_root = pages
    application.state.extra_hosts = extra_hosts
    application.state.slots = SlotCoordinator()

    @application.middleware("http")
    async def guard_local_http(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if forwarded_host_present(request.headers) or not host_is_trusted(
            request.headers.get("host"), request.app.state.extra_hosts
        ):
            return _rejected(request)
        if request.method == "POST" and request.url.path in {"/ask", "/assist"}:
            origin = request.headers.get("origin")
            host_header = request.headers.get("host", "")
            scheme = str(request.scope.get("scheme", "http"))
            if origin is not None and not origin_is_same(origin, scheme, host_header):
                return _rejected(request)
        return await call_next(request)

    @application.get("/health", response_model=HealthResponse)
    def get_health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            application=APPLICATION_NAME,
            version=__version__,
        )

    @application.get("/", include_in_schema=False)
    def get_home() -> Response:
        path = index_file(request_web_root(application))
        if path is None:
            return _missing()
        return FileResponse(path, media_type=INDEX_CONTENT_TYPE, headers=NOSNIFF)

    @application.get("/assets/{name}", include_in_schema=False)
    def get_asset(name: str) -> Response:
        found = asset_file(request_web_root(application), name)
        if found is None:
            return _missing()
        path, content_type = found
        return FileResponse(path, media_type=content_type, headers=NOSNIFF)

    @application.post(
        "/ask",
        response_model=AskResponse,
        responses={
            413: {"model": AskError},
            422: {"model": AskError},
            429: {"model": AskError},
            502: {"model": AskError},
            503: {"model": AskError},
            504: {"model": AskError},
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {"schema": _AskInput.model_json_schema()}
                },
            }
        },
    )
    async def post_ask(request: Request) -> AskResponse | JSONResponse:
        request_id = f"ask-{uuid.uuid4().hex}"
        try:
            product_id, question = await _parse_ask(request)
            return await answer_ask(
                request.app.state.ask_config,
                product_id,
                question,
                request_id,
            )
        except AskFailure as failure:
            return JSONResponse(
                status_code=failure.status_code,
                content={"code": failure.code, "request_id": request_id},
            )

    @application.post(
        "/assist",
        response_model=AssistResponse,
        responses={
            413: {"model": AskError},
            422: {"model": AskError},
            429: {"model": AskError},
            502: {"model": AskError},
            503: {"model": AskError},
            504: {"model": AskError},
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {"schema": _AssistInput.model_json_schema()}
                },
            }
        },
    )
    async def post_assist(request: Request) -> AssistResponse | JSONResponse:
        request_id = f"assist-{uuid.uuid4().hex}"
        try:
            incoming = await _parse_assist(request)
            view = _slot_view(request, incoming)
            if view.needs_model:
                if view.product_id is None or view.question is None:
                    raise AssistError("invalid_request")
                answered = await answer_ask(
                    request.app.state.ask_config,
                    view.product_id,
                    view.question,
                    request_id,
                )
                return _assist_from_ask(request_id, view, answered)
            return _assist_from_view(request_id, view)
        except AssistError as exc:
            status_code = {"slot_capacity": 503}.get(exc.code, 422)
            return JSONResponse(
                status_code=status_code,
                content={"code": exc.code, "request_id": request_id},
            )
        except AskFailure as failure:
            return JSONResponse(
                status_code=failure.status_code,
                content={"code": failure.code, "request_id": request_id},
            )

    return application


app = create_app()


def request_web_root(application: FastAPI) -> Path:
    """Process web root stored at startup. Requests cannot replace this path."""
    return Path(application.state.web_root)


def _rejected(request: Request) -> JSONResponse:
    prefix = {"/ask": "ask", "/assist": "assist"}.get(request.url.path, "req")
    request_id = f"{prefix}-{uuid.uuid4().hex}"
    return JSONResponse(
        status_code=422,
        content={"code": "invalid_request", "request_id": request_id},
    )


def _missing() -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": "Not Found"})


async def _parse_ask(request: Request) -> tuple[str, str]:
    if not _json_content_type(request.headers.get("content-type")):
        raise AskFailure(422, "invalid_request")
    raw = await _read_limited(request)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise AskFailure(422, "invalid_request") from None
    except ValueError as exc:
        if not _integer_conversion_limit(exc):
            raise
        raise AskFailure(422, "invalid_request") from None
    try:
        parsed = _AskInput.model_validate(payload)
    except ValidationError:
        raise AskFailure(422, "invalid_request") from None
    question = parsed.question
    if (
        not question.strip()
        or len(question) > _MAX_QUESTION_CHARS
        or not _utf8(question)
    ):
        raise AskFailure(422, "invalid_request")
    return parsed.product_id, question


async def _read_limited(request: Request) -> bytes:
    claimed = _content_length(request.headers.get("content-length"))
    if claimed is not None and claimed > MAX_BODY_BYTES:
        raise AskFailure(413, "body_too_large")
    size = 0
    parts: list[bytes] = []
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            raise AskFailure(413, "body_too_large")
        parts.append(chunk)
    return b"".join(parts)


def _content_length(value: str | None) -> int | None:
    if value is None:
        return None
    if not value.isascii() or not value.isdigit() or len(value) > 10:
        raise AskFailure(422, "invalid_request")
    return int(value)


def _json_content_type(value: str | None) -> bool:
    if value is None:
        return False
    return value.split(";", 1)[0].strip().lower() == "application/json"


def _integer_conversion_limit(exc: ValueError) -> bool:
    """True only for CPython's digit limit while decoding a JSON number.

    Other ``ValueError`` values stay uncaught so a parser bug is not reported
    as a client input error.
    """
    return type(exc) is ValueError and "integer string conversion" in str(exc)


async def _parse_assist(request: Request) -> _AssistInput:
    if not _json_content_type(request.headers.get("content-type")):
        raise AskFailure(422, "invalid_request")
    raw = await _read_limited(request)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise AskFailure(422, "invalid_request") from None
    except ValueError as exc:
        if not _integer_conversion_limit(exc):
            raise
        raise AskFailure(422, "invalid_request") from None
    try:
        return _AssistInput.model_validate(payload)
    except ValidationError:
        raise AskFailure(422, "invalid_request") from None


def _slot_view(request: Request, incoming: _AssistInput) -> AssistView:
    coordinator: SlotCoordinator = request.app.state.slots
    fields = {
        "product_id": incoming.product_id,
        "question": incoming.question,
        "steps_tried": incoming.steps_tried,
    }
    if incoming.action == "start":
        if incoming.slot_token is not None:
            raise AssistError("invalid_request")
        return coordinator.start(**fields)
    if incoming.slot_token is None:
        raise AssistError("invalid_slot")
    if incoming.action == "reset":
        if any(value is not None for value in fields.values()):
            raise AssistError("invalid_request")
        return coordinator.reset(incoming.slot_token)
    return coordinator.continue_slot(incoming.slot_token, **fields)


def _assist_from_view(request_id: str, view: AssistView) -> AssistResponse:
    status = "clarify" if view.status == "clarify" else "handoff"
    return AssistResponse(
        request_id=request_id,
        status=status,
        slot_token=view.token,
        attempt=view.attempt,
        missing=list(view.missing),
        product_id=view.product_id,
        question=view.question,
        steps_tried=view.steps_tried,
        summary=view.summary,
        user_statements_unverified=True,
        answer=None,
        reason=view.reason,
        sources=[],
        usage=None,
        usage_code=None,
        usage_fields=[],
    )


def _assist_from_ask(
    request_id: str, view: AssistView, answered: AskResponse
) -> AssistResponse:
    summary = None
    if answered.status == "handoff":
        summary = handoff_summary(
            product_id=view.product_id,
            question=view.question,
            steps_tried=view.steps_tried,
            missing=(),
            reason=answered.reason or "no_evidence",
        )
    return AssistResponse(
        request_id=request_id,
        status=answered.status,
        slot_token=view.token,
        attempt=view.attempt,
        missing=[],
        product_id=view.product_id,
        question=view.question,
        steps_tried=view.steps_tried,
        summary=summary,
        user_statements_unverified=True,
        answer=answered.answer,
        reason=answered.reason,
        sources=answered.sources,
        usage=answered.usage,
        usage_code=answered.usage_code,
        usage_fields=answered.usage_fields,
    )


def _utf8(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True
