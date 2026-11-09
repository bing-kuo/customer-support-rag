from datetime import datetime, timedelta, timezone
from typing import Literal, Optional
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import asyncio
import sqlite3
import os
LOCAL_PORT=int(os.environ.get("SUPPORT_RAG_PORT","8000"))
if not 1024<=LOCAL_PORT<=65535:
    raise ValueError("Invalid local port")
ORIGINS={None,f"http://localhost:{LOCAL_PORT}",f"http://127.0.0.1:{LOCAL_PORT}"}
from app import service
from app.lifecycle import pin
from app.ollama import DependencyError, CapacityError, BudgetError, TurnTimeout


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class SessionInput(StrictModel):
    pass


class TurnInput(StrictModel):
    client_turn_id: str = Field(min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_-]+$')
    message: str = Field(min_length=1, max_length=2000)
    product_id: Optional[Literal['hp-a', 'hp-b']] = None


from app.contracts import Session, Turn, Source, Health

from app.contracts import Error
RESPONSES={code:{'model':Error} for code in [400,404,409,413,422,429,503,504]}
app = FastAPI(title='Synthetic customer support demonstration',responses=RESPONSES)

sessions = {}


def error(code, status, request_id):
    return JSONResponse({'error': {'code': code, 'message': 'Request cannot be completed.'}, 'request_id': request_id}, status_code=status, headers={'Retry-After':'30'} if status==429 else None)


@app.middleware('http')
async def boundary(request: Request, call_next):
    request.state.request_id = uuid4().hex
    if request.headers.get('host','').split(':')[0] not in ('localhost','127.0.0.1','testserver'):
        return error('invalid_request',400,request.state.request_id)
    if request.headers.get('origin') not in ORIGINS:
        return error('invalid_request', 400, request.state.request_id)
    body = bytearray()
    async for part in request.stream():
        body.extend(part)
        if len(body) > 16384:
            return error('payload_too_large', 413, request.state.request_id)
    request._body = bytes(body)
    try:
        return await call_next(request)
    except Exception:
        return error('dependency_unavailable',503,request.state.request_id)


from fastapi.exceptions import RequestValidationError


@app.exception_handler(RequestValidationError)
async def invalid(request, exc):
    return error('validation_error', 422, request.state.request_id)


@app.post('/api/v1/sessions', status_code=201, response_model=Session)
def create_session(payload: SessionInput, request: Request):
    now = datetime.now(timezone.utc)
    for key in list(sessions):
        if sessions[key]['expires'] <= now:
            sessions.pop(key)
    if len(sessions) >= 1000:
        return error('capacity_exceeded', 429, request.state.request_id)
    sid = uuid4().hex
    sessions[sid] = dict(expires=now + timedelta(minutes=30),product=None,history=[],clarifications=0,completed={},inflight=False)
    return {'session_id': sid, 'expires_at': sessions[sid]['expires'].isoformat(), 'request_id': request.state.request_id}


async def monitored_turn(request,session,payload,request_id,release):
    """Cancel queued work on an actual ASGI disconnect as well as task cancellation."""
    stopped=asyncio.Event()
    async def disconnected():
        while not stopped.is_set():
            if await request.is_disconnected():return
            try:await asyncio.wait_for(stopped.wait(),timeout=.05)
            except asyncio.TimeoutError:pass
    work=asyncio.create_task(service.respond(session,payload,request_id,release))
    monitor=asyncio.create_task(disconnected())
    try:
        done,_=await asyncio.wait({work,monitor},return_when=asyncio.FIRST_COMPLETED)
        if work in done:return await work
        raise DependencyError('Client disconnected')
    finally:
        stopped.set()
        monitor.cancel()
        if not work.done():work.cancel()
        await asyncio.gather(work,monitor,return_exceptions=True)


@app.post('/api/v1/sessions/{sid}/turns', response_model=Turn)
async def turn(sid: str, payload: TurnInput, request: Request):
    session=sessions.get(sid)
    if not session or session['expires'] <= datetime.now(timezone.utc):
        return error('session_not_found',404,request.state.request_id)
    key=payload.client_turn_id
    body=payload.model_dump()
    if key in session['completed']:
        old,result=session['completed'][key]
        if old!=body:
            return error('idempotency_conflict',409,request.state.request_id)
        if any(service.withdrawn(c['doc_id']) for c in result['citations']):
            return error('dependency_unavailable',503,request.state.request_id)
        return result
    if session['inflight']:
        previous=session.get('inflight_payload')
        if previous and previous['client_turn_id']==key and previous!=body:
            return error('idempotency_conflict',409,request.state.request_id)
        return error('turn_conflict',409,request.state.request_id)
    if len(session['completed'])>=20:
        return error('session_limit',409,request.state.request_id)
    try:
        release=pin(service.DB)
    except (ValueError,OSError,sqlite3.Error):
        return error('dependency_unavailable',503,request.state.request_id)
    if not release:
        return error('no_active_release',503,request.state.request_id)
    session['inflight']=True
    session['inflight_payload']=body
    session.pop('background',None)
    try:
        result=await monitored_turn(request,session,payload,request.state.request_id,release)
        result=Turn.model_validate(result).model_dump()
        session['completed'][key]=(body,result)
        session['expires']=datetime.now(timezone.utc)+timedelta(minutes=30)
        return result
    except CapacityError:
        return error('capacity_exceeded',429,request.state.request_id)
    except BudgetError:
        return error('invalid_request',400,request.state.request_id)
    except (asyncio.TimeoutError,TurnTimeout):
        return error('turn_timeout',504,request.state.request_id)
    except (DependencyError,ValueError,OSError,sqlite3.Error):
        return error('dependency_unavailable',503,request.state.request_id)
    finally:
        def settled(future):
            if future.cancelled():
                session['inflight']=False
                return
            exc=future.exception()
            upstream=getattr(exc,'upstream',None)
            if upstream and not upstream.done():
                upstream.add_done_callback(settled)
            else:
                session['inflight']=False
        background=session.get('background')
        if background:
            if background.done():settled(background)
            else:background.add_done_callback(settled)
        else:
            session['inflight']=False


@app.get('/api/v1/sources/{source_id}', response_model=Source)
def source(source_id: str, request: Request):
    import re
    if not re.fullmatch(r'[a-f0-9]{128}',source_id):
        return error('source_not_found',404,request.state.request_id)
    try:
        value=service.source_value(source_id[:64],source_id[64:])
        return value if value else error('source_not_found',404,request.state.request_id)
    except (ValueError,OSError,sqlite3.Error):
        return error('source_not_found',404,request.state.request_id)


@app.get('/api/v1/health', response_model=Health)
async def health(request: Request):
    try:
        pin(service.DB)
    except (ValueError,OSError,sqlite3.Error):
        return error('dependency_unavailable',503,request.state.request_id)
    import httpx
    from app.providers import adapter
    available='unavailable'
    try:
        async with httpx.AsyncClient(base_url='http://127.0.0.1:11434',trust_env=False,timeout=httpx.Timeout(2,connect=2)) as client:
            await adapter.verify(client)
        available='ok'
    except (httpx.HTTPError,ValueError,DependencyError):
        pass
    import time
    generation='unavailable' if not adapter.generation.ready or adapter.uncertain or adapter.open_until>time.monotonic() else available
    return {'status':'ok' if generation=='ok' and pin(service.DB) else 'degraded','components':{'storage':'ok','embedding':available,'generation':generation},'request_id':request.state.request_id}


@app.exception_handler(HTTPException)
async def http_error(request, exc):
    return error('invalid_request',exc.status_code,request.state.request_id)


# Only explicit static assets are served; API misses keep JSON error responses.
web=Path(__file__).resolve().parents[1]/'web'
if web.exists():
    app.mount('/static',StaticFiles(directory=web),name='static')
    from fastapi.responses import FileResponse
    @app.get('/')
    def home():
        return FileResponse(web/'index.html')
    for asset in ['app.js','contract.js','style.css','demo-fixtures.json']:
        def factory(name):
            def asset_response():
                return FileResponse(web/name)
            return asset_response
        app.add_api_route('/'+asset,factory(asset),methods=['GET'],include_in_schema=False)
