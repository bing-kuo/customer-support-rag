import asyncio
import httpx
from app.main import app,sessions
from app import service
from app.knowledge import build,connect
from app.lifecycle import mark_ready,activate
from app.ollama import TurnTimeout
from tests.test_lifecycle import fixture_records


def test_timeout_keeps_session_conflict_until_upstream_ends(tmp_path,monkeypatch):
    path=tmp_path/'db';rid=build(path);db=connect(path);cp,ep,rp=fixture_records(tmp_path,rid,db)
    mark_ready(path,rid,cp,ep);activate(path,rid,None,rp)
    monkeypatch.setattr(service,'DB',path)
    async def scenario():
        upstream=asyncio.get_running_loop().create_future()
        async def timeout(session,*args,**kwargs):
            background=asyncio.get_running_loop().create_future()
            error=TurnTimeout('unit timeout');error.upstream=upstream
            background.set_exception(error);session['background']=background
            raise error
        monkeypatch.setattr(service,'respond',timeout)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://localhost') as client:
            sid=(await client.post('/api/v1/sessions',json={})).json()['session_id'];url=f'/api/v1/sessions/{sid}/turns'
            body={'client_turn_id':'one','message':'配對','product_id':'hp-a'}
            assert (await client.post(url,json=body)).status_code==504
            assert sessions[sid]['inflight']
            assert (await client.post(url,json=body)).json()['error']['code']=='turn_conflict'
            assert (await client.post(url,json=dict(body,message='different'))).json()['error']['code']=='idempotency_conflict'
            upstream.set_result(None);await asyncio.sleep(0)
            assert not sessions[sid]['inflight']
    asyncio.run(scenario())
