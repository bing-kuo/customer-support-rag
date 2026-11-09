"""Real service cancellation paths with a synthetic upstream, no inference/network."""
import asyncio
import pytest
from app import service
from app.main import TurnInput,monitored_turn
from app.knowledge import build,connect
from app.lifecycle import mark_ready,activate
from app.ollama import Adapter,Generated,DependencyError
from tests.test_lifecycle import fixture_records


def prepare(tmp_path,monkeypatch):
    path=tmp_path/'db';release=build(path);db=connect(path)
    cp,ep,rp=fixture_records(tmp_path,release,db)
    mark_ready(path,release,cp,ep);activate(path,release,None,rp)
    monkeypatch.setattr(service,'DB',path)
    return release


@pytest.mark.parametrize('mode',['cancel','disconnect','timeout'])
def test_service_cancel_queued_never_generates_or_leaks_queue_capacity(tmp_path,monkeypatch,mode):
    release=prepare(tmp_path,monkeypatch)
    async def scenario():
        actual_disconnect=mode=='disconnect'
        if mode=='timeout':
            original=asyncio.wait_for
            async def bounded(awaitable,timeout):return await original(awaitable,.01 if timeout==75 else timeout)
            monkeypatch.setattr(asyncio,'wait_for',bounded)
        adapter=Adapter(historical_test=True);adapter.slot=asyncio.Semaphore(0);adapter.pending=1
        queued=asyncio.Event();disconnected=asyncio.Event();started=[]
        async def upstream(prompt,evidence):
            try:
                queued.set()
                async with adapter.slot:
                    asyncio.current_task().upstream_started=True;started.append(True)
                    return Generated(answer=evidence[0]['text'],source_ids=['e0']),{}
            finally:adapter.pending-=1
        monkeypatch.setattr(adapter,'_run',upstream);monkeypatch.setattr(service,'adapter',adapter)
        session={'product':None,'history':[],'clarifications':0}
        payload=TurnInput(client_turn_id='synthetic',message='配對',product_id='hp-a')
        class Request:
            async def is_disconnected(self):return disconnected.is_set()
        call=asyncio.create_task(monitored_turn(Request(),session,payload,'unit',release) if actual_disconnect else service.respond(session,payload,'unit',release))
        await asyncio.wait_for(queued.wait(),1)
        assert adapter.pending==2
        if actual_disconnect:disconnected.set()
        elif mode=='cancel':call.cancel()
        expected=DependencyError if actual_disconnect else asyncio.TimeoutError if mode=='timeout' else asyncio.CancelledError
        with pytest.raises(expected):
            await asyncio.wait_for(call,1)
        assert adapter.pending==1
        adapter.slot.release();await asyncio.sleep(.01)
        assert not started and adapter.pending==1
    asyncio.run(scenario())


def test_service_cancel_started_keeps_slot_until_real_upstream_finishes(tmp_path,monkeypatch):
    release=prepare(tmp_path,monkeypatch)
    async def scenario():
        adapter=Adapter(historical_test=True);started=asyncio.Event();finish=asyncio.Event()
        async def upstream(prompt,evidence):
            try:
                async with adapter.slot:
                    asyncio.current_task().upstream_started=True;started.set();await finish.wait()
                    return Generated(answer=evidence[0]['text'],source_ids=['e0']),{}
            finally:adapter.pending-=1
        monkeypatch.setattr(adapter,'_run',upstream);monkeypatch.setattr(service,'adapter',adapter)
        session={'product':None,'history':[],'clarifications':0}
        payload=TurnInput(client_turn_id='started',message='配對',product_id='hp-a')
        call=asyncio.create_task(service.respond(session,payload,'unit',release))
        await asyncio.wait_for(started.wait(),1);call.cancel()
        with pytest.raises(asyncio.CancelledError):await call
        assert adapter.pending==1 and adapter.slot.locked() and not session['background'].done()
        finish.set();await session['background']
        assert adapter.pending==0 and not adapter.slot.locked()
    asyncio.run(scenario())


def test_started_disconnect_and_unknown_upstream_end_keeps_reservation(tmp_path,monkeypatch):
    import httpx
    import app.ollama as module
    release=prepare(tmp_path,monkeypatch)
    async def scenario():
        adapter=Adapter(historical_test=True);started=asyncio.Event();fail=asyncio.Event();posts=[]
        class Client:
            def __init__(self,**kwargs):pass
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            async def get(self,url):
                return httpx.Response(200,json={'models':[{'name':n,'digest':d} for n,d in module.MODELS.items()]},request=httpx.Request('GET','http://127.0.0.1:11434/api/tags'))
            async def post(self,url,json):
                posts.append(url);started.set();await fail.wait()
                raise httpx.ReadError('Synthetic uncertain upstream end')
        monkeypatch.setattr(module.httpx,'AsyncClient',Client);monkeypatch.setattr(service,'adapter',adapter)
        session={'product':None,'history':[],'clarifications':0}
        payload=TurnInput(client_turn_id='uncertain',message='配對',product_id='hp-a')
        call=asyncio.create_task(service.respond(session,payload,'unit',release))
        await asyncio.wait_for(started.wait(),1);call.cancel()
        with pytest.raises(asyncio.CancelledError):await call
        assert adapter.pending==1 and adapter.slot.locked()
        fail.set()
        with pytest.raises(DependencyError):await session['background']
        assert adapter.uncertain and adapter.pending==1
        with pytest.raises(module.CircuitOpen):await adapter.generate('again',[],[])
        assert len(posts)==1
    asyncio.run(scenario())
