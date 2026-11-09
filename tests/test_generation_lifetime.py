"""Exercise queue ownership without contacting any inference service."""
import asyncio
import json
import pytest
from app.ollama import Adapter,CapacityError,TurnTimeout


def test_cancelled_caller_keeps_upstream_slot_until_completion(monkeypatch):
    async def scenario():
        adapter=Adapter(historical_test=True);started=asyncio.Event();finish=asyncio.Event();entered=0
        async def run(prompt,evidence):
            nonlocal entered
            try:
                async with adapter.slot:
                    asyncio.current_task().upstream_started=True
                    entered+=1;started.set();await finish.wait()
                    return __import__('app.ollama',fromlist=['Generated']).Generated(answer='',source_ids=[]),{}
            finally:adapter.pending-=1
        monkeypatch.setattr(adapter,'_run',run)
        caller=asyncio.create_task(adapter.generate('question',[],[]));await started.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):await caller
        assert adapter.pending==1 and entered==1 and adapter.slot.locked()
        second=asyncio.create_task(adapter.generate('question2',[],[]))
        third=asyncio.create_task(adapter.generate('question3',[],[]))
        await asyncio.sleep(0)
        with pytest.raises(CapacityError):await adapter.generate('fourth',[],[])
        assert entered==1
        finish.set();await second;await third
        await asyncio.sleep(0)
        assert adapter.pending==0 and entered==3
    asyncio.run(scenario())


def test_logical_timeout_preserves_upstream_and_slot(monkeypatch):
    async def scenario():
        adapter=Adapter(historical_test=True);started=asyncio.Event();finish=asyncio.Event()
        original=asyncio.wait_for
        async def bounded(awaitable,timeout):return await original(awaitable,.01)
        monkeypatch.setattr(asyncio,'wait_for',bounded)
        async def run(prompt,evidence):
            try:
                async with adapter.slot:
                    asyncio.current_task().upstream_started=True
                    started.set();await finish.wait()
                    return __import__('app.ollama',fromlist=['Generated']).Generated(answer='',source_ids=[]),{}
            finally:adapter.pending-=1
        monkeypatch.setattr(adapter,'_run',run)
        with pytest.raises(TurnTimeout):await adapter.generate('question',[],[])
        assert started.is_set() and adapter.pending==1 and adapter.slot.locked()
        finish.set();await asyncio.sleep(.01)
        assert adapter.pending==0
    asyncio.run(scenario())


def test_queued_deadline_never_starts_background_inference(monkeypatch):
    async def scenario():
        adapter=Adapter(historical_test=True);adapter.slot=asyncio.Semaphore(0);entered=0
        original=asyncio.wait_for
        async def bounded(awaitable,timeout):return await original(awaitable,.01)
        monkeypatch.setattr(asyncio,'wait_for',bounded)
        async def run(prompt,evidence):
            nonlocal entered
            try:
                async with adapter.slot:
                    asyncio.current_task().upstream_started=True
                    entered+=1
            finally:adapter.pending-=1
        monkeypatch.setattr(adapter,'_run',run)
        with pytest.raises(TurnTimeout):await adapter.generate('queued',[],[])
        await asyncio.sleep(.01)
        adapter.slot.release();await asyncio.sleep(0)
        assert entered==0 and adapter.pending==0
    asyncio.run(scenario())
