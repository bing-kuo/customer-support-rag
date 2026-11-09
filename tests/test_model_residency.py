"""Synthetic model ownership and lifecycle tests; no live model calls."""
import asyncio
import json
import httpx
import pytest
from app.ollama import Adapter,MODELS,RESIDENCY,ResidencyError,CapacityError,DependencyError,CircuitOpen


def upstream(monkeypatch,*,initial=None,unload_sticks=False,unload_error=False,embed_error=False):
    import app.ollama as module
    models=list(initial or []);calls=[]
    def response(value):return httpx.Response(200,json=value,request=httpx.Request('POST','http://127.0.0.1:11434/api/chat'))
    class Client:
        def __init__(self,**kwargs):assert kwargs['trust_env'] is False
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def get(self,url):
            return response({'models':list(models) if url=='/api/ps' else [{'name':n,'digest':d} for n,d in MODELS.items()]})
        async def post(self,url,json):
            calls.append((url,json));model=json['model']
            if url=='/api/generate':
                if unload_error:raise httpx.ReadTimeout('Synthetic unload uncertainty')
                if not unload_sticks:models[:]=[m for m in models if m['name']!=model]
                return response({'done':True})
            if not any(m['name']==model for m in models):models.append({'name':model,'digest':MODELS[model]})
            if url=='/api/embed':
                if embed_error:raise httpx.ReadTimeout('Synthetic embedding uncertainty')
                return response({'model':model,'embeddings':[[1.]*1024 for t in json['input']]})
            return response({'model':model,'message':{'content':__import__('json').dumps({'answer':'核准合成句。','source_ids':['e0']})}})
    monkeypatch.setattr(module.httpx,'AsyncClient',Client)
    return models,calls


def test_owned_phases_unload_confirmed_and_finite_keep_alive(monkeypatch):
    models,calls=upstream(monkeypatch)
    async def run():
        a=Adapter(historical_test=True);await a.begin_owned_batch(exclusive_slot=True)
        await a.embed(['synthetic']);assert a.owned_models=={'qwen3-embedding:0.6b'}
        await a.unload_owned('qwen3-embedding:0.6b');assert not models
        await a.generate('synthetic',[{'source_id':'one','text':'核准合成句。'}],[])
        assert a.owned_models=={'qwen3:14b'}
        await a.finish_owned_batch()
        assert not models and not a.exclusive_batch and not a.owned_models and not a.pending
    asyncio.run(run())
    assert next(p for u,p in calls if u=='/api/embed')['keep_alive']=='60s'
    assert next(p for u,p in calls if u=='/api/chat')['keep_alive']=='60s'
    assert all(p['keep_alive']==0 and 'prompt' not in p for u,p in calls if u=='/api/generate')


@pytest.mark.parametrize('initial',[[{'name':'unrelated','digest':'external'}],[{'name':'qwen3:14b','digest':MODELS['qwen3:14b']} ]])
def test_preexisting_model_even_same_digest_is_not_owned(monkeypatch,initial):
    models,calls=upstream(monkeypatch,initial=initial)
    async def run():
        a=Adapter(historical_test=True)
        with pytest.raises(ResidencyError,match='lifecycle'):await a.begin_owned_batch(exclusive_slot=True)
        with pytest.raises(ResidencyError):await a.unload_owned(initial[0]['name'])
        assert not a.exclusive_batch and not a.maintenance and not calls
    asyncio.run(run())


def test_exclusive_assertion_required_and_external_appearance_never_unloaded(monkeypatch):
    models,calls=upstream(monkeypatch)
    async def run():
        a=Adapter(historical_test=True)
        with pytest.raises(ResidencyError):await a.begin_owned_batch()
        await a.begin_owned_batch(exclusive_slot=True);await a.embed(['synthetic'])
        models.append({'name':'external-job','digest':'unrelated'})
        with pytest.raises(ResidencyError) as error:await a.unload_owned('qwen3-embedding:0.6b')
        assert error.value.code=='external_model_residency' and a.residency_uncertain
        assert all(u!='/api/generate' for u,p in calls)
        with pytest.raises(CircuitOpen):await a.generate('x',[],[])
        assert a.failures==0 and not a.pending
    asyncio.run(run())


@pytest.mark.parametrize('attribute,value',[('pending',1),('embedding_pending',1),('uncertain',True),('maintenance',True)])
def test_unload_refuses_queued_started_embedding_and_unknown_completion(monkeypatch,attribute,value):
    models,calls=upstream(monkeypatch)
    async def run():
        a=Adapter(historical_test=True);await a.begin_owned_batch(exclusive_slot=True);await a.embed(['synthetic'])
        setattr(a,attribute,value)
        with pytest.raises(ResidencyError):await a.unload_owned('qwen3-embedding:0.6b')
        assert not [u for u,p in calls if u=='/api/generate']
    asyncio.run(run())


@pytest.mark.parametrize('mode',['unload_sticks','unload_error'])
def test_unconfirmed_unload_blocks_next_generation_without_phantom_counts(monkeypatch,mode):
    models,calls=upstream(monkeypatch,**{mode:True})
    async def run():
        a=Adapter(historical_test=True);await a.begin_owned_batch(exclusive_slot=True);await a.embed(['synthetic'])
        with pytest.raises(ResidencyError):await a.unload_owned('qwen3-embedding:0.6b')
        with pytest.raises(CircuitOpen):await a.generate('x',[],[])
        assert a.pending==0 and a.embedding_pending==0 and a.failures==0 and a.lifecycle_failures
        assert not [u for u,p in calls if u=='/api/chat']
    asyncio.run(run())


def test_embedding_transport_uncertainty_prevents_unload_or_generation(monkeypatch):
    models,calls=upstream(monkeypatch,embed_error=True)
    async def run():
        a=Adapter(historical_test=True);await a.begin_owned_batch(exclusive_slot=True)
        with pytest.raises(DependencyError):await a.embed(['synthetic'])
        assert a.uncertain and a.embedding_pending==0 and a.dependency_counts['embedding_transport_error']==1
        with pytest.raises(ResidencyError):await a.finish_owned_batch()
        with pytest.raises(CircuitOpen):await a.generate('x',[],[])
        assert a.failures==0 and a.pending==0
    asyncio.run(run())


def test_residency_config_changes_invalidate_configuration_without_budget_change(monkeypatch):
    from app.knowledge import configuration,digest
    from app.lifecycle import canonical
    before=configuration();original=digest(canonical(before))
    from app.cloud_config import POLICY
    monkeypatch.setitem(POLICY,'pricing_revision','unit-changed-revision')
    after=configuration()
    assert digest(canonical(after))!=original
    assert 'num_ctx' not in after['generation'] and after['generation']['model']=='gpt-6-luna'
    assert before['generation']['max_output_tokens']==after['generation']['max_output_tokens']==512
    assert before['embedding']['truncate'] is False and before['embedding']['model_digest']==MODELS['qwen3-embedding:0.6b']


def test_expired_owned_model_is_not_reloaded_just_to_unload(monkeypatch):
    models,calls=upstream(monkeypatch)
    async def run():
        a=Adapter(historical_test=True);await a.begin_owned_batch(exclusive_slot=True);await a.embed(['synthetic'])
        models.clear();await a.unload_owned('qwen3-embedding:0.6b')
        assert not a.owned_models and not [u for u,p in calls if u=='/api/generate']
    asyncio.run(run())


def test_embedding_and_maintenance_block_generation_before_admission():
    async def run():
        for attribute in ['embedding_pending','maintenance']:
            a=Adapter(historical_test=True);setattr(a,attribute,1)
            with pytest.raises(CapacityError):await a.generate('x',[],[])
            assert a.pending==0
    asyncio.run(run())


def test_lexical_serving_does_not_load_embedding(tmp_path,monkeypatch):
    from tests.test_policy_behavior import setup,state,payload
    from app import service
    release,captured=setup(tmp_path,monkeypatch)
    async def forbidden(*args,**kwargs):raise AssertionError('Lexical served turns must not embed')
    monkeypatch.setattr(service.adapter,'embed',forbidden)
    result=asyncio.run(service.respond(state(),payload('如何配對','hp-a'),'unit',release,offline=True))
    assert result['status']=='answer' and captured
