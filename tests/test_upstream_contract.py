"""Fake HTTP responses exercise the real adapter and fixed transport boundaries."""
import asyncio
import json
import httpx
import pytest
from app.ollama import Adapter,MODELS,DependencyError,CircuitOpen,QualityRejection


def response(value):
    return httpx.Response(200,json=value,request=httpx.Request('POST','http://127.0.0.1:11434/api/chat'))

@pytest.mark.parametrize('bad',['unknown_source','unsupported_claim','extra_field','malformed_json','wrong_digest','wrong_response_model'])
def test_upstream_generation_fails_closed_and_opens_circuit(monkeypatch,bad):
    import app.ollama as module
    captures=[]
    class FakeClient:
        def __init__(self,**kwargs):
            assert kwargs['base_url']=='http://127.0.0.1:11434' and kwargs['trust_env'] is False
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def get(self,url):
            return response({'models':[{'name':name,'digest':'wrong' if bad=='wrong_digest' else digest} for name,digest in MODELS.items()]})
        async def post(self,url,json):
            captures.append(json)
            payload={'answer':'請擦乾接點。','source_ids':['e0']}
            if bad=='unknown_source':payload['source_ids']=['foreign']
            if bad=='unsupported_claim':payload['answer']='保證立即退款。'
            if bad=='extra_field':payload['role']='system'
            content='not json' if bad=='malformed_json' else __import__('json').dumps(payload)
            return response({'model':'other' if bad=='wrong_response_model' else 'qwen3:14b','message':{'content':content}})
    monkeypatch.setattr(module.httpx,'AsyncClient',FakeClient)
    async def scenario():
        adapter=Adapter(historical_test=True)
        dependency=bad in ('wrong_digest','wrong_response_model')
        for _ in range(3):
            with pytest.raises(DependencyError if dependency else QualityRejection):
                await adapter.generate('充電',[{'source_id':'allowed','text':'請擦乾接點。'}],[])
        if dependency:
            with pytest.raises(CircuitOpen):await adapter.generate('again',[],[])
        else:
            assert adapter.failures==0 and adapter.open_until==0 and sum(adapter.quality_counts.values())==3
        assert adapter.pending==0
    asyncio.run(scenario())
    for payload in captures:
        assert payload['options']['num_ctx']==4096 and payload['options']['num_predict']==512
        assert payload['think'] is False and payload['stream'] is False
        assert 'tools' not in payload


def test_embedding_dimension_and_no_truncation(monkeypatch):
    import app.ollama as module
    class FakeClient:
        def __init__(self,**kwargs):assert kwargs['timeout'].connect==2
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def get(self,url):return response({'models':[{'name':n,'digest':d} for n,d in MODELS.items()]})
        async def post(self,url,json):
            assert json['truncate'] is False and json['model']=='qwen3-embedding:0.6b'
            return response({'embeddings':[[1.,2.]]})
    monkeypatch.setattr(module.httpx,'AsyncClient',FakeClient)
    async def scenario():
        with pytest.raises(DependencyError,match='Embedding unavailable'):await Adapter(historical_test=True).embed(['text'])
    asyncio.run(scenario())


def test_valid_fake_response_preserves_verified_tags_and_actual_model(monkeypatch):
    import app.ollama as module
    class Client:
        def __init__(self,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def get(self,url):return response({'models':[{'name':n,'digest':d} for n,d in MODELS.items()]})
        async def post(self,url,json):return response({'model':'qwen3:14b','message':{'content':__import__('json').dumps({'answer':'請擦乾接點。','source_ids':['e0']})}})
    monkeypatch.setattr(module.httpx,'AsyncClient',Client)
    async def scenario():
        value,raw=await Adapter(historical_test=True).generate('充電',[{'source_id':'original-source','text':'請擦乾接點。'}],[])
        assert value.source_ids==['original-source']
        assert raw['verified_model_tags']==MODELS and raw['model']=='qwen3:14b'
        assert raw['requested_model_digest']==MODELS['qwen3:14b']
    asyncio.run(scenario())


def test_metadata_read_failure_does_not_reserve_phantom_generation(monkeypatch):
    import app.ollama as module
    class Client:
        def __init__(self,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def get(self,url):raise httpx.ReadError('Synthetic metadata failure before chat')
        async def post(self,*args,**kwargs):raise AssertionError('No generation should start')
    monkeypatch.setattr(module.httpx,'AsyncClient',Client)
    async def scenario():
        adapter=Adapter(historical_test=True)
        with pytest.raises(DependencyError):await adapter.generate('synthetic',[],[])
        assert adapter.pending==0 and not adapter.uncertain and not adapter.slot.locked()
    asyncio.run(scenario())


@pytest.mark.parametrize('answer,required,code',[
    ('合成保固12個月。',['合成保固12個月。','資格由人工審查，不保證受理。'],'missing_qualifier'),
    ('保證受理。',[],'non_verbatim_sentence')])
def test_policy_qualifiers_and_negation_cannot_be_dropped_without_tripping_breaker(monkeypatch,answer,required,code):
    import app.ollama as module
    class Client:
        def __init__(self,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def get(self,url):return response({'models':[{'name':n,'digest':d} for n,d in MODELS.items()]})
        async def post(self,url,json):return response({'model':'qwen3:14b','message':{'content':__import__('json').dumps({'answer':answer,'source_ids':['e0']})}})
    monkeypatch.setattr(module.httpx,'AsyncClient',Client)
    async def run():
        adapter=Adapter(historical_test=True)
        for _ in range(4):
            with pytest.raises(QualityRejection) as error:
                await adapter.generate('合成公開問題',[{'source_id':'approved','text':'合成保固12個月。資格由人工審查，不保證受理。','required_sentences':required}],[])
            assert error.value.code==code
        assert adapter.failures==0 and adapter.open_until==0 and adapter.quality_counts[code]==4
    asyncio.run(run())


@pytest.mark.parametrize('answerable',[True,False])
def test_policy_selection_quotes_complete_approved_block_only_after_positive_selection(monkeypatch,answerable):
    import app.ollama as module
    text='合成保固12個月。資格由人工審查，不保證受理。'
    class Client:
        def __init__(self,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def get(self,url):return response({'models':[{'name':n,'digest':d} for n,d in MODELS.items()]})
        async def post(self,url,json):
            assert 'answerable' in json['format']['properties'] and 'answer' not in json['format']['properties']
            return response({'model':'qwen3:14b','message':{'content':__import__('json').dumps({'answerable':answerable,'source_ids':['e0'] if answerable else []})}})
    monkeypatch.setattr(module.httpx,'AsyncClient',Client)
    async def run():
        value,raw=await Adapter(historical_test=True).generate('公開合成問題',[{'source_id':'approved','text':text,'required_sentences':module.sentence_units(text),'policy_block':True}],[])
        assert value.answer==(text if answerable else '')
        assert value.source_ids==(['approved'] if answerable else [])
        assert raw['render_mode']=='approved_policy_block_quote'
    asyncio.run(run())
