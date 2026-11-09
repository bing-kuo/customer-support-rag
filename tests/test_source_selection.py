"""Cloud chooses current eligible sources; server preserves exact approved facts."""
import asyncio,json,stat
from dataclasses import replace
import httpx,pytest
from app.cloud import GenerationAdapter,SELECTION_SYSTEM
from app.ollama import QualityRejection
from app import service
from app.providers import ProviderAdapter
from app.knowledge import build,connect
from app.lifecycle import schema
from app.main import TurnInput
from scripts.public_rejection_capture import PublicSyntheticCapture
from scripts.source_selection_retest import plan
from tests.test_cloud_offline import config,reply


def selected(ids=('e0',),**extra):return json.dumps(dict(answerable=bool(ids),source_ids=list(ids),**extra),ensure_ascii=False)


def test_source_choice_not_static_first_mapping_and_exact_renderer(tmp_path):
    evidence=[{'source_id':'pairing','text':'先按盒按鈕3秒。'},{'source_id':'cleaning','text':'使用乾布。不可浸泡。','required_sentences':['不可浸泡。']}]
    c=config()
    def handler(request):
        body=json.loads(request.content);prompt=json.loads(body['input'][0]['content'][0]['text'])
        assert body['instructions']==SELECTION_SYSTEM and prompt['response_mode']=='source_selection'
        assert set(body['text']['format']['schema']['properties'])=={'answerable','source_ids'}
        return httpx.Response(200,json=reply(selected(('e1',))))
    a=GenerationAdapter(config_loader=lambda:c,ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(handler))
    value,metrics=asyncio.run(a.generate('如何清潔？',evidence,[],request_id='choice',scope='developer_smoke'))
    assert value.answer=='使用乾布。不可浸泡。' and value.source_ids==['cleaning']
    assert metrics['answer_origin']=='server_approved_text' and metrics['model_output_kind']=='source_selection' and metrics['render_mode']=='approved_procedural_source_quote'


@pytest.mark.parametrize('bad',[selected(('foreign-source',)),selected(('e9',)),selected(answer='自行新增不支援句子。'),selected(product_id='hp-b'),json.dumps({'answerable':True,'source_ids':[]}),json.dumps({'answerable':False,'source_ids':['e0']})])
def test_wrong_source_or_unsupported_added_answer_never_rendered(tmp_path,bad):
    c=config();a=GenerationAdapter(config_loader=lambda:c,ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(lambda r:httpx.Response(200,json=reply(bad))))
    async def run():
        with pytest.raises(QualityRejection):await a.generate('配對', [{'source_id':'hp-a-source','text':'按盒按鈕3秒。'}],[],request_id='reject',scope='developer_smoke')
    asyncio.run(run());assert a.ledger.status()['states']=={'settled':1}


@pytest.mark.parametrize('ids',[(),('e1',)])
def test_critical_omission_rejected_even_selection_false_or_other_source(tmp_path,ids):
    c=config();evidence=[{'source_id':'critical','text':'請停止使用並聯絡人工。','required_sentences':['請停止使用並聯絡人工。'],'critical':True},{'source_id':'ordinary','text':'請擦乾接點。'}]
    a=GenerationAdapter(config_loader=lambda:c,ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(lambda r:httpx.Response(200,json=reply(selected(ids)))))
    async def run():
        with pytest.raises(QualityRejection) as failure:await a.generate('異常發熱',evidence,[],request_id='critical',scope='developer_smoke')
        assert failure.value.code=='missing_qualifier'
    asyncio.run(run())


def test_policy_eligibility_complete_and_model_cannot_replace_qualifier(tmp_path):
    c=config();text='合成保固12個月。資格由人工判斷。不承諾退款或受理。'
    a=GenerationAdapter(config_loader=lambda:c,ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(lambda r:httpx.Response(200,json=reply(selected()))))
    value,metrics=asyncio.run(a.generate('一般保固？',[{'source_id':'general','text':text,'required_sentences':['資格由人工判斷。','不承諾退款或受理。'],'policy_block':True}],[],request_id='qualified',scope='developer_smoke'))
    assert value.answer==text and metrics['render_mode']=='approved_policy_block_quote'


def setup_service(tmp_path,monkeypatch,wrong_product=False):
    path=tmp_path/'knowledge';release=build(path);db=connect(path);schema(db);db.close()
    a=ProviderAdapter();c=replace(config(),scopes=('public_eval',));prompts=[]
    def handler(request):
        body=json.loads(request.content);prompt=json.loads(body['input'][0]['content'][0]['text']);prompts.append(prompt)
        ids=('hp-a-pairing:2',) if wrong_product and 'hp-b' in prompt['evidence'][0]['text'] else ('e0',)
        return httpx.Response(200,json=reply(selected(ids)))
    a.generation=GenerationAdapter(config_loader=lambda:c,ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(handler));a.preflight('public_eval')
    monkeypatch.setattr(service,'DB',path);monkeypatch.setattr(service,'adapter',a)
    return release,a,prompts


def test_true_multiround_null_inherits_then_explicit_switch_uses_different_approved_steps(tmp_path,monkeypatch):
    release,a,prompts=setup_service(tmp_path,monkeypatch);session={'product':None,'history':[],'clarifications':0}
    async def turn(key,product):return await service.respond(session,TurnInput(client_turn_id=key,message='如何配對？',product_id=product),key,release,offline=True,on_date='2026-11-08')
    async def run():
        first=await turn('prior-a','hp-a');assert '盒按鈕3秒' in first['answer'] and '觸控區5秒' not in first['answer']
        inherited=await turn('null-a',None);assert all(c['doc_id']=='hp-a-pairing' for c in inherited['citations'])
        switched=await turn('switch-b','hp-b');assert '觸控區5秒' in switched['answer'] and '盒按鈕3秒' not in switched['answer']
        assert all(c['doc_id']=='hp-b-pairing' for c in switched['citations']) and session['product']=='hp-b'
        assert prompts[-1]['history']==[] and all('hp-a 合成配對' not in e['text'] for e in prompts[-1]['evidence'])
    asyncio.run(run());assert a.generation.ledger.status()['states']=={'settled':3}


def test_wrong_previous_product_source_on_switch_is_safe_handoff(tmp_path,monkeypatch):
    release,a,prompts=setup_service(tmp_path,monkeypatch,wrong_product=True);session={'product':None,'history':[],'clarifications':0}
    async def run():
        first=await service.respond(session,TurnInput(client_turn_id='prior',message='如何配對？',product_id='hp-a'),'a',release,offline=True)
        assert first['status']=='answer'
        second=await service.respond(session,TurnInput(client_turn_id='switch',message='如何配對？',product_id='hp-b'),'b',release,offline=True)
        assert second['status']=='handoff' and not second['citations'] and second['answer'] is None
        assert session['quality_code']=='unknown_citation' and session['product']=='hp-b'
    asyncio.run(run())


@pytest.mark.parametrize('scope,sensitive,huge',[('public_eval',False,False),('holdout',False,False),('public_eval',True,False),('public_eval',False,True)])
def test_restricted_synthetic_capture_never_logs_secrets_or_private_holdout(tmp_path,scope,sensitive,huge):
    c=replace(config(),scopes=(scope,));capture=PublicSyntheticCapture(plan(),root=tmp_path/'capture');capture.select_case('dev-public-v2-07')
    raw_text=selected(answer='unapproved synthetic addition')
    if sensitive:raw_text=selected(answer=c.key)
    if huge:raw_text=selected(answer='x'*3000)
    a=GenerationAdapter(config_loader=lambda:c,ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(lambda r:httpx.Response(200,json=reply(raw_text))),rejection_capture=capture)
    async def run():
        with pytest.raises(QualityRejection):await a.generate('公開合成配對',[{'source_id':'synthetic-source','text':'先按盒按鈕3秒。'}],[],request_id='capture',scope=scope)
    asyncio.run(run())
    with a.ledger.db() as db:assert 'unapproved synthetic addition' not in '\n'.join(db.iterdump())
    target=tmp_path/'capture'/'capture.json'
    if scope=='holdout':assert not target.exists()
    else:
        data=target.read_bytes();value=json.loads(data)
        assert len(data)<=4096 and c.key not in data.decode() and stat.S_IMODE(target.stat().st_mode)==0o600
        assert value['raw_capture_suppressed']==(sensitive or huge)
        if not sensitive and not huge:assert 'unapproved synthetic addition' in value['rejected_output_text']
        assert 'raw' not in json.dumps(a.last_diagnostic) and a.last_diagnostic['candidate_aliases']==['e0']


def test_retest_frozen_public_gold_max_four_and_no_calls():
    value=plan()
    from app.lifecycle import canonical
    import hashlib
    assert PublicSyntheticCapture(value).manifest_hash==hashlib.sha256(canonical(value)).hexdigest()
    assert value['maximum_provider_calls']==4 and sum(value['expected_call_budget'].values())==4
    switch=value['cases'][0];assert switch['id']=='dev-public-v2-07' and switch['prior_turns'][0]['product_id']=='hp-a' and switch['product_id']=='hp-b'


@pytest.mark.parametrize('unsupported',['合成保固12個月。','保證退款。','可忽略人工審查。'])
def test_policy_missing_or_negated_qualifier_prose_is_rejected_not_used(tmp_path,unsupported):
    c=config();text='合成保固12個月。資格由人工判斷。不承諾退款或受理。'
    a=GenerationAdapter(config_loader=lambda:c,ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(lambda r:httpx.Response(200,json=reply(selected(answer=unsupported)))))
    async def run():
        with pytest.raises(QualityRejection) as failure:await a.generate('保固資格',[{'source_id':'general','text':text,'required_sentences':['資格由人工判斷。','不承諾退款或受理。'],'policy_block':True}],[],request_id='qualifier-reject',scope='developer_smoke')
        assert failure.value.code=='invalid_model_schema'
    asyncio.run(run());assert a.ledger.status()['states']=={'settled':1}


def test_future_retest_adapter_hard_four_attempts_and_public_capture_context(monkeypatch):
    from scripts.source_selection_retest import RetestAdapter
    a=RetestAdapter();a.generation_scope='public_eval';a.case='dev-public-v2-07'
    async def synthetic(*args,**kwargs):return None,{}
    monkeypatch.setattr(ProviderAdapter,'generate',synthetic)
    async def run():
        for _ in range(4):await a.generate('synthetic',[],[])
        from app.cloud_config import SetupUnavailable
        with pytest.raises(SetupUnavailable) as failure:await a.generate('synthetic',[],[])
        assert failure.value.code=='retest_request_limit' and len(a.calls)==4 and a.capture.case=='dev-public-v2-07'
    asyncio.run(run())
