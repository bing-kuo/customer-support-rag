"""Bounded candidate integration, explicit MockTransport; never actual project credentials."""
import asyncio,json
from dataclasses import replace
import httpx,pytest
from scripts import public_dev10 as stage
from app import service
from app.cloud import GenerationAdapter
from app.cloud_config import SetupUnavailable
from app.knowledge import build,connect
from app.lifecycle import schema
from tests.test_cloud_offline import config,reply


def test_frozen_selection_exact_public_cases_and_no_gold_rewrite():
    manifest=stage.selected_manifest()
    assert len(manifest['cases'])==10 and manifest['maximum_provider_calls']==10 and not manifest['gold_changed']
    original=json.loads((stage.FROZEN/'dev.json').read_bytes())
    assert all(q in original for q in manifest['cases'])
    assert any(q.get('prior_turns') for q in manifest['cases'])


def test_application_guard_counts_attempted_errors_and_stops_at_ten(monkeypatch):
    a=stage.BoundedPublicAdapter();a.generation_scope='public_eval'
    async def failure(*args):raise SetupUnavailable('synthetic_error')
    monkeypatch.setattr(stage.ProviderAdapter,'generate',failure)
    async def run():
        for _ in range(10):
            with pytest.raises(SetupUnavailable):await a.generate('synthetic',[],[])
        with pytest.raises(SetupUnavailable) as exc:await a.generate('synthetic',[],[])
        assert exc.value.code=='stage_request_limit' and len(a.calls)==10
    asyncio.run(run())


def test_real_candidate_service_with_mock_cloud_lexical_and_zero_cloud_paths(tmp_path,monkeypatch):
    path=tmp_path/'knowledge';release=build(path);db=connect(path);schema(db);db.close()
    a=stage.BoundedPublicAdapter();c=replace(config(),scopes=('public_eval',));requests=[]
    def handler(r):
        requests.append(r);body=json.loads(r.content);prompt=json.loads(body['input'][0]['content'][0]['text'])
        text=json.dumps({'answerable':True,'source_ids':['e0']}) if prompt['response_mode'] in ('policy_selection','source_selection') else json.dumps({'answer':prompt['evidence'][0]['text'],'source_ids':['e0']},ensure_ascii=False)
        return httpx.Response(200,json=reply(text))
    a.generation=GenerationAdapter(config_loader=lambda:c,ledger_path=tmp_path/'budget',transport=httpx.MockTransport(handler));a.preflight('public_eval')
    monkeypatch.setattr(service,'DB',path);monkeypatch.setattr(service,'adapter',a)
    data=stage.selected_manifest()['cases'];ids={'dev-normal-00','dev-public-v2-02','dev-missing_info-00','dev-normal-07','dev-public-v2-06'}
    result,stopped=asyncio.run(stage.execute_cases([q for q in data if q['id'] in ids],release,'a'*64,a,tmp_path/'checkpoint'))
    assert not stopped and all(q['status_matches'] for q in result)
    assert len(requests)==3
    warranty=next(q for q in result if q['id']=='dev-public-v2-02')['turns'][-1]['response']
    assert '資格與結果由人工判斷' in warranty['answer'] and '不承諾退款或保固受理' in warranty['answer']
    safety=next(q for q in result if q['id']=='dev-normal-07')['turns'][-1]
    assert safety['provider_admission_attempts']==0 and '停止使用並聯絡人工' in safety['response']['answer']
    slot=next(q for q in result if q['id']=='dev-public-v2-06')['turns']
    assert slot[0]['response']['status']=='clarify' and slot[-1]['product_slot']=='hp-b'
    assert all('hp-b' in c['doc_id'] for c in slot[-1]['response']['citations'])


def test_candidate_stage_stops_on_first_safe_error_without_processing_rest(tmp_path,monkeypatch):
    a=stage.BoundedPublicAdapter();a.generation_scope='public_eval'
    async def unavailable(*args,**kwargs):raise SetupUnavailable('usage_missing')
    monkeypatch.setattr(service,'respond',unavailable)
    cases=stage.selected_manifest()['cases']
    rows,stop=asyncio.run(stage.execute_cases(cases,'a'*64,'b'*64,a,tmp_path/'checkpoint'))
    assert stop and len(rows)==1 and rows[0]['turns'][0]['error_code']=='usage_missing'


def test_maximum_ten_preflight_strict_cap_and_pending_are_zero_network(tmp_path):
    from app.budget import Ledger
    c=replace(config(),scopes=('public_eval',));l=Ledger(tmp_path/'ledger')
    assert stage.budget_preflight(c,l)
    assert not stage.budget_preflight(replace(c,max_requests=9),l)
    assert not stage.budget_preflight(replace(c,cap_units=10*c.price(3584,512)),l)
    l.reserve(c,'pending','a'*64,100,512,'public_eval')
    assert not stage.budget_preflight(c,l)
    assert l.status()['states']=={'reserved':1}


def test_unknown_even_when_service_masks_dependency_stops_stage(tmp_path,monkeypatch):
    a=stage.BoundedPublicAdapter();a.generation.uncertain=True
    async def unavailable(*args,**kwargs):return {'status':'unavailable','answer':None,'citations':[]}
    monkeypatch.setattr(service,'respond',unavailable)
    cases=stage.selected_manifest()['cases']
    rows,stop=asyncio.run(stage.execute_cases(cases,'a'*64,'b'*64,a,tmp_path/'checkpoint'))
    assert stop and len(rows)==1
