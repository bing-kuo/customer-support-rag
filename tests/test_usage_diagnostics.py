"""Documented synthetic usage shapes; never actual provider responses or secrets."""
import asyncio,copy,json,sqlite3
from dataclasses import replace
from datetime import datetime,timedelta,timezone
import httpx,pytest
from app.cloud import GenerationAdapter
from app.budget import Ledger,BudgetUnavailable,safe_diagnostic
from app.cloud_config import SetupUnavailable,fingerprint
from app.ollama import QualityRejection,TurnTimeout
from tests.test_cloud_offline import config,reply,EVIDENCE


def raw_usage():
    return dict(input_tokens=100,output_tokens=20,total_tokens=120,input_tokens_details=dict(cached_tokens=10,cache_write_tokens=70,cache_write_tokens_details=dict(ttl_30m_tokens=30,ttl_12h_tokens=40)),output_tokens_details=dict(reasoning_tokens=5))


@pytest.mark.parametrize('variant',['ttl','aggregate','zero','omitted','null'])
def test_documented_cache_shapes_and_unknown_optional_not_zero(variant):
    u=raw_usage()
    if variant=='aggregate':del u['input_tokens_details']['cache_write_tokens_details']
    if variant=='zero':u['input_tokens_details']=dict(cached_tokens=0,cache_write_tokens=0,cache_write_tokens_details=dict(ttl_30m_tokens=0,ttl_12h_tokens=0))
    if variant=='omitted':u.pop('input_tokens_details');u.pop('output_tokens_details')
    if variant=='null':u['input_tokens_details']=None;u['output_tokens_details']=None
    parsed=GenerationAdapter.usage({'usage':u});assert parsed.input_tokens==100
    if variant in ('omitted','null'):assert parsed.cached_tokens is None and parsed.cache_write_tokens is None and parsed.reasoning_tokens is None
    if variant=='aggregate':assert parsed.ttl_30m_tokens is None and parsed.cache_write_tokens==70
    if variant=='ttl':assert parsed.ttl_30m_tokens+parsed.ttl_12h_tokens==parsed.cache_write_tokens


@pytest.mark.parametrize('case',['bool','negative','float','nan','huge','total','overlap','ttl_mismatch','ttl_bool','ttl_partial','new_input','new_ttl','new_output','new_usage','details_list','individual_null','missing_usage'])
def test_invalid_usage_never_settles_and_safe_code_survives_restart(tmp_path,case):
    u=raw_usage();body=reply(usage=u)
    if case=='bool':u['input_tokens']=True
    if case=='negative':u['output_tokens']=-1
    if case=='float':u['input_tokens']=100.0
    if case=='nan':u['input_tokens']=float('nan')
    if case=='huge':u['input_tokens']=2**100
    if case=='total':u['total_tokens']=121
    if case=='overlap':u['input_tokens_details']['cached_tokens']=90
    if case=='ttl_mismatch':u['input_tokens_details']['cache_write_tokens_details']['ttl_12h_tokens']=41
    if case=='ttl_bool':u['input_tokens_details']['cache_write_tokens_details']['ttl_12h_tokens']=False
    if case=='ttl_partial':del u['input_tokens_details']['cache_write_tokens_details']['ttl_12h_tokens']
    if case=='new_input':u['input_tokens_details']['hostile-private-fee']=0
    if case=='new_ttl':u['input_tokens_details']['cache_write_tokens_details']['hostile-private-fee']=0
    if case=='new_output':u['output_tokens_details']['hostile-private-fee']=0
    if case=='new_usage':u['hostile-private-fee']=0
    if case=='details_list':u['input_tokens_details']=[]
    if case=='individual_null':u['input_tokens_details']['cached_tokens']=None
    if case=='missing_usage':body.pop('usage')
    # reply() retains the provided mutable dictionary.
    c=config();calls=[]
    def handler(r):calls.append(r);return httpx.Response(200,json=body,headers={'x-request-id':'req_synthetic'})
    a=GenerationAdapter(config_loader=lambda:c,ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(handler))
    async def run():
        with pytest.raises(SetupUnavailable):await a.generate('合成問題',EVIDENCE,[],request_id='synthetic',scope='developer_smoke')
    # JSON intentionally cannot encode NaN via httpx; build the synthetic response bytes.
    if case=='nan':a.transport=httpx.MockTransport(lambda r:httpx.Response(200,content=json.dumps(body).encode()))
    asyncio.run(run());assert a.ledger.status()['states']=={'unknown':1}
    persisted=Ledger(tmp_path/'ledger');diag=persisted.diagnostics('synthetic')
    assert diag[-1]['stage']=='accounting_or_dependency_rejected' and diag[-1]['code']!='none'
    serialized=json.dumps(diag);assert 'hostile-private-fee' not in serialized and c.key not in serialized and '合成問題' not in serialized
    with pytest.raises(BudgetUnavailable):persisted.reserve(c,'second','b'*64,100,512,'developer_smoke')


@pytest.mark.parametrize('terminal',['completed','incomplete','failed','cancelled','refusal','schema','qualifier'])
def test_accounting_precedes_terminal_quality_status_and_preserves_components(tmp_path,terminal):
    c=config();u=raw_usage();body=reply(usage=u)
    if terminal in ('incomplete','failed','cancelled'):body['status']=terminal
    if terminal=='refusal':body['output'][1]['content']=[{'type':'refusal','refusal':'never log this mock body'}]
    if terminal=='schema':body['output']=[]
    evidence=EVIDENCE
    if terminal=='qualifier':
        evidence=[dict(source_id='s',text='請擦乾接點。請停止使用。',required_sentences=['請停止使用。'])]
        body=reply(json.dumps({'answerable':True,'source_ids':['e0'],'answer':'請擦乾接點。'}),usage=u)
    a=GenerationAdapter(config_loader=lambda:c,ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(lambda r:httpx.Response(200,json=body)))
    async def run():
        if terminal=='completed':await a.generate('合成問題',evidence,[],request_id='quality',scope='developer_smoke')
        else:
            with pytest.raises((QualityRejection,SetupUnavailable)):await a.generate('合成問題',evidence,[],request_id='quality',scope='developer_smoke')
    asyncio.run(run());assert a.ledger.status()['states']=={'settled':1} and not a.uncertain
    diag=Ledger(tmp_path/'ledger').diagnostics('quality')
    accounting=[r for r in diag if r['stage']=='accounting_settled']
    assert accounting and accounting[0]['usage']['ttl_12h_tokens']==40
    assert accounting[0]['usage']['cache_write_tokens']==70
    assert diag[-1]['stage'] in ('application_completed','application_rejected','accounting_or_dependency_rejected')
    assert 'never log' not in json.dumps(diag)
    with a.ledger.db() as db:assert db.execute('SELECT used_cost FROM requests').fetchone()[0]==c.price(100,20)


def test_settlement_storage_failure_stays_unknown_and_diagnostic_failure_does_not_mask(tmp_path,monkeypatch):
    c=config();a=GenerationAdapter(config_loader=lambda:c,ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(lambda r:httpx.Response(200,json=reply(usage=raw_usage()))))
    a.preflight('developer_smoke')
    def broken(*args,**kwargs):raise BudgetUnavailable('settlement_storage_failed')
    monkeypatch.setattr(a.ledger,'settle',broken)
    async def run():
        with pytest.raises(SetupUnavailable) as failure:await a.generate('合成',EVIDENCE,[],request_id='storage',scope='developer_smoke')
        assert failure.value.code=='settlement_storage_failed'
    asyncio.run(run());assert a.ledger.status()['states']=={'unknown':1}
    assert Ledger(tmp_path/'ledger').diagnostics('storage')[-1]['code']=='settlement_storage_failed'
    # A failing diagnostics channel must not replace the original safe classification.
    monkeypatch.setattr(a.ledger,'diagnostic',lambda *a:(_ for _ in ()).throw(RuntimeError('mock private storage details')))
    a.record('storage','accounting_or_dependency_rejected','usage_missing')
    assert a.diagnostic_write_failed and a.last_diagnostic['code']=='usage_missing'
    assert a.ledger.status()['states']=={'unknown':1}


def test_crash_after_sent_preserves_stage_and_blocks_foreign_admission(tmp_path):
    c=config();l=Ledger(tmp_path/'ledger');l.reserve(c,'crashed','a'*64,100,512,'developer_smoke');l.sent('crashed')
    l.diagnostic('crashed',safe_diagnostic('sent','none',{}))
    again=Ledger(tmp_path/'ledger');assert again.diagnostics('crashed')[0]['stage']=='sent'
    with pytest.raises(BudgetUnavailable):again.reserve(c,'second','b'*64,100,512,'developer_smoke')
    assert again.status()['states']=={'sent':1}


def test_safe_diagnostic_never_persists_unknown_values_or_names():
    d=safe_diagnostic('http_received','private-secret-code',dict(provider_status={'private':'secret'},service_tier='mock-private-tier',usage_fields=['input_tokens','private-fee-name'],usage={'input_tokens':100,'private-fee-name':123},elapsed_seconds=float('nan'),key='mock-secret'))
    text=json.dumps(d);assert 'private' not in text and 'mock-secret' not in text and 'elapsed_seconds' not in d
    assert d['code']=='unclassified_failure' and d['usage_fields']==['input_tokens']


def test_atomic_settlement_sql_failure_and_restart_preserve_unknown(tmp_path):
    c=config();a=GenerationAdapter(config_loader=lambda:c,ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(lambda r:httpx.Response(200,json=reply(usage=raw_usage()))))
    a.preflight('developer_smoke')
    with a.ledger.db() as db:db.execute("CREATE TRIGGER mock_disk_failure BEFORE UPDATE OF state ON requests WHEN NEW.state='settled' BEGIN SELECT RAISE(ABORT,'private-storage-details'); END")
    async def run():
        with pytest.raises(SetupUnavailable) as failure:await a.generate('合成',EVIDENCE,[],request_id='atomic',scope='developer_smoke')
        assert failure.value.code=='ledger_schema_or_storage_unavailable'
    asyncio.run(run());again=Ledger(tmp_path/'ledger');assert again.status()['states']=={'unknown':1}
    assert again.diagnostics('atomic')[-1]['code']=='ledger_schema_or_storage_unavailable'
    with again.db() as db:assert db.execute('SELECT used_input,used_cost FROM requests').fetchone()==(None,None)


def test_timeout_diagnostic_survives_late_terminal_settlement_without_double_release(tmp_path):
    c=config();gate=None
    async def handler(r):await gate.wait();return httpx.Response(200,json=reply(usage=raw_usage()))
    a=GenerationAdapter(config_loader=lambda:c,ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(handler),deadline=.01)
    async def run():
        nonlocal gate
        gate=asyncio.Event()
        with pytest.raises(TurnTimeout) as failure:await a.generate('合成',EVIDENCE,[],request_id='timeout',scope='developer_smoke')
        assert a.ledger.status()['states']=={'unknown':1}
        assert any(d['code']=='logical_timeout' for d in a.ledger.diagnostics('timeout'))
        gate.set();await failure.value.upstream
        assert a.ledger.status()['states']=={'settled':1} and not a.uncertain and a.pending==0
        with a.ledger.db() as db:assert db.execute("SELECT count(*) FROM events WHERE event='settled_usage_upper_bound'").fetchone()[0]==1
    asyncio.run(run())


def test_explicit_null_service_tier_remains_unknown_without_contract_proof(tmp_path):
    c=config();a=GenerationAdapter(config_loader=lambda:c,ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(lambda r:httpx.Response(200,json=reply(service_tier=None,usage=raw_usage()))))
    async def run():
        with pytest.raises(SetupUnavailable):await a.generate('合成',EVIDENCE,[],request_id='tier-null',scope='developer_smoke')
    asyncio.run(run());assert a.ledger.status()['states']=={'unknown':1}
    assert a.ledger.diagnostics('tier-null')[-1]['code']=='unverified_service_tier'


def test_initial_diagnostic_storage_failure_stops_before_transport_and_retains_marker(tmp_path,monkeypatch):
    c=config();calls=[];a=GenerationAdapter(config_loader=lambda:c,ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(lambda r:calls.append(r)))
    a.preflight('developer_smoke')
    monkeypatch.setattr(a.ledger,'diagnostic',lambda *a:(_ for _ in ()).throw(RuntimeError('mock private storage')))
    async def run():
        with pytest.raises(SetupUnavailable) as failure:await a.generate('合成',EVIDENCE,[],request_id='diagnostic-failure',scope='developer_smoke')
        assert failure.value.code=='diagnostic_storage_failed'
    asyncio.run(run());assert calls==[] and a.diagnostic_write_failed and a.ledger.status()['states']=={'unknown':1}
    assert a.last_diagnostic['code']=='diagnostic_storage_failed'
