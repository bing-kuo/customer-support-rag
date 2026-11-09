"""Mock-only provider and synthetic accounting units; no credentials or paid calls."""
import asyncio,json,sqlite3
from dataclasses import replace
from datetime import datetime,timedelta,timezone
from concurrent.futures import ThreadPoolExecutor
import httpx
import pytest
from app.cloud import GenerationAdapter
from app.cloud_config import CloudConfig,SetupUnavailable,MODEL,config_from_mapping,load_config
from app.budget import Ledger,BudgetUnavailable
from app.ollama import QualityRejection,TurnTimeout,BudgetError


def config(**changes):
    # Fictional accounting units, not approved funding or pricing.
    cfg=CloudConfig(key='mock-only-credential-never-real',budget_id='unit-budget',scopes=('developer_smoke',),expires_at=datetime.now(timezone.utc)+timedelta(hours=1),pricing_expires_at=datetime.now(timezone.utc)+timedelta(hours=1),max_requests=20,max_input_tokens=100000,max_output_tokens=20000,max_total_tokens=120000,cap_units=10**9,input_rate_units=10**6,output_rate_units=10**6,extra_units=0)
    return replace(cfg,**changes)

EVIDENCE=[{'source_id':'unit-source','text':'請擦乾接點。'}]

def reply(text=None,**changes):
    raw={'id':'resp_unit','model':MODEL,'status':'completed','usage':{'input_tokens':100,'output_tokens':20,'total_tokens':120,'input_tokens_details':{'cached_tokens':10},'output_tokens_details':{'reasoning_tokens':0}},'output':[{'type':'reasoning'},{'type':'message','role':'assistant','status':'completed','content':[{'type':'output_text','text':text or json.dumps({'answerable':True,'source_ids':['e0']},ensure_ascii=False)}]}]}
    raw.update(changes);return raw


def adapter(tmp_path,handler,cfg=None,**kwargs):
    cfg=cfg or config()
    return GenerationAdapter(config_loader=lambda :cfg,ledger_path=tmp_path/'budget.sqlite3',transport=httpx.MockTransport(handler),**kwargs)


@pytest.mark.parametrize('mutation',['missing','scope','expiry','pricing_expiry','cap','request_cap','input_cap','output_cap','token_cap'])
def test_cloud_missing_or_exhausted_approval_never_reaches_transport(tmp_path,mutation):
    calls=[];cfg=config()
    if mutation=='scope':cfg=replace(cfg,scopes=('serve',))
    if mutation=='expiry':cfg=replace(cfg,expires_at=datetime.now(timezone.utc)-timedelta(seconds=1))
    if mutation=='pricing_expiry':cfg=replace(cfg,pricing_expires_at=datetime.now(timezone.utc)-timedelta(seconds=1))
    if mutation=='cap':cfg=replace(cfg,cap_units=1)
    if mutation=='request_cap':cfg=replace(cfg,max_requests=0)
    if mutation=='input_cap':cfg=replace(cfg,max_input_tokens=1)
    if mutation=='output_cap':cfg=replace(cfg,max_output_tokens=1)
    if mutation=='token_cap':cfg=replace(cfg,max_total_tokens=1)
    a=adapter(tmp_path,lambda req:calls.append(req),cfg)
    if mutation=='missing':a.config_loader=lambda :(_ for _ in ()).throw(SetupUnavailable())
    async def run():
        with pytest.raises(SetupUnavailable):await a.generate('如何清潔？',EVIDENCE,[],request_id='unit',scope='developer_smoke')
    asyncio.run(run());assert calls==[] and a.pending==0


def test_official_request_shape_usage_identity_and_redacted_ledger(tmp_path):
    cfg=config();calls=[]
    def handler(request):
        calls.append(request);body=json.loads(request.content)
        assert str(request.url)=='https://api.openai.com/v1/responses'
        assert body['model']==MODEL and body['reasoning']=={'effort':'none'}
        assert body['store'] is False and body['background'] is False and body['stream'] is False
        assert not set(body)&{'tools','conversation','previous_response_id','temperature','think','options','num_ctx'}
        assert body['max_output_tokens']==512
        schema=body['text']['format'];assert schema['type']=='json_schema' and schema['strict'] is True and schema['schema']['additionalProperties'] is False
        assert 'unit-source' not in request.content.decode() and cfg.key not in request.content.decode()
        return httpx.Response(200,json=reply(),headers={'x-request-id':'req_unit'})
    a=adapter(tmp_path,handler,cfg)
    async def run():
        value,metrics=await a.generate('如何清潔？',EVIDENCE,[],request_id='unit',scope='developer_smoke')
        assert value.answer=='請擦乾接點。' and value.source_ids==['unit-source']
        assert metrics['requested_model']==MODEL and metrics['input_tokens']==100
        with pytest.raises(BudgetUnavailable):await a.generate('如何清潔？',EVIDENCE,[],request_id='unit',scope='developer_smoke')
    asyncio.run(run());assert len(calls)==1
    db=sqlite3.connect(tmp_path/'budget.sqlite3');dump='\n'.join(db.iterdump());db.close()
    assert cfg.key not in dump and '如何清潔' not in dump and '請擦乾' not in dump
    assert 'req_unit' in dump and 'resp_unit' in dump and a.ledger.status()['states']=={'settled':1}


@pytest.mark.parametrize('case',['refusal','incomplete','extra','unknown_source','negation','qualifier','critical','bad_output'])
def test_quality_rejection_is_charged_once_without_transport_breaker_or_body_exposure(tmp_path,case):
    evidence=EVIDENCE
    raw=reply()
    if case=='refusal':raw['output'][1]['content']=[{'type':'refusal','refusal':'secret provider body'}]
    if case=='incomplete':raw['status']='incomplete'
    if case=='extra':raw=reply(json.dumps({'answer':'請擦乾接點。','source_ids':['e0'],'extra':'secret'}))
    if case=='unknown_source':raw=reply(json.dumps({'answer':'請擦乾接點。','source_ids':['foreign']}))
    if case=='negation':evidence=[{'source_id':'s','text':'不保證受理。'}];raw=reply(json.dumps({'answer':'保證受理。','source_ids':['e0']}))
    if case in ('qualifier','critical'):
        evidence=[{'source_id':'s','text':'合成期限12個月。資格由人工審查。','required_sentences':['合成期限12個月。','資格由人工審查。'],'critical':case=='critical'}]
        raw=reply(json.dumps({'answer':'合成期限12個月。','source_ids':['e0']}))
    if case=='bad_output':raw['output']=[None]
    a=adapter(tmp_path,lambda request:httpx.Response(200,json=raw))
    async def run():
        with pytest.raises(QualityRejection) as failure:await a.generate('合成問題',evidence,[],request_id='unit',scope='developer_smoke')
        assert 'secret' not in str(failure.value)
    asyncio.run(run())
    assert a.failures==0 and sum(a.quality_counts.values())==1 and a.ledger.status()['states']=={'settled':1}


@pytest.mark.parametrize('case',['http401','http429','http503','missing_usage','wrong_model','unknown_billing','network','bad_json'])
def test_unknown_cost_survives_restart_and_stops_all_further_requests(tmp_path,case):
    calls=[]
    def handler(request):
        calls.append(request)
        if case=='network':raise httpx.ReadError('untrusted body with credentials',request=request)
        if case.startswith('http'):return httpx.Response(int(case[4:]),text='secret raw error')
        if case=='bad_json':return httpx.Response(200,text='secret not-json')
        raw=reply()
        if case=='missing_usage':raw.pop('usage')
        if case=='wrong_model':raw['model']='other-model'
        if case=='unknown_billing':raw['usage']['output_tokens_details']['unpriced_feature']=1
        return httpx.Response(200,json=raw)
    cfg=config();a=adapter(tmp_path,handler,cfg)
    async def run():
        with pytest.raises(__import__('app.ollama',fromlist=['CapacityError']).CapacityError if case=='http429' else SetupUnavailable) as exc:await a.generate('合成問題',EVIDENCE,[],request_id='unit',scope='developer_smoke')
        assert 'secret' not in str(exc.value)
        b=adapter(tmp_path,handler,cfg)
        with pytest.raises(SetupUnavailable):await b.generate('合成問題',EVIDENCE,[],request_id='next',scope='developer_smoke')
    asyncio.run(run());assert len(calls)==1 and a.ledger.status()['states']=={'unknown':1}


def test_queued_cancel_releases_only_unsent_reservation(tmp_path):
    async def run():
        started=asyncio.Event();finish=asyncio.Event();calls=[]
        async def handler(req):calls.append(req);started.set();await finish.wait();return httpx.Response(200,json=reply())
        a=adapter(tmp_path,handler)
        first=asyncio.create_task(a.generate('合成',EVIDENCE,[],request_id='first',scope='developer_smoke'));await started.wait()
        queued=asyncio.create_task(a.generate('合成',EVIDENCE,[],request_id='queued',scope='developer_smoke'));await asyncio.sleep(.01);queued.cancel()
        with pytest.raises(asyncio.CancelledError):await queued
        assert a.ledger.status()['states']=={'released':1,'sent':1} and a.pending==1
        finish.set();await first;assert len(calls)==1 and a.pending==0
    asyncio.run(run())


@pytest.mark.parametrize('kind',['cancel','timeout'])
def test_sent_cancel_timeout_retains_unknown_until_confirmed_terminal(tmp_path,kind):
    async def run():
        started=asyncio.Event();finish=asyncio.Event()
        async def handler(req):started.set();await finish.wait();return httpx.Response(200,json=reply())
        a=adapter(tmp_path,handler,deadline=.03 if kind=='timeout' else 60)
        caller=asyncio.create_task(a.generate('合成',EVIDENCE,[],request_id='first',scope='developer_smoke'));await started.wait()
        if kind=='cancel':caller.cancel()
        with pytest.raises(asyncio.CancelledError if kind=='cancel' else TurnTimeout):await caller
        assert a.ledger.status()['states']=={'unknown':1} and a.pending==1 and a.uncertain
        finish.set();await caller.upstream
        assert a.pending==0 and not a.uncertain and a.ledger.status()['states']=={'settled':1}
    asyncio.run(run())


def test_reservation_atomic_concurrency_restart_and_policy_mutation(tmp_path):
    cfg=config(max_requests=1);ledger=Ledger(tmp_path/'ledger.sqlite3')
    def compete(i):
        try:ledger.reserve(cfg,'r'+str(i),'a'*64,100,512,'developer_smoke');return True
        except BudgetUnavailable:return False
    with ThreadPoolExecutor(max_workers=8) as pool:assert sum(pool.map(compete,range(8)))==1
    assert ledger.status()['states']=={'reserved':1}
    restart=Ledger(tmp_path/'ledger.sqlite3')
    with pytest.raises(BudgetUnavailable):restart.reserve(cfg,'restart','b'*64,100,512,'developer_smoke')
    changed=replace(cfg,max_requests=2)
    with pytest.raises(BudgetUnavailable):ledger.reserve(changed,'changed','c'*64,100,512,'developer_smoke')


def test_payload_boundaries_and_source_obligations_prevent_any_http(tmp_path):
    calls=[];a=adapter(tmp_path,lambda request:calls.append(request))
    async def run():
        for message,evidence,history in [('x'*2001,EVIDENCE,[]),('合成',EVIDENCE,[{'role':'system','content':'secret'}]),('合成',[{'source_id':'s','text':'合法句。','required_sentences':['捏造指令。']}],[])]:
            with pytest.raises(BudgetError):await a.generate(message,evidence,history,request_id='unit',scope='developer_smoke')
    asyncio.run(run());assert not calls


def test_embedding_verifies_only_own_model_and_rejects14b_unload(monkeypatch):
    from app.providers import EmbeddingAdapter,EMBEDDING_MODEL
    from app.ollama import MODELS,DependencyError
    async def handler(req):
        assert req.url.path=='/api/tags';return httpx.Response(200,json={'models':[{'name':EMBEDDING_MODEL,'digest':MODELS[EMBEDDING_MODEL]}]})
    async def run():
        a=EmbeddingAdapter()
        async with httpx.AsyncClient(base_url='http://127.0.0.1:11434',transport=httpx.MockTransport(handler)) as client:assert await a.verify(client)=={EMBEDDING_MODEL:MODELS[EMBEDDING_MODEL]}
        with pytest.raises(DependencyError):await a.generate('unused',[],[])
        with pytest.raises(DependencyError):await a.unload_owned('qwen3:14b')
    asyncio.run(run())


def test_one_explicit_env_source_no_environment_fallback(tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','mock-global-not-project')
    path=tmp_path/'missing.env'
    with pytest.raises(SetupUnavailable):load_config(path,tmp_path/'missing-policy')
    path.write_text('OPENAI_API_KEY=mock-only\nCLOUD_GENERATION_APPROVED=false\n')
    with pytest.raises(SetupUnavailable):load_config(path,tmp_path/'missing-policy')
    assert 'mock-only' not in repr(config())


def test_policy_selection_preserves_complete_block_and_critical_unselected_is_rejected(tmp_path):
    evidence=[{'source_id':'p','text':'合成期限12個月。資格由人工審查。','required_sentences':['合成期限12個月。','資格由人工審查。'],'policy_block':True}]
    a=adapter(tmp_path,lambda request:httpx.Response(200,json=reply(json.dumps({'answerable':True,'source_ids':['e0']}))))
    async def run():
        answer,_=await a.generate('一般政策',evidence,[],request_id='policy',scope='developer_smoke')
        assert answer.answer==evidence[0]['text']
    asyncio.run(run())
    other=adapter(tmp_path/'critical',lambda request:httpx.Response(200,json=reply()))
    async def reject():
        critical=EVIDENCE+[{'source_id':'safety','text':'請停止使用。','required_sentences':['請停止使用。'],'critical':True}]
        with pytest.raises(QualityRejection):await other.generate('合成',critical,[],request_id='critical',scope='developer_smoke')
    asyncio.run(reject())


@pytest.mark.parametrize('field',['provider','model','pricing_revision','budget_policy'])
def test_old_approval_evidence_cannot_rebind_provider_model_or_policy(tmp_path,monkeypatch,field):
    from app.cloud_config import POLICY
    from app.knowledge import build,connect,configuration,digest
    from app.lifecycle import canonical,mark_ready
    from tests.test_lifecycle import fixture_records
    path=tmp_path/'release';release=build(path);db=connect(path);cp,ep,rp=fixture_records(tmp_path,release,db)
    before=digest(canonical(configuration()))
    monkeypatch.setitem(POLICY,field,'synthetic-changed')
    assert digest(canonical(configuration()))!=before
    with pytest.raises(ValueError,match='configuration'):mark_ready(path,release,cp,ep)


def test_cancel_before_worker_starts_and_conflicting_payload_never_charge(tmp_path):
    async def run():
        calls=[];a=adapter(tmp_path,lambda req:(calls.append(req),httpx.Response(200,json=reply()))[1])
        # Occupy the slot so the reservation is definitely unsent.
        a.slot=asyncio.Semaphore(0)
        caller=asyncio.create_task(a.generate('合成',EVIDENCE,[],request_id='queued',scope='developer_smoke'));await asyncio.sleep(0);caller.cancel()
        with pytest.raises(asyncio.CancelledError):await caller
        assert a.pending==0 and a.ledger.status()['states']=={'released':1} and not calls
    asyncio.run(run())


def test_ledger_replay_and_changed_pricing_cannot_reset_spent_cap(tmp_path):
    cfg=config(max_requests=1);l=Ledger(tmp_path/'ledger');l.reserve(cfg,'one','a'*64,100,512,'developer_smoke');l.sent('one');l.settle(cfg,'one',(10,5,0,0),None,None,0)
    with pytest.raises(BudgetUnavailable):l.reserve(replace(cfg,key='mock-rotated-project-key'),'two','b'*64,100,512,'developer_smoke')
    with pytest.raises(BudgetUnavailable):l.reserve(replace(cfg,input_rate_units=cfg.input_rate_units+1),'other','b'*64,100,512,'developer_smoke')
    with pytest.raises(BudgetUnavailable):l.reserve(cfg,'one','c'*64,100,512,'developer_smoke')


@pytest.mark.parametrize('bad',['NaN','Infinity','-1','not-a-number'])
def test_nonfinite_negative_private_caps_fail_closed_without_transport(bad):
    from app.cloud_config import money_units
    with pytest.raises(SetupUnavailable):money_units(bad,cap=True)


def test_confirmed_provider_failures_open_breaker_but_do_not_retry(tmp_path):
    calls=[]
    def handler(request):calls.append(request);return httpx.Response(200,json=reply(status='failed'))
    a=adapter(tmp_path,handler)
    async def run():
        from app.ollama import CircuitOpen
        for index in range(3):
            with pytest.raises(SetupUnavailable):await a.generate('合成',EVIDENCE,[],request_id='fail'+str(index),scope='developer_smoke')
        with pytest.raises(CircuitOpen):await a.generate('合成',EVIDENCE,[],request_id='stopped',scope='developer_smoke')
    asyncio.run(run());assert len(calls)==3 and a.failures==3 and a.ledger.status()['states']=={'settled':3}


def test_scope_expiry_while_queued_releases_unsent_budget_without_http(tmp_path):
    async def run():
        current=[config()];calls=[]
        a=GenerationAdapter(config_loader=lambda :current[0],ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(lambda req:calls.append(req)))
        a.slot=asyncio.Semaphore(0)
        caller=asyncio.create_task(a.generate('合成',EVIDENCE,[],request_id='queued',scope='developer_smoke'));await asyncio.sleep(.01)
        current[0]=replace(current[0],expires_at=datetime.now(timezone.utc)-timedelta(seconds=1));a.slot.release()
        with pytest.raises(SetupUnavailable):await caller
        assert calls==[] and a.pending==0 and a.ledger.status()['states']=={'released':1}
    asyncio.run(run())


def test_queued_timeout_without_start_releases_reservation(tmp_path):
    a=adapter(tmp_path,lambda req:(_ for _ in ()).throw(AssertionError('No network')),deadline=.01)
    async def run():
        a.slot=asyncio.Semaphore(0)
        with pytest.raises(TurnTimeout):await a.generate('合成',EVIDENCE,[],request_id='queued',scope='developer_smoke')
        assert a.pending==0 and a.ledger.status()['states']=={'released':1}
    asyncio.run(run())


def test_public_scanner_never_reads_secrets_ledger_private_evidence_or_symlink(tmp_path,monkeypatch):
    from scripts.scan_public_sources import stage_sources
    root=tmp_path/'source';root.mkdir();(root/'app').mkdir();(root/'runtime').mkdir();(root/'eval').mkdir()
    for name in ['.env','project.env','app/.env.local','runtime/cloud-budget.sqlite3','eval/private-holdout.json']:(root/name).write_text('mock-sensitive-do-not-read')
    (root/'app/public.py').write_text('public source')
    (root/'app/link.py').symlink_to(root/'.env')
    destination=tmp_path/'stage';destination.mkdir();manifest=stage_sources(root,destination)
    assert set(manifest['files'])=={'app/public.py'}
    assert not list(destination.rglob('.env*')) and not (destination/'app/link.py').exists()


def test_legacy_default_generation_is_retired_before_any_transport():
    from app.ollama import Adapter,DependencyError
    async def run():
        with pytest.raises(DependencyError,match='retired'):await Adapter().generate('unused',[],[])
    asyncio.run(run())


def test_late_cancel_cannot_turn_confirmed_settlement_into_unknown(tmp_path):
    cfg=config();ledger=Ledger(tmp_path/'ledger');ledger.reserve(cfg,'done','a'*64,100,512,'developer_smoke');ledger.sent('done');ledger.settle(cfg,'done',(10,5,0,0),None,None,0)
    assert ledger.unknown('done') is False
    assert ledger.status()['states']=={'settled':1} and not ledger.status()['unknown_or_sent']


def test_new_budget_identity_cannot_reset_project_spent_cap(tmp_path):
    cfg=config(max_requests=1);ledger=Ledger(tmp_path/'ledger');ledger.reserve(cfg,'one','a'*64,100,512,'developer_smoke');ledger.sent('one');ledger.settle(cfg,'one',(10,5,0,0),None,None,0)
    with pytest.raises(BudgetUnavailable):ledger.reserve(replace(cfg,budget_id='new-id'),'two','b'*64,100,512,'developer_smoke')


def test_secret_cannot_be_misassigned_as_persisted_budget_identity(tmp_path):
    cfg=config();cfg=replace(cfg,budget_id=cfg.key);ledger=Ledger(tmp_path/'ledger')
    with pytest.raises(SetupUnavailable):ledger.reserve(cfg,'safe-id','a'*64,100,512,'developer_smoke')
    db=sqlite3.connect(tmp_path/'ledger');assert cfg.key not in '\n'.join(db.iterdump());db.close()
