"""Fictional private policies and accounting; no actual key/policy/network access."""
import asyncio,json,sqlite3
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone
import httpx,pytest
from app.cloud_config import load_config,MODEL,PRICING_REVISION,RESERVATION_INPUT,RESERVATION_OUTPUT,money_units,SetupUnavailable
from app.budget import Ledger,BudgetUnavailable
from app.cloud import GenerationAdapter
from tests.test_cloud_offline import config,reply,EVIDENCE
from scripts.cloud_setup import check,diagnose


def fixture_policy(tmp_path,**changes):
    p=tmp_path/'policy.json'
    value=dict(version=1,model=MODEL,budget_id='unit-policy',cap_units=10**10,input_rate_units=money_units(RESERVATION_INPUT),output_rate_units=money_units(RESERVATION_OUTPUT),extra_units=0,pricing_revision=PRICING_REVISION,pricing_expires_at=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat(),enabled_scopes=['developer_smoke','public_eval'],enabled=True)
    value.update(changes);p.write_text(json.dumps(value));p.chmod(0o600);return p


def test_key_only_config_ignores_legacy_fields_not_global_env(tmp_path,monkeypatch):
    p=tmp_path/'env';p.write_text('OPENAI_API_KEY=mock-only-key\nCLOUD_SCOPE=\nCLOUD_CAP_USD=\nCLOUD_GENERATION_APPROVED=false\n')
    policy=fixture_policy(tmp_path);before=p.read_bytes();cfg=load_config(p,policy)
    assert all(check(lambda:cfg).values()) and diagnose(p,policy)=={'ready':True,'issues':[]}
    assert p.read_bytes()==before
    monkeypatch.setenv('OPENAI_API_KEY','must-not-use-global');p.write_text('')
    with pytest.raises(SetupUnavailable):load_config(p,policy)
    assert diagnose(p,policy)['issues']==[{'category':'missing_or_blank','field':'OPENAI_API_KEY'}]


@pytest.mark.parametrize('change',[{'enabled':False},{'model':'other-model'},{'pricing_expires_at':'2020-01-01T00:00:00Z'},{'input_rate_units':1},{'enabled_scopes':['unknown']}])
def test_private_policy_fails_closed_without_private_values(tmp_path,change):
    p=tmp_path/'env';p.write_text('OPENAI_API_KEY=mock-only-key')
    policy=fixture_policy(tmp_path,**change)
    with pytest.raises(SetupUnavailable):load_config(p,policy)
    result=diagnose(p,policy);assert not result['ready']
    assert 'mock-only-key' not in json.dumps(result) and 'cap_units' not in json.dumps(result)


def test_cumulative_strict_boundary_and_restart_key_rotation(tmp_path):
    c=config();cost=c.price(100,512);c=replace(c,cap_units=cost*2)
    path=tmp_path/'ledger';l=Ledger(path);l.reserve(c,'first','a'*64,100,512,'developer_smoke')
    l.sent('first');l.settle(c,'first',(100,512,0,0),None,None,0)
    reopened=Ledger(path)
    with pytest.raises(BudgetUnavailable):reopened.reserve(replace(c,key='mock-rotated-key'),'second','b'*64,100,512,'developer_smoke')
    assert reopened.status()['states']=={'settled':1}
    with pytest.raises(BudgetUnavailable):reopened.reserve(replace(c,cap_units=c.cap_units+1),'third','c'*64,100,512,'developer_smoke')


def test_concurrent_cap_reservation_no_equal_boundary(tmp_path):
    c=config();c=replace(c,cap_units=c.price(100,512)*2);l=Ledger(tmp_path/'ledger')
    def reserve(i):
        try:l.reserve(c,'req'+str(i),str(i)*64,100,512,'developer_smoke');return True
        except BudgetUnavailable:return False
    with ThreadPoolExecutor(2) as pool:assert sum(pool.map(reserve,[1,2]))==1
    assert l.status()['states']=={'reserved':1}


def test_unknown_restart_and_migration_cannot_reclaim(tmp_path):
    c=config();path=tmp_path/'ledger';l=Ledger(path)
    l.reserve(c,'pending','a'*64,100,512,'developer_smoke');l.sent('pending');l.unknown('pending')
    again=Ledger(path)
    with pytest.raises(BudgetUnavailable):again.migrate_policy(c)
    with pytest.raises(BudgetUnavailable):again.reserve(replace(c,key='mock-rotated-key'),'new','b'*64,100,512,'developer_smoke')
    assert again.status()['states']=={'unknown':1}


def test_legacy_migration_preserves_usage_and_replay(tmp_path):
    c=config();path=tmp_path/'ledger';l=Ledger(path)
    l.reserve(c,'paid-before','a'*64,100,512,'developer_smoke');l.sent('paid-before');l.settle(c,'paid-before',(100,20,0,0),None,None,0)
    with l.db() as db:db.execute('DROP TABLE policy_guard')
    again=Ledger(path);new=replace(c,budget_id='unit-policy',cap_units=c.cap_units-1,input_rate_units=c.input_rate_units*2)
    again.migrate_policy(new)
    assert again.status()['states']=={'settled':1} and again.status()['input_tokens']==100
    with pytest.raises(BudgetUnavailable):again.reserve(new,'paid-before','a'*64,100,512,'developer_smoke')
    with again.db() as db:
        assert db.execute('SELECT used_cost FROM requests').fetchone()[0]>=c.price(100,20)


def test_incompatible_ledger_stops_without_destroying_records(tmp_path):
    path=tmp_path/'ledger';d=sqlite3.connect(path);d.execute('CREATE TABLE requests(secret TEXT)');d.execute("INSERT INTO requests VALUES('synthetic-existing')");d.commit();d.close()
    with pytest.raises(BudgetUnavailable):Ledger(path)
    d=sqlite3.connect(path);assert d.execute('SELECT secret FROM requests').fetchone()[0]=='synthetic-existing';d.close()


def test_nonstandard_returned_tier_retains_unknown_mock_only(tmp_path):
    calls=[]
    def handler(r):
        calls.append(r);assert json.loads(r.content)['service_tier']=='default'
        return httpx.Response(200,json=reply(service_tier='fast'))
    c=config();a=GenerationAdapter(config_loader=lambda:c,ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(handler))
    async def run():
        with pytest.raises(SetupUnavailable):await a.generate('合成清潔問題',EVIDENCE,[],request_id='tier',scope='developer_smoke')
    asyncio.run(run());assert len(calls)==1 and a.ledger.status()['states']=={'unknown':1}


def test_key_only_smoke_real_loader_mock_transport_restart_no_duplicate(tmp_path):
    from scripts.cloud_smoke import run
    env=tmp_path/'env';env.write_text('OPENAI_API_KEY=mock-project-key')
    policy=fixture_policy(tmp_path);loader=lambda:load_config(env,policy);calls=[]
    def handler(request):calls.append(request);return httpx.Response(200,json=reply())
    path=tmp_path/'ledger';a=GenerationAdapter(ledger_path=path,transport=httpx.MockTransport(handler))
    value=asyncio.run(run(dry_run=False,authorised=True,loader=loader,adapter=a))
    assert value['completed'] and len(calls)==1
    env.write_text('OPENAI_API_KEY=mock-rotated-project-key')
    reopened=GenerationAdapter(ledger_path=path,transport=httpx.MockTransport(handler))
    with pytest.raises(SetupUnavailable):asyncio.run(run(dry_run=False,authorised=True,loader=loader,adapter=reopened))
    assert len(calls)==1 and reopened.ledger.status()['states']=={'settled':1}


def test_key_only_policy_revoked_in_file_before_send_zero_mock_calls(tmp_path):
    from scripts.cloud_smoke import run
    env=tmp_path/'env';env.write_text('OPENAI_API_KEY=mock-project-key');policy=fixture_policy(tmp_path);loads=[];calls=[]
    def loader():
        loads.append(1)
        if len(loads)==3:
            data=json.loads(policy.read_bytes());data['enabled']=False;policy.write_text(json.dumps(data))
        return load_config(env,policy)
    a=GenerationAdapter(ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(lambda r:calls.append(r)))
    with pytest.raises(SetupUnavailable):asyncio.run(run(dry_run=False,authorised=True,loader=loader,adapter=a))
    assert calls==[] and a.ledger.status()['states']=={'released':1} and a.pending==0


@pytest.mark.parametrize('corruption',['null_state','negative_cost','null_model'])
def test_legacy_incompatible_rows_stop_transactional_migration(tmp_path,corruption):
    c=config();path=tmp_path/'ledger';l=Ledger(path)
    l.reserve(c,'historical','a'*64,100,512,'developer_smoke');l.sent('historical');l.settle(c,'historical',(100,20,0,0),None,None,0)
    with l.db() as db:
        if corruption=='null_state':db.execute('UPDATE requests SET state=NULL')
        if corruption=='negative_cost':db.execute('UPDATE requests SET used_cost=-1')
        if corruption=='null_model':db.execute('UPDATE requests SET requested_model=NULL')
        before=db.execute('SELECT * FROM requests').fetchall()
    with pytest.raises(BudgetUnavailable):l.migrate_policy(c)
    with l.db() as db:assert db.execute('SELECT * FROM requests').fetchall()==before
