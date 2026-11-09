"""Synthetic explicit-user decision fixtures; never a provider authenticity assertion."""
import asyncio,copy,json
from dataclasses import replace
import httpx,pytest
from app.budget import BudgetUnavailable
from app.cloud import GenerationAdapter
from app.cloud_config import fingerprint,TIER_BOUND_MULTIPLIER,PRICE_BOUND_REVISION,RESERVATION_INPUT,RESERVATION_OUTPUT,money_units
from scripts.cloud_smoke import run,REQUEST_ID,NEW_REQUEST_ID,NEW_MESSAGE,MESSAGE
from tests.test_conservative_debit import fixture,dump
from tests.test_cloud_offline import reply,config


def accepted(tmp_path):
    l,c,d,_=fixture(tmp_path)
    d.pop('completion_evidence');d.pop('bound_evidence')
    d['user_risk_acceptance']=dict(decision_id='user-risk-original-v1',source='explicit_user_decision',source_reference='orca-task:task_665fb43e8a1e',accept_original_cost_uncertainty=True,preserve_original_full_reservation=True,actual_usage='unknown',provider_verified=False,user_reported_total_tokens=279)
    r=dict(reviewer='Dana',decision_sha256=fingerprint(d),configuration_sha256=d['configuration_sha256'],user_decision_id='user-risk-original-v1')
    return l,c,d,r


def test_explicit_risk_acceptance_needs_no_invented_terminal_and_persists_source(tmp_path):
    l,c,d,r=accepted(tmp_path);before=dump(l)
    preview=l.conservative_debit(c,'original',d,r)
    assert preview['decision_basis']=='explicit_user_risk_acceptance' and preview['provider_verified'] is False and dump(l)==before
    l.conservative_debit(c,'original',d,r,dry_run=False)
    with l.db() as db:
        state,i,o,cost=db.execute('SELECT state,used_input,used_output,used_cost FROM requests').fetchone()
        metadata=json.loads(db.execute('SELECT decision_metadata FROM conservative_debits').fetchone()[0])
        assert (state,i,o,cost)==('unknown',None,None,None)
        assert metadata['user_risk_acceptance']['user_reported_total_tokens']==279
        assert metadata['user_risk_acceptance']['source']=='explicit_user_decision'
    assert l.conservative_debit(c,'original',d,r,dry_run=False)['already_recorded']


@pytest.mark.parametrize('field,value',[('decision_id',''),('source','provider_export'),('source_reference','unverifiable'),('accept_original_cost_uncertainty',False),('preserve_original_full_reservation',False),('actual_usage','settled'),('provider_verified',True),('user_reported_total_tokens',True)])
def test_risk_decision_cannot_claim_verified_or_drop_explicit_acceptance(tmp_path,field,value):
    l,c,d,r=accepted(tmp_path);before=dump(l)
    d['user_risk_acceptance'][field]=value;r['decision_sha256']=fingerprint(d)
    with pytest.raises(BudgetUnavailable):l.conservative_debit(c,'original',d,r,dry_run=False)
    assert dump(l)==before


def test_legacy_original_pricing_and_reserve_not_repriced_or_refunded(tmp_path):
    l,c,d,r=accepted(tmp_path)
    with l.db() as db:
        db.execute("UPDATE requests SET pricing_hash=?,res_cost=res_cost/2 WHERE id='original'",('f'*64,))
        old=db.execute('SELECT res_cost,res_input,res_output FROM requests').fetchone()
    d['pricing_hash']='f'*64;d['reservation_sha256']=fingerprint(dict(zip(('res_cost','res_input','res_output'),old)));r['decision_sha256']=fingerprint(d)
    l.conservative_debit(c,'original',d,r,dry_run=False)
    with l.db() as db:assert db.execute('SELECT res_cost FROM conservative_debits').fetchone()[0]==old[0]
    assert not l.status()['unknown_or_sent']
    with pytest.raises(BudgetUnavailable):l.reserve(c,'original','a'*64,100,512,'developer_smoke')


def test_future_fee_bound_covers_all_documented_short_luna_tiers_without_changing_cap():
    c=config(input_rate_units=money_units(RESERVATION_INPUT),output_rate_units=money_units(RESERVATION_OUTPUT));before=c.cap_units
    standard=c.input_rate_units*3072+c.output_rate_units*512
    assert c.price(3072,512)*1000000>=standard*TIER_BOUND_MULTIPLIER
    assert TIER_BOUND_MULTIPLIER==2 and PRICE_BOUND_REVISION in str({'revision':PRICE_BOUND_REVISION})
    assert c.cap_units==before


def test_new_smoke_distinct_body_id_once_with_no_original_resend(tmp_path):
    l,c,d,r=accepted(tmp_path);l.conservative_debit(c,'original',d,r,dry_run=False)
    # The production fixed original ID remains permanently denied; use a separate purpose only.
    with l.db() as db:db.execute('UPDATE requests SET id=?',(REQUEST_ID,));db.execute('UPDATE conservative_debits SET id=?',(REQUEST_ID,))
    calls=[]
    def handler(request):calls.append(request);return httpx.Response(200,json=reply())
    a=GenerationAdapter(config_loader=lambda:c,ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(handler))
    assert NEW_REQUEST_ID!=REQUEST_ID and NEW_MESSAGE!=MESSAGE
    result=asyncio.run(run(dry_run=False,authorised=True,loader=lambda:c,adapter=a,new_request=True))
    assert result['completed'] and len(calls)==1
    with pytest.raises(BudgetUnavailable):asyncio.run(run(dry_run=False,authorised=True,loader=lambda:c,adapter=a,new_request=True))
    assert len(calls)==1
