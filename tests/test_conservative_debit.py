"""Synthetic reviewed operator fixtures only; never open the actual project ledger."""
import copy,sqlite3
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone
import pytest
from app.budget import Ledger,BudgetUnavailable
from app.cloud_config import MODEL,fingerprint
from tests.test_cloud_offline import config


def fixture(tmp_path,cfg=None):
    cfg=cfg or config();l=Ledger(tmp_path/'ledger');l.reserve(cfg,'original','a'*64,100,512,'developer_smoke');l.sent('original')
    l.provider_identity('original','req_fixture','resp_fixture',MODEL);l.unknown('original')
    with l.db() as db:r=db.execute('SELECT res_cost,res_input,res_output FROM requests').fetchone()
    d=dict(local_request_id='original',project=cfg.budget_id,body_sha256='a'*64,pricing_hash=cfg.pricing_hash(),approval_binding=cfg.binding(),configuration_sha256='b'*64,ceiling_fingerprint=fingerprint({'ceiling':cfg.cap_units}),reservation_sha256=fingerprint(dict(zip(('res_cost','res_input','res_output'),r))),completion_evidence=dict(artifact_sha256='c'*64,kind='authenticated_same_request_terminal',provider_request_id='req_fixture',provider_response_id='resp_fixture',status='completed',terminal_at=(datetime.now(timezone.utc)+timedelta(seconds=1)).isoformat()),bound_evidence=dict(artifact_sha256='d'*64,kind='independently_reviewed_applicable_fee_bound',request_sha256='a'*64,provider_request_id='req_fixture',model=MODEL,service_tier='default',pricing_hash=cfg.pricing_hash(),all_components_bounded=True))
    review=dict(reviewer='Dana',decision_sha256=fingerprint(d),configuration_sha256='b'*64,completion_artifact_sha256='c'*64,bound_artifact_sha256='d'*64)
    return l,cfg,d,review


def dump(l):
    with l.db() as db:return '\n'.join(db.iterdump())


def test_dryrun_nonmutation_and_full_consumption_not_actual_usage(tmp_path):
    l,c,d,r=fixture(tmp_path);before=dump(l)
    result=l.conservative_debit(c,'original',d,r)
    assert result['dry_run'] and not result['reservation_permanently_counted'] and dump(l)==before
    result=l.conservative_debit(c,'original',d,r,dry_run=False)
    assert result['actual_usage']=='unknown' and result['provider_settled'] is False
    assert l.status()['states']=={'unknown':1} and not l.status()['unknown_or_sent']
    with l.db() as db:
        row=db.execute('SELECT state,res_cost,used_input,used_output,used_cost FROM requests').fetchone()
        assert row[0]=='unknown' and row[2:]==(None,None,None)
        assert db.execute('SELECT res_cost FROM conservative_debits').fetchone()[0]==row[1]
        assert db.execute("SELECT count(*) FROM events WHERE event='unknown_cost_retained'").fetchone()[0]==1


def test_idempotent_restart_concurrency_and_distinct_future_admission(tmp_path):
    l,c,d,r=fixture(tmp_path)
    with ThreadPoolExecutor(2) as pool:results=list(pool.map(lambda _:l.conservative_debit(c,'original',d,r,dry_run=False),range(2)))
    assert sum(not a['already_recorded'] for a in results)==1
    again=Ledger(tmp_path/'ledger');assert again.conservative_debit(c,'original',d,r,dry_run=False)['already_recorded']
    with again.db() as db:assert db.execute("SELECT count(*) FROM events WHERE event='full_reservation_permanently_consumed_unknown_usage'").fetchone()[0]==1
    again.reserve(replace(c,key='fixture-rotated-key'),'distinct-purpose','e'*64,100,512,'developer_smoke');again.sent('distinct-purpose')
    assert again.status()['states']=={'unknown':1,'sent':1}


@pytest.mark.parametrize('case',['wrong_id','wrong_body','wrong_pricing','wrong_config','wrong_ceiling','wrong_reservation','missing_terminal','pending_terminal','wrong_response','unbounded_fee','unknown_tier','missing_bound','wrong_review','unreviewed_change'])
def test_missing_or_wrong_prerequisites_never_change_unknown(case,tmp_path):
    l,c,d,r=fixture(tmp_path);before=dump(l);request='original'
    if case=='wrong_id':request='other'
    if case=='wrong_body':d['body_sha256']='e'*64
    if case=='wrong_pricing':d['pricing_hash']='e'*64
    if case=='wrong_config':d['configuration_sha256']='e'*64
    if case=='wrong_ceiling':d['ceiling_fingerprint']='e'*64
    if case=='wrong_reservation':d['reservation_sha256']='e'*64
    if case=='missing_terminal':d['completion_evidence']={}
    if case=='pending_terminal':d['completion_evidence']['status']='in_progress'
    if case=='wrong_response':d['completion_evidence']['provider_response_id']='resp_other'
    if case=='unbounded_fee':d['bound_evidence']['all_components_bounded']=False
    if case=='unknown_tier':d['bound_evidence']['service_tier']='unknown'
    if case=='missing_bound':d['bound_evidence']={}
    if case=='wrong_review':r['reviewer']='self'
    if case=='unreviewed_change':d['completion_evidence']['artifact_sha256']='e'*64
    # Most adversarial cases supply a self-consistent decision hash, which is insufficient.
    if case!='unreviewed_change':r['decision_sha256']=fingerprint(d)
    with pytest.raises(BudgetUnavailable):l.conservative_debit(c,request,d,r,dry_run=False)
    assert dump(l)==before and l.status()['unknown_or_sent']


def test_original_id_and_body_permanently_refused_cap_not_refunded(tmp_path):
    l,c,d,r=fixture(tmp_path);l.conservative_debit(c,'original',d,r,dry_run=False)
    for request,body in [('original','a'*64),('original','e'*64),('different-id','a'*64)]:
        with pytest.raises(BudgetUnavailable):l.reserve(c,request,body,100,512,'developer_smoke')
    with pytest.raises(BudgetUnavailable):l.migrate_policy(c)
    with pytest.raises(BudgetUnavailable):l.reserve(replace(c,cap_units=c.cap_units+1),'new','e'*64,100,512,'developer_smoke')
    with l.db() as db:
        # No extra accounting row is added for a debit; the original full reservation stays counted once.
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0]==1
        assert db.execute("SELECT sum(coalesce(used_cost,res_cost)) FROM requests WHERE state!='released'").fetchone()[0]==db.execute('SELECT res_cost FROM conservative_debits').fetchone()[0]


def test_conflicting_decision_and_crash_storage_failure_rollback(tmp_path):
    l,c,d,r=fixture(tmp_path);before=dump(l)
    with l.db() as db:db.execute("CREATE TRIGGER failure BEFORE INSERT ON conservative_debits BEGIN SELECT RAISE(ABORT,'synthetic-private'); END")
    with pytest.raises(BudgetUnavailable):l.conservative_debit(c,'original',d,r,dry_run=False)
    with l.db() as db:
        assert db.execute('SELECT count(*) FROM conservative_debits').fetchone()[0]==0
        db.execute('DROP TRIGGER failure')
    assert dump(l)==before
    l.conservative_debit(c,'original',d,r,dry_run=False)
    d['completion_evidence']['artifact_sha256']='f'*64;r.update(decision_sha256=fingerprint(d),completion_artifact_sha256='f'*64)
    with pytest.raises(BudgetUnavailable):l.conservative_debit(c,'original',d,r,dry_run=False)


def test_other_unknown_and_pending_are_never_blanket_unblocked(tmp_path):
    l,c,d,r=fixture(tmp_path)
    with l.db() as db:db.execute("INSERT INTO requests(id,budget,body_hash,owner,state,res_cost,res_input,res_output) VALUES('other',?,'other','other','sent',1,1,1)",(c.budget_id,))
    with pytest.raises(BudgetUnavailable):l.conservative_debit(c,'original',d,r,dry_run=False)
    with l.db() as db:db.execute("UPDATE requests SET state='unknown' WHERE id='other'")
    l.conservative_debit(c,'original',d,r,dry_run=False)
    assert l.status()['unknown_or_sent']
    with pytest.raises(BudgetUnavailable):l.reserve(c,'future','e'*64,100,512,'developer_smoke')


def test_full_reservation_still_counts_at_strict_future_cap_boundary(tmp_path):
    c=config();c=replace(c,cap_units=c.price(100,512)*2)
    l,c,d,r=fixture(tmp_path,c)
    l.conservative_debit(c,'original',d,r,dry_run=False)
    with pytest.raises(BudgetUnavailable):l.reserve(c,'future','e'*64,100,512,'developer_smoke')
    assert l.status()['states']=={'unknown':1}


def test_late_actual_settlement_cannot_overwrite_permanent_unknown_disposition(tmp_path):
    l,c,d,r=fixture(tmp_path);l.conservative_debit(c,'original',d,r,dry_run=False)
    # Permanent debit must not later be discounted by the original owner's late worker.
    with pytest.raises(BudgetUnavailable):l.settle(c,'original',(100,0,0,0),'req_fixture','resp_fixture',0)
    assert l.status()['states']=={'unknown':1}
