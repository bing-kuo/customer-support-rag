"""Pure synthetic previews; no real request IDs, ledger handles or provider evidence."""
import copy,json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone
import pytest
from app.cloud_config import fingerprint,SetupUnavailable,MODEL
from app.reconciliation import preview
from tests.test_cloud_offline import config
from tests.test_usage_diagnostics import raw_usage


def fixture():
    c=config();created=datetime.now(timezone.utc)-timedelta(minutes=1)
    s=dict(project=c.budget_id,local_request_id='synthetic-local',provider_request_id='req_synthetic',provider_response_id='resp_synthetic',body_sha256='a'*64,configuration_sha256='b'*64,approval_binding=c.binding(),pricing_hash=c.pricing_hash(),state='unknown',created_at=created.isoformat(),reserved_input=200,reserved_output=512,reserved_cost=c.price(200,512),ceiling_fingerprint=fingerprint({'ceiling':c.cap_units}),spent_and_reserved=c.price(200,512))
    e={k:s[k] for k in ('project','local_request_id','provider_request_id','provider_response_id','body_sha256','configuration_sha256','approval_binding','pricing_hash')}
    e.update(model=MODEL,service_tier='default',status='completed',terminal_at=datetime.now(timezone.utc).isoformat(),usage=raw_usage(),source={'method':'complete_response','artifact_sha256':'c'*64})
    r=dict(reviewer='Marcus',evidence_sha256=fingerprint(e),source_artifact_sha256='c'*64,configuration_sha256=s['configuration_sha256'])
    return s,c,e,r


def test_preview_repeated_concurrent_calls_are_readonly_and_not_authorization(tmp_path):
    s,c,e,r=fixture();before=copy.deepcopy((s,e,r));files=set(tmp_path.iterdir())
    with ThreadPoolExecutor(2) as pool:results=list(pool.map(lambda _:preview(s,c,e,r),range(2)))
    assert results[0]==results[1] and results[0]['would_mutate'] is False
    assert results[0]['real_reconciliation_authorized'] is False and results[0]['provider_authenticity_verified'] is False
    assert (s,e,r)==before and set(tmp_path.iterdir())==files and s['state']=='unknown'


@pytest.mark.parametrize('field',['project','local_request_id','provider_request_id','provider_response_id','body_sha256','configuration_sha256','approval_binding','pricing_hash','model','service_tier','status','terminal_at'])
def test_wrong_identity_or_terminal_binding_rejected_even_with_new_review_hash(field):
    s,c,e,r=fixture();e[field]='synthetic-wrong';r['evidence_sha256']=fingerprint(e)
    with pytest.raises(SetupUnavailable):preview(s,c,e,r)
    assert s['state']=='unknown'


@pytest.mark.parametrize('case',['tampered_evidence','missing_breakdown','over_reservation','new_fee','wrong_source','wrong_reviewer','stale_review','ceiling_changed','at_cap','not_unknown'])
def test_untrusted_or_unbounded_evidence_cannot_unlock(case):
    s,c,e,r=fixture()
    if case=='tampered_evidence':e['usage']['input_tokens']=99
    if case=='missing_breakdown':e['usage'].pop('input_tokens_details')
    if case=='over_reservation':s['reserved_input']=1
    if case=='new_fee':e['usage']['private-fee-name']=1
    if case=='wrong_source':e['source']['method']='hand_entered_or_screenshot'
    if case=='wrong_reviewer':r['reviewer']='self'
    if case=='stale_review':r['configuration_sha256']='d'*64
    if case=='ceiling_changed':s['ceiling_fingerprint']='d'*64
    if case=='at_cap':s['spent_and_reserved']=c.cap_units
    if case=='not_unknown':s['state']='released'
    if case!='tampered_evidence':r['evidence_sha256']=fingerprint(e)
    with pytest.raises(SetupUnavailable):preview(s,c,e,r)
