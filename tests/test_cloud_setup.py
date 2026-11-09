"""Synthetic configuration only; transport is globally blocked except explicit mocks."""
import asyncio,json
from dataclasses import replace
import httpx,pytest
from tests.test_cloud_offline import config,reply
from app.cloud import GenerationAdapter
from app.cloud_config import SetupUnavailable
from scripts.cloud_setup import check
from scripts.cloud_smoke import run,REQUEST_ID
from scripts.scan_public_sources import stage_sources


def smoke_config():return replace(config(),max_requests=1,max_input_tokens=3072,max_output_tokens=512,max_total_tokens=3584)


def test_setup_only_booleans_and_redacted_failure(tmp_path):
    before=set(tmp_path.iterdir());c=smoke_config();assert all(check(lambda:c).values())
    def bad():raise RuntimeError(c.key+' private financial and expiry')
    result=check(bad);assert not any(result.values())
    assert all(type(v) is bool for v in result.values()) and c.key not in json.dumps(result)
    assert set(tmp_path.iterdir())==before


def test_smoke_dry_run_does_not_read_config_or_create_ledger(tmp_path):
    def forbidden():raise AssertionError('Must not read configuration')
    a=GenerationAdapter(config_loader=forbidden,ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(lambda r:pytest.fail('network')))
    result=asyncio.run(run(loader=forbidden,adapter=a))
    assert result['network_attempted'] is False and not (tmp_path/'ledger').exists()


def test_smoke_requires_explicit_authorisation_and_exact_scope_caps(tmp_path):
    calls=[];a=GenerationAdapter(ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(lambda r:calls.append(r)))
    for kwargs in [{'authorised':False},{'authorised':True,'loader':lambda:replace(smoke_config(),scopes=('serve',))},{'authorised':True,'loader':lambda:replace(smoke_config(),max_requests=0)}]:
        with pytest.raises(SetupUnavailable):asyncio.run(run(dry_run=False,adapter=a,**kwargs))
    assert calls==[] and not (tmp_path/'ledger').exists()


def test_smoke_one_mock_request_no_repeat_charge(tmp_path):
    calls=[]
    def handler(r):calls.append(r);return httpx.Response(200,json=reply())
    a=GenerationAdapter(ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(handler));c=smoke_config()
    result=asyncio.run(run(dry_run=False,authorised=True,loader=lambda:c,adapter=a))
    assert result==dict(completed=True,input_tokens=100,output_tokens=20,total_tokens=120)
    with pytest.raises(SetupUnavailable):asyncio.run(run(dry_run=False,authorised=True,loader=lambda:c,adapter=a))
    assert len(calls)==1 and a.pending==0


def test_public_scan_includes_documents_not_approvals_or_private(tmp_path):
    root=tmp_path/'source';stage=tmp_path/'stage';root.mkdir()
    names=['knowledge/documents/synthetic.md','docs/setup.md','licenses/dependency.txt','notices/third-party.md','knowledge/approvals/private.json','knowledge/manifest.json','eval/private/raw.json','.env','runtime/ledger.sqlite3']
    for n in names:
        p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic')
    files=stage_sources(root,stage)['files']
    assert set(files)==set(names[:4])


def test_revision_defaults_and_lower_override_rejected():
    from app.cloud_config import config_from_mapping,STANDARD_INPUT,STANDARD_OUTPUT,money_units,PRICING_REVISION
    c=smoke_config()
    values={'OPENAI_API_KEY':c.key,'CLOUD_GENERATION_APPROVED':'true','CLOUD_SYNTHETIC_ONLY':'true','CLOUD_PRICING_APPROVED':'true','CLOUD_BUDGET_ID':'fixture','CLOUD_SCOPE':'developer_smoke','CLOUD_EXPIRES_AT':c.expires_at.isoformat(),'CLOUD_PRICING_VALID_UNTIL':c.pricing_expires_at.isoformat(),'CLOUD_MAX_REQUESTS':'1','CLOUD_MAX_INPUT_TOKENS':'3072','CLOUD_MAX_OUTPUT_TOKENS':'512','CLOUD_MAX_TOTAL_TOKENS':'3584','CLOUD_CAP_USD':'1','CLOUD_MAX_EXTRA_USD_PER_REQUEST':'0','CLOUD_PRICING_REVISION':PRICING_REVISION,'CLOUD_BILLING_COMPONENTS':'input,output'}
    cfg=config_from_mapping(values)
    assert cfg.input_rate_units==money_units(STANDARD_INPUT) and cfg.output_rate_units==money_units(STANDARD_OUTPUT)
    for key in ['CLOUD_INPUT_USD_PER_MILLION','CLOUD_OUTPUT_USD_PER_MILLION']:
        with pytest.raises(SetupUnavailable):config_from_mapping(dict(values,**{key:'0'}))
    assert config_from_mapping(dict(values,CLOUD_INPUT_USD_PER_MILLION='',CLOUD_OUTPUT_USD_PER_MILLION='')).pricing_revision==PRICING_REVISION


def test_smoke_provider_error_stops_without_retry(tmp_path):
    calls=[]
    def handler(r):calls.append(r);return httpx.Response(503,text='mock secret body')
    a=GenerationAdapter(ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(handler));c=smoke_config()
    with pytest.raises(Exception):asyncio.run(run(dry_run=False,authorised=True,loader=lambda:c,adapter=a))
    assert len(calls)==1 and a.ledger.status()['unknown_or_sent']


def test_setup_cli_redacts_config_exception(monkeypatch,capsys):
    import scripts.cloud_setup as cli
    def broken():raise RuntimeError('mock-sensitive-value')
    monkeypatch.setattr(cli,'check',lambda:check(broken));monkeypatch.setattr('sys.argv',['cloud_setup'])
    assert cli.main()==1
    output=capsys.readouterr();assert output.err=='' and 'mock-sensitive-value' not in output.out
    assert set(json.loads(output.out).values())=={False}


def test_smoke_cli_redacts_exception_without_retry(monkeypatch,capsys):
    import scripts.cloud_smoke as cli
    count=[]
    async def broken(**kwargs):count.append(kwargs);raise RuntimeError('mock-sensitive-value')
    monkeypatch.setattr(cli,'run',broken);monkeypatch.setattr('sys.argv',['cloud_smoke','--execute-authorized-one-request'])
    assert cli.main()==1 and len(count)==1
    output=capsys.readouterr();assert output.err=='' and 'mock-sensitive-value' not in output.out
    assert json.loads(output.out)==dict(completed=False,status='stopped_no_automatic_retry')


@pytest.mark.parametrize('phase',[2,3])
@pytest.mark.parametrize('mutation',['approval','scope','cap','expiry','key'])
def test_smoke_dynamic_revocation_before_admission_or_send(tmp_path,phase,mutation):
    from datetime import datetime,timedelta,timezone
    import sqlite3
    c=smoke_config();loads=[];calls=[]
    def loader():
        loads.append(1)
        if len(loads)<phase:return c
        if mutation=='approval':raise SetupUnavailable()
        if mutation=='scope':return replace(c,scopes=('serve',))
        if mutation=='cap':return replace(c,cap_units=1)
        if mutation=='expiry':return replace(c,expires_at=datetime.now(timezone.utc)-timedelta(seconds=1))
        return replace(c,key='mock-only-revoked-replacement-key')
    path=tmp_path/'ledger';a=GenerationAdapter(ledger_path=path,transport=httpx.MockTransport(lambda r:calls.append(r)))
    with pytest.raises(SetupUnavailable):asyncio.run(run(dry_run=False,authorised=True,loader=loader,adapter=a))
    assert calls==[] and len(loads)==phase and a.pending==0 and not a.uncertain
    if phase==2:assert not path.exists()
    else:
        assert a.ledger.status()['states']=={'released':1}
        db=sqlite3.connect(path)
        assert db.execute('SELECT event FROM events ORDER BY at').fetchall()==[('reserved',),('released_before_send',)]
        assert db.execute('SELECT body_hash FROM requests').fetchone()[0]
        db.close()


def test_smoke_revocation_does_not_clear_other_unknown_reservation(tmp_path):
    from app.budget import Ledger
    c=smoke_config();ledger=Ledger(tmp_path/'ledger')
    ledger.reserve(c,'prior-synthetic','a'*64,100,512,'developer_smoke');ledger.sent('prior-synthetic');ledger.unknown('prior-synthetic')
    calls=[];a=GenerationAdapter(ledger_path=tmp_path/'ledger',transport=httpx.MockTransport(lambda r:calls.append(r)))
    with pytest.raises(SetupUnavailable):asyncio.run(run(dry_run=False,authorised=True,loader=lambda:c,adapter=a))
    assert calls==[] and ledger.status()['states']=={'unknown':1} and a.pending==0
