from fastapi.testclient import TestClient
from app.main import app, sessions
from app import service
from app.knowledge import connect,build,seal
from app.lifecycle import mark_ready,activate
from tests.test_lifecycle import fixture_records


def test_pinned_clarification_replay_conflict_and_refund(tmp_path,monkeypatch):
    path=tmp_path/'runtime';rid=seal(path,build(path));db=connect(path)
    cp,ep,rp=fixture_records(tmp_path,rid,db)
    mark_ready(path,rid,cp,ep);activate(path,rid,None,rp)
    monkeypatch.setattr(service,'DB',path)
    client=TestClient(app)
    sid=client.post('/api/v1/sessions',json={}).json()['session_id']
    endpoint=f'/api/v1/sessions/{sid}/turns'
    body={'client_turn_id':'one','message':'耳機配對','product_id':None}
    first=client.post(endpoint,json=body)
    assert first.status_code==200 and first.json()['status']=='clarify'
    assert first.json()['release_id']==rid
    assert client.post(endpoint,json=body).json()==first.json()
    assert client.post(endpoint,json=dict(body,message='changed')).status_code==409
    sessions[sid]['inflight']=True
    assert client.post(endpoint,json=dict(body,client_turn_id='two')).status_code==409
    sessions[sid]['inflight']=False
    response=client.post(endpoint,json=dict(body,client_turn_id='refund',message='我要退款',product_id='hp-a'))
    assert response.json()['handoff_summary']['reason']=='refund_request'
    assert client.get('/api/v1/not-real').json()['request_id']
    assert client.get('/api/v1/health',headers={'host':'evil.example'}).status_code==400


def test_positive_product_and_date_gold(tmp_path):
    from app.knowledge import search_candidate
    path=tmp_path/'db';rid=build(path)
    rows=search_candidate(path,rid,'充電接點','hp-a','2026-11-08')
    assert rows and any(row[1]=='hp-a-charging' for row in rows)
    assert not any(row[1].startswith('hp-b') for row in rows)
    assert search_candidate(path,rid,'充電接點','hp-a','2026-11-08','en-US')==[]
    for when,start in [('2026-06-30','2026-01-01'),('2026-07-01','2026-07-01')]:
        rows=search_candidate(path,rid,'保固',None,when)
        assert rows
        db=connect(path)
        assert all(db.execute('SELECT start FROM chunks WHERE release=? AND id=?',(rid,row[0])).fetchone()[0]==start for row in rows)


def test_model_answer_eligible_citation_and_withdrawn_source(tmp_path,monkeypatch):
    from app.ollama import Generated
    from app.lifecycle import withdraw
    path=tmp_path/'runtime';rid=seal(path,build(path));db=connect(path)
    cp,ep,rp=fixture_records(tmp_path,rid,db)
    mark_ready(path,rid,cp,ep);activate(path,rid,None,rp)
    monkeypatch.setattr(service,'DB',path)
    async def fake(message,evidence,history):
        assert evidence
        return Generated(answer=evidence[0]['text'],source_ids=[evidence[0]['source_id']]),{}
    monkeypatch.setattr(service.adapter,'generate',fake)
    client=TestClient(app)
    sid=client.post('/api/v1/sessions',json={}).json()['session_id']
    result=client.post(f'/api/v1/sessions/{sid}/turns',json={'client_turn_id':'answer','message':'充電接點','product_id':'hp-a'})
    assert result.status_code==200 and result.json()['status']=='answer'
    citation=result.json()['citations'][0]
    assert citation['doc_id'].startswith('hp-a') and len(citation['source_id'])==128
    source=client.get(citation['url'])
    assert source.status_code==200 and source.json()['release_id']==rid
    withdraw(path,citation['doc_id'])
    assert client.get(citation['url']).json()['text'] is None


def test_session_limit_expiry_and_transport_errors(tmp_path,monkeypatch):
    from datetime import datetime,timedelta,timezone
    from app.ollama import TurnTimeout,CapacityError,DependencyError
    path=tmp_path/'runtime';rid=build(path);db=connect(path)
    cp,ep,rp=fixture_records(tmp_path,rid,db);mark_ready(path,rid,cp,ep);activate(path,rid,None,rp)
    monkeypatch.setattr(service,'DB',path)
    client=TestClient(app);sid=client.post('/api/v1/sessions',json={}).json()['session_id'];endpoint=f'/api/v1/sessions/{sid}/turns'
    payload={'client_turn_id':'try','message':'配對','product_id':'hp-a'}
    for exception,status in [(TurnTimeout('timeout'),504),(CapacityError('full'),429),(DependencyError('bad'),503)]:
        async def fail(*args,exc=exception,**kwargs):raise exc
        monkeypatch.setattr(service,'respond',fail)
        response=client.post(endpoint,json=payload)
        assert response.status_code==status and not sessions[sid]['completed']
        if status==429:assert int(response.headers['retry-after'])>0
    sessions[sid]['completed']={str(i):(None,None) for i in range(20)}
    assert client.post(endpoint,json=payload).json()['error']['code']=='session_limit'
    sessions[sid]['expires']=datetime.now(timezone.utc)-timedelta(seconds=1)
    assert client.post(endpoint,json=payload).status_code==404


def test_replay_does_not_resurrect_withdrawn_answer(tmp_path,monkeypatch):
    from app.ollama import Generated
    from app.lifecycle import withdraw
    path=tmp_path/'db';rid=build(path);db=connect(path);cp,ep,rp=fixture_records(tmp_path,rid,db)
    mark_ready(path,rid,cp,ep);activate(path,rid,None,rp);monkeypatch.setattr(service,'DB',path)
    async def fake(message,evidence,history):return Generated(answer=evidence[0]['text'],source_ids=[evidence[0]['source_id']]),{}
    monkeypatch.setattr(service.adapter,'generate',fake)
    client=TestClient(app);sid=client.post('/api/v1/sessions',json={}).json()['session_id'];url=f'/api/v1/sessions/{sid}/turns'
    body={'client_turn_id':'same','message':'配對','product_id':'hp-a'}
    value=client.post(url,json=body);assert value.status_code==200
    withdraw(path,value.json()['citations'][0]['doc_id'])
    assert client.post(url,json=body).status_code==503
    assert client.get('/api/v1/sources/'+'0'*128).status_code==404
