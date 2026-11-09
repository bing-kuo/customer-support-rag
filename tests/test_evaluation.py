"""All generated approvals/reports are synthetic unit fixtures, never release evidence."""
import json
import pytest
from app.knowledge import build,connect
from app.lifecycle import mark_ready,canonical,pin
from tests.test_lifecycle import fixture_records

@pytest.mark.parametrize('mutation',['lying','empty','config','dataset','missing_category','no_answers'])
def test_evaluation_cannot_trust_passed_boolean(tmp_path,mutation):
    path=tmp_path/'db';rid=build(path);db=connect(path)
    cp,ep,rp=fixture_records(tmp_path,rid,db)
    value=json.loads(ep.read_bytes());value['passed']=True
    if mutation=='lying':
        value['suites']['regression']['results'][0]['answer']='保證退款與不存在的承諾'
    elif mutation=='empty':
        value['suites']['regression']['results']=[]
    elif mutation=='config':
        value['configuration_sha256']='0'*64
    elif mutation=='dataset':
        value['suites']['regression']['dataset'][0]['message']='modified'
    elif mutation=='missing_category':
        value['suites']['holdout']['dataset']=[]
    else:
        for row in value['suites']['regression']['results']:
            row.update(status='handoff',answer=None,citations=[])
    ep.write_bytes(canonical(value))
    with pytest.raises(ValueError):mark_ready(path,rid,cp,ep)
    assert pin(path) is None


def test_real_corpus_approval_exact_set_and_metadata(tmp_path):
    from app.lifecycle import record,corpus_identity
    from app.knowledge import ROOT
    path=ROOT/'approvals'/'marcus-synthetic-corpus-v2-2026-11-08.json'
    if not path.exists():pytest.skip('Independent review pending')
    approval,_=record(path)
    manifest=json.loads((ROOT/'manifest.json').read_bytes())
    assert approval['corpus_sha256']==corpus_identity(manifest)
    assert approval['document_hashes']=={d['doc_id']+':'+d['revision']:d['content_sha256'] for d in manifest}
    changed=[dict(d) for d in manifest];changed[0]['title']='changed metadata'
    assert approval['corpus_sha256']!=corpus_identity(changed)


def test_handpicked_gold_ranking_cannot_pass_as_actual_lexical_run(tmp_path):
    from app.evaluation import score,FROZEN
    path=tmp_path/'db';rid=build(path);db=connect(path)
    cp,ep,rp=fixture_records(tmp_path,rid,db);report=json.loads(ep.read_bytes());suite=report['suites']['regression']
    row=next(r for r in suite['results'] if r['retrieved'])
    foreign=db.execute('SELECT id FROM chunks WHERE release=? AND id!=? LIMIT 1',(rid,row['retrieved'][0])).fetchone()[0]
    row['retrieved']=[foreign]
    with pytest.raises(ValueError,match='replay'):score(db,rid,suite['dataset'],suite['results'])
    # Dense comparator is a separate offline metric, never an activation mode.
    metrics=score(db,rid,suite['dataset'],suite['results'],retrieval_mode='dense')
    assert 'recall_at_5' in metrics['metrics']
    ep.write_bytes(canonical(report))
    with pytest.raises(ValueError,match='replay'):mark_ready(path,rid,cp,ep)


def test_verification_hash_and_mandatory_tests_are_not_optional(tmp_path):
    from app.evaluation import EVIDENCE_ROOT
    path=tmp_path/'db';rid=build(path);db=connect(path);cp,ep,rp=fixture_records(tmp_path,rid,db)
    value=json.loads(ep.read_bytes());value.pop('verification_sha256')
    ep.write_bytes(canonical(value))
    with pytest.raises(ValueError,match='verification'):mark_ready(path,rid,cp,ep)
