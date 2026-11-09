"""Public synthetic shapes only; no access to the sealed private holdout."""
import copy
import hashlib
import json
import pytest
from app.evaluation import normalize_dataset,verify_dataset_suite
from app.lifecycle import canonical

RAW=b'[{"id":"example","category":"missing_info","message":"synthetic","product_id":null,"on_date":"2026-11-08","locale":"zh-TW","expected":"clarify","gold":[],"prior_turns":[]}]'


def suite(raw):
    data,proof=normalize_dataset(raw)
    return {'dataset':data,'dataset_adapter':proof,'dataset_raw':raw.decode(),'dataset_sha256':hashlib.sha256(raw).hexdigest()}


def test_both_formats_preserve_raw_bytes_and_equal_normalized_representation():
    assert hashlib.sha256(RAW).hexdigest()=='607a85a5e707db3d3c376464e0da83e398d2c31ee39d3022c34c5aecc54992e0'
    original=json.loads(RAW)[0]
    aliased=dict(original,category='missing_information',expected_status=original['expected'])
    aliased.pop('expected')
    raw=json.dumps({'cases':[aliased]},indent=2).encode()
    left=suite(RAW);right=suite(raw)
    assert left['dataset']==right['dataset']
    assert left['dataset_adapter']['normalized_sha256']==right['dataset_adapter']['normalized_sha256']
    assert left['dataset_sha256']!=right['dataset_sha256']
    assert right['dataset_adapter']['source_shape']=='object.cases'
    assert verify_dataset_suite(right)[0]==raw


def test_preserves_prior_turns_and_gold_date_product_locale_without_mutation():
    q=json.loads(RAW)[0]
    q.update(category='unanswerable',expected='handoff',product_id='hp-b',on_date='2026-07-01',prior_turns=[{'message':'synthetic prior','product_id':None}],gold=[{'doc_id':'synthetic','revision':'3','span':'合成片段'}])
    envelope={'cases':[q]};before=copy.deepcopy(envelope)
    normalized,_=normalize_dataset(json.dumps(envelope).encode())
    assert envelope==before
    assert normalized[0]['category']=='no_answer'
    for key in ['prior_turns','gold','on_date','product_id','locale']:
        assert normalized[0][key]==q[key]


@pytest.mark.parametrize('kind',['conflict','unknown','duplicate','empty','empty_id','duplicate_json_key'])
def test_rejects_invalid_or_ambiguous_shape(kind):
    q=json.loads(RAW)[0];value={'cases':[q]}
    if kind=='conflict':q['expected_status']='handoff'
    if kind=='unknown':q['category']='not-a-category'
    if kind=='duplicate':value['cases'].append(dict(q))
    if kind=='empty':value['cases']=[]
    if kind=='empty_id':q['id']=''
    raw=json.dumps(value).encode()
    if kind=='duplicate_json_key':raw=b'{"cases":[],"cases":[]}'
    with pytest.raises(ValueError):normalize_dataset(raw)


@pytest.mark.parametrize('kind',['raw','normalized','declared_raw_hash','version','normalized_hash'])
def test_gate_recomputes_all_provenance_not_declared_hashes(kind):
    value=suite(RAW)
    if kind=='raw':value['dataset_raw']=value['dataset_raw'].replace('synthetic','changed')
    if kind=='normalized':value['dataset'][0]['prior_turns']=[{'message':'changed'}]
    if kind=='declared_raw_hash':value['dataset_sha256']='0'*64
    if kind=='version':value['dataset_adapter']['version']='invented'
    if kind=='normalized_hash':value['dataset_adapter']['normalized_sha256']='0'*64
    with pytest.raises(ValueError):verify_dataset_suite(value)


def test_owner_preflight_is_read_only_and_has_no_questions_or_inference(tmp_path,monkeypatch,capsys):
    from scripts import evaluate_candidate as runner
    q=json.loads(RAW)[0];cases=[]
    for category in ['normal','wrong_version','missing_information','unanswerable','injection']:
        for index in range(6):cases.append(dict(q,id=category+str(index),category=category))
    raw=canonical({'cases':cases});path=tmp_path/'synthetic-sealed.json';path.write_bytes(raw)
    async def forbidden(*args,**kwargs):raise AssertionError('No inference permitted')
    monkeypatch.setattr(runner.adapter,'embed',forbidden)
    proof=runner.dataset_preflight(path,hashlib.sha256(raw).hexdigest())
    assert proof['case_count']==30 and proof['category_counts']['missing_info']==6
    assert 'synthetic' not in json.dumps(proof) and 'message' not in proof
    assert path.read_bytes()==raw
    assert not capsys.readouterr().out
