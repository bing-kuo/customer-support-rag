"""Private-runner contract uses synthetic unit questions, never Marcus holdout."""
import asyncio
import json
from types import SimpleNamespace
import numpy as np
import pytest
from app.knowledge import build,connect,ROOT,digest
from app.lifecycle import canonical
from app.ollama import Generated
from tests.test_lifecycle import fixture_records


@pytest.mark.parametrize('shape',['array','object.cases'])
def test_private_holdout_runner_keeps_cases_out_of_stdout(tmp_path,monkeypatch,capsys,shape):
    from scripts import evaluate_candidate as runner
    from app import evaluation,service
    path=tmp_path/'fixture-db';rid=build(path);db=connect(path)
    cp,ep,rp=fixture_records(tmp_path,rid,db)
    unit_report=json.loads(ep.read_bytes())
    unit=unit_report['suites']['holdout']['dataset']
    monkeypatch.setattr(runner,'VERIFICATION_FILE',evaluation.EVIDENCE_ROOT/unit_report['verification_sha256']/'verification.json')
    if shape=='object.cases':
        unit=[dict(q,expected_status=q['expected'],category={'missing_info':'missing_information','no_answer':'unanswerable'}.get(q['category'],q['category']),prior_turns=[]) for q in unit]
        for q in unit:q.pop('expected')
    raw=canonical(unit if shape=='array' else {'cases':unit});dataset=tmp_path/'private-unit-only.json';dataset.write_bytes(raw)
    output=tmp_path/'private-unit-report.json'
    runtime=tmp_path/'runtime';runtime.mkdir()
    monkeypatch.setattr(runner,'ROOT',runtime)
    monkeypatch.setattr(runner,'FROZEN',evaluation.FROZEN)
    monkeypatch.setattr(service,'DB',runtime/'runtime.sqlite3')
    async def fake_embed(texts):return np.ones((len(texts),1024))/32
    async def fake_generate(message,evidence,history):
        return Generated(answer=evidence[0]['text'],source_ids=[evidence[0]['source_id']]),{}
    async def fake_lifecycle(*args,**kwargs):pass
    monkeypatch.setattr(runner.adapter,'begin_owned_batch',fake_lifecycle)
    monkeypatch.setattr(runner.adapter,'unload_owned',fake_lifecycle)
    monkeypatch.setattr(runner.adapter,'finish_owned_batch',fake_lifecycle)
    monkeypatch.setattr(runner.adapter,'preflight',lambda scope:None)
    monkeypatch.setattr(runner.adapter,'embed',fake_embed)
    monkeypatch.setattr(service.adapter,'generate',fake_generate)
    args=SimpleNamespace(split='holdout',dataset_file=str(dataset),expected_dataset_sha256=digest(raw),output=str(output),corpus_approval=str(cp),candidate=None)
    asyncio.run(runner.run(args))
    result=json.loads(output.read_bytes())
    assert result['holdout_sha256']==digest(raw)
    assert result['suites']['holdout']['dataset_adapter']['source_shape']==shape
    assert result['suites']['holdout']['dataset_raw'].encode()==raw
    captured=capsys.readouterr().out
    assert all(case['message'] not in captured for case in unit)
    assert not list((ROOT/'approvals').glob('*unit*'))


def test_private_output_cannot_replace_sealed_input(tmp_path,monkeypatch):
    from scripts import evaluate_candidate as runner
    cases=[]
    for category in ['normal','wrong_version','missing_info','no_answer','injection']:
        for i in range(6):
            cases.append({'id':category+str(i),'category':category,'message':'synthetic','product_id':None,'on_date':'2026-11-08','locale':'zh-TW','expected':'clarify','gold':[],'prior_turns':[]})
    raw=canonical({'cases':cases});path=tmp_path/'sealed-synthetic.json';path.write_bytes(raw)
    async def forbidden(*args,**kwargs):raise AssertionError('No inference allowed')
    monkeypatch.setattr(runner.adapter,'embed',forbidden)
    args=SimpleNamespace(split='holdout',corpus_approval='synthetic-unused',dataset_file=str(path),expected_dataset_sha256=digest(raw),output=str(path))
    with pytest.raises(ValueError,match='overwrite'):asyncio.run(runner.run(args))
    assert path.read_bytes()==raw
