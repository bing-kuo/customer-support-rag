"""Synthetic check/approval fixtures only; no genuine review records are modified."""
import json
import shutil
from pathlib import Path
import pytest
from app import evaluation
from app.knowledge import build,connect,digest
from app.lifecycle import canonical,mark_ready,activate
from tests.test_lifecycle import fixture_records


@pytest.mark.parametrize('changed',['tests/test_evaluation.py','scripts/verify_local.py','scripts/evaluate_candidate.py','.gitleaks.toml','requirements.lock'])
def test_same_names_but_changed_sources_cannot_reuse_check_evidence(tmp_path,monkeypatch,changed):
    path=tmp_path/'db';rid=build(path);db=connect(path);cp,ep,rp=fixture_records(tmp_path,rid,db)
    report=json.loads(ep.read_bytes())
    root=tmp_path/'source-copy';root.mkdir()
    for relative in evaluation.verification_sources():
        target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(evaluation.SOURCE_ROOT/relative,target)
    monkeypatch.setattr(evaluation,'SOURCE_ROOT',root)
    assert evaluation.verify_checks(report,report['configuration_sha256'],rid)
    with (root/changed).open('a') as f:f.write('\n# Synthetic source change while test names remain unchanged.\n')
    with pytest.raises(ValueError,match='configuration'):evaluation.verify_checks(report,report['configuration_sha256'],rid)


@pytest.mark.parametrize('kind',['missing','low_support','wrong_report','wrong_dataset','missing_relevance','missing_case'])
def test_exact_span_pass_never_replaces_independent_semantic_review(tmp_path,kind):
    path=tmp_path/'db';rid=build(path);db=connect(path);cp,ep,rp=fixture_records(tmp_path,rid,db)
    mark_ready(path,rid,cp,ep)
    approval=json.loads(rp.read_bytes())
    review=approval['manual_semantic_review']
    if kind=='missing':approval.pop('manual_semantic_review')
    elif kind=='low_support':review['suites']['holdout']['supported_claims']={'numerator':94,'denominator':100}
    elif kind=='wrong_report':review['evaluation_sha256']='0'*64
    elif kind=='wrong_dataset':review['suites']['holdout']['dataset_sha256']='0'*64
    elif kind=='missing_relevance':review['suites']['holdout']['relevance_reviewed']=False
    else:review['suites']['holdout']['reviewed_result_ids'].pop()
    rp.write_bytes(canonical(approval))
    with pytest.raises(ValueError):activate(path,rid,None,rp)


def test_altered_execution_command_cannot_be_resigned_by_self_consistent_hash(tmp_path):
    path=tmp_path/'db';rid=build(path);db=connect(path);cp,ep,rp=fixture_records(tmp_path,rid,db)
    report=json.loads(ep.read_bytes());folder=evaluation.EVIDENCE_ROOT/report['verification_sha256']
    value=json.loads((folder/'verification.json').read_bytes());value['commands']['test']=['true']
    raw=canonical(value);new_id=digest(raw);target=evaluation.EVIDENCE_ROOT/new_id;shutil.copytree(folder,target)
    (target/'verification.json').write_bytes(raw);report['verification_sha256']=new_id
    with pytest.raises(ValueError,match='configuration'):evaluation.verify_checks(report,report['configuration_sha256'],rid)
