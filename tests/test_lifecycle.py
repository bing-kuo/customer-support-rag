"""Approval records in this module are SYNTHETIC UNIT TEST FIXTURES only."""
import json
import pytest
from app.knowledge import build, seal, connect, ROOT, digest
from app.lifecycle import canonical, corpus_identity, mark_ready, activate, pin, withdraw


def fixture_records(tmp_path, release, db):
    manifest=json.loads(db.execute('SELECT manifest FROM releases WHERE id=?',(release,)).fetchone()[0])
    corpus={'reviewer':'Marcus','approved_at':'2026-11-08T00:00:00+00:00','fixture':True,'manifest_sha256':manifest['source_manifest_sha256'],'corpus_sha256':corpus_identity(manifest['documents']),'document_hashes':{d['doc_id']+':'+d['revision']:d['content_sha256'] for d in manifest['documents']}}
    frozen=__import__('app.evaluation',fromlist=['FROZEN']).FROZEN
    regression=json.loads((frozen/'regression.json').read_bytes())
    holdout=[]
    for category in ['normal','wrong_version','missing_info','no_answer','injection']:
        examples=[q for q in regression if q['category']==category]
        for i,q in enumerate(examples[:6]):
            holdout.append(dict(q,id=f'synthetic-fixture-{category}-{i}'))
    def results(dataset):
        values=[]
        for q in dataset:
            ids=[];answer=None
            if q['gold']:
                g=q['gold'][0]
                row=db.execute('SELECT c.id FROM chunks c JOIN provenance p ON p.release=c.release AND p.id=c.id WHERE c.release=? AND c.doc=? AND p.revision=? AND instr(c.text,?)>0',(release,g['doc_id'],g['revision'],g['span'])).fetchone()
                ids=[row[0]];answer=g['span']
            from app.planning import search as search_candidate
            # Reproduce actual eligible lexical ranking for synthetic unit evidence.
            # The active test DB path is obtained from SQLite, never from the report.
            path=db.execute('PRAGMA database_list').fetchone()[2]
            from app.service import retrieval_context
            query,_,scope=retrieval_context(path,release,q)
            ranked=[r[0] for r in search_candidate(path,release,query,scope,q['on_date'],q['locale'])]
            values.append({'id':q['id'],'status':q['expected'],'retrieved':ranked,'citations':ids,'answer':answer,'request_sha256':digest(canonical({'release':release,'configuration':manifest['configuration_sha256'],'case':q}))})
        return values
    suites={name:{'dataset':dataset,'dataset_sha256':digest(canonical(dataset)),'results':results(dataset),'reviewer':'Marcus'} for name,dataset in [('regression',regression),('holdout',holdout)]}
    from app.evaluation import normalize_dataset,verification_sources,verification_commands,SCANNER_IMAGE
    for suite in suites.values():
        suite['dataset_raw']=canonical(suite['dataset']).decode()
        suite['dataset_adapter']=normalize_dataset(suite['dataset_raw'].encode())[1]
    evaluation={'release_id':release,'manifest_sha256':release,'configuration_sha256':manifest['configuration_sha256'],'suites':suites,'source_fingerprints':verification_sources(),'holdout_sha256':suites['holdout']['dataset_sha256'],'fixture':True}
    # These raw check artifacts are explicit synthetic unit fixtures, not real QA.
    from app.evaluation import EVIDENCE_ROOT,REQUIRED_TESTS
    test_xml=('<testsuites><testsuite errors="0" failures="0" skipped="0">'+''.join('<testcase name="'+name+'"/>' for name in sorted(REQUIRED_TESTS))+'</testsuite></testsuites>').encode()
    scan=b'[]'
    check={'scanner_image':SCANNER_IMAGE,'release_id':release,'source_fingerprints':verification_sources(),'commands':verification_commands(),'test_exit':0,'scan_exit':0,'configuration_sha256':manifest['configuration_sha256'],'test_xml_sha256':digest(test_xml),'gitleaks_report_sha256':digest(scan),'fixture':'SYNTHETIC UNIT EVIDENCE'}
    from scripts.scan_public_sources import SCAN_POLICY
    scan_files=dict(verification_sources());scan_files.update({'app/'+name:value for name,value in __import__('app.knowledge',fromlist=['configuration']).configuration()['code_sha256'].items()})
    scan_manifest=canonical({'policy':SCAN_POLICY,'files':scan_files,'fixture':'SYNTHETIC UNIT ONLY'})
    check.update(scan_scope=SCAN_POLICY,scan_manifest_sha256=digest(scan_manifest))
    check_raw=canonical(check);check_id=digest(check_raw);folder=EVIDENCE_ROOT/check_id;folder.mkdir(parents=True,exist_ok=True)
    (folder/'verification.json').write_bytes(check_raw);(folder/'pytest.xml').write_bytes(test_xml);(folder/'gitleaks.json').write_bytes(scan);(folder/'scan-manifest.json').write_bytes(scan_manifest)
    evaluation['verification_sha256']=check_id
    evaluation['corpus_approval_sha256']=digest(canonical(corpus))
    cp=tmp_path/'fixture-corpus.json';ep=tmp_path/'fixture-eval.json';rp=tmp_path/'fixture-release.json'
    cp.write_bytes(canonical(corpus));ep.write_bytes(canonical(evaluation))
    from app.ollama import sentence_units
    def manual_suite(suite):
        audits=[{'id':row['id'],'result_sha256':digest(canonical(row)),'relevant':True,'qualifications_complete':True,'critical_instructions_complete':True,'applicability_rationale':'SYNTHETIC UNIT FIXTURE ONLY; not a genuine review','claims':[{'text':text,'source_ids':row['citations'],'supported':True,'relevant':True,'rationale':'SYNTHETIC UNIT FIXTURE ONLY; no real semantic approval'} for text in sentence_units(row.get('answer') or '')]} for row in suite['results']]
        n=sum(len(a['claims']) for a in audits)
        return {'dataset_sha256':suite['dataset_sha256'],'supported_claims':{'numerator':n,'denominator':n},'relevance_reviewed':True,'qualifications_reviewed':True,'reviewed_result_ids':[r['id'] for r in suite['results']],'case_reviews':audits,'relevance':{'numerator':len(audits),'denominator':len(audits)},'critical_failure_count':0,'critical_failed_case_ids':[]}
    rp.write_bytes(canonical({'reviewer':'Marcus','approved_at':corpus['approved_at'],'fixture':True,'release_id':release,'manifest_sha256':release,'corpus_approval_sha256':digest(cp.read_bytes()),'evaluation_sha256':digest(ep.read_bytes()),'holdout_sha256':evaluation['holdout_sha256'],'manual_semantic_review':{'reviewer':'Marcus','independent':True,'release_id':release,'configuration_sha256':manifest['configuration_sha256'],'evaluation_sha256':digest(ep.read_bytes()),'suites':{name:manual_suite(suite) for name,suite in suites.items()}}}))
    return cp,ep,rp


def test_cas_rollback_and_withdrawal(tmp_path):
    path=tmp_path/'db';release=seal(path,build(path));db=connect(path)
    cp,ep,rp=fixture_records(tmp_path,release,db)
    assert pin(path) is None
    with pytest.raises(ValueError):
        activate(path,release,None,rp)
    mark_ready(path,release,cp,ep)
    activate(path,release,None,rp)
    assert pin(path)==release
    with pytest.raises(ValueError,match='conflict'):
        activate(path,release,None,rp)
    withdraw(path,'hp-a-pairing')
    with pytest.raises(ValueError,match='Withdrawn|thresholds|replay'):
        activate(path,release,release,rp)
    assert pin(path)==release


def test_metadata_approval_and_storage_tamper_rejected(tmp_path):
    path=tmp_path/'db';release=seal(path,build(path));db=connect(path)
    cp,ep,rp=fixture_records(tmp_path,release,db)
    value=json.loads(cp.read_bytes());value['corpus_sha256']='0'*64;cp.write_bytes(canonical(value))
    with pytest.raises(ValueError,match='corpus'):
        mark_ready(path,release,cp,ep)
    db.execute('UPDATE chunks SET text=? WHERE release=?',('tampered',release));db.commit()
    with pytest.raises(ValueError,match='chunks'):
        pin_target=__import__('app.lifecycle',fromlist=['verify']).verify(db,release)
    assert pin(path) is None


def test_build_failure_rolls_back_whole_candidate(tmp_path,monkeypatch):
    import app.knowledge as k
    path=tmp_path/'db';existing=build(path)
    docs=k.load()
    monkeypatch.setattr(k,'load',lambda:[(docs[0][0],'## Fault fixture\n'+('x'*241+'\n')*1100)])
    with pytest.raises(ValueError,match='Chunk limit'):
        build(path)
    db=connect(path)
    assert db.execute('SELECT id FROM releases').fetchall()==[(existing,)]
    assert db.execute('SELECT count(*) FROM chunks WHERE release!=?',(existing,)).fetchone()[0]==0
