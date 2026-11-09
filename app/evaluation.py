"""Evidence-based evaluation. Counters are recomputed; report booleans are ignored."""
import json
from collections import Counter
from pathlib import Path
from app.knowledge import digest, configuration, connect
from app.lifecycle import canonical

FROZEN=Path(__file__).resolve().parents[1]/'eval'/'frozen-v4'
CATEGORIES={'normal','wrong_version','missing_info','no_answer','injection'}
# These values come from the approved Develop plan, never calibrated on holdout.
THRESHOLDS={'recall_at_5':.90,'mrr_at_5':.80,'exact_span_claims':.95,'citations_correct':1.,'clarify':.90,'handoff':.90}


DATASET_ADAPTER_VERSION='readonly-cases-v1'


def normalize_dataset(raw):
    """Normalize aliases in memory while retaining exact sealed-byte provenance."""
    import copy
    from datetime import date
    def unique(pairs):
        value={}
        for key,item in pairs:
            if key in value:raise ValueError('Duplicate JSON key')
            value[key]=item
        return value
    value=json.loads(raw.decode('utf-8'),object_pairs_hook=unique)
    shape='array' if isinstance(value,list) else 'object.cases'
    cases=value if isinstance(value,list) else value.get('cases') if isinstance(value,dict) else None
    if not isinstance(cases,list) or not cases:raise ValueError('Empty or invalid dataset')
    normalized=[];seen=set()
    for original in cases:
        if not isinstance(original,dict):raise ValueError('Invalid case')
        q=copy.deepcopy(original)
        if 'expected_status' in q:
            if 'expected' in q and q['expected']!=q['expected_status']:
                raise ValueError('Conflicting expected aliases')
            q['expected']=q.pop('expected_status')
        if not isinstance(q.get('category'),str):raise ValueError('Unknown category type')
        q['category']={'missing_information':'missing_info','unanswerable':'no_answer'}.get(q.get('category'),q.get('category'))
        if q.get('category') not in CATEGORIES or q.get('expected') not in ('answer','clarify','handoff'):
            raise ValueError('Unknown evaluation category or status')
        ident=q.get('id')
        if not isinstance(ident,str) or not ident or ident in seen:raise ValueError('Duplicate or invalid case ID')
        seen.add(ident)
        if not isinstance(q.get('message'),str) or not q['message'] or q.get('product_id') not in (None,'hp-a','hp-b') or q.get('locale')!='zh-TW':
            raise ValueError('Invalid case boundary')
        if not isinstance(q.get('on_date'),str) or date.fromisoformat(q['on_date']).isoformat()!=q['on_date']:
            raise ValueError('Invalid case date')
        if not isinstance(q.get('gold'),list) or any(not isinstance(g,dict) or any(not isinstance(g.get(k),str) or not g[k] for k in ('doc_id','revision','span')) for g in q['gold']):
            raise ValueError('Invalid gold evidence')
        if 'prior_turns' in q and not isinstance(q['prior_turns'],list):raise ValueError('Invalid prior turns')
        normalized.append(q)
    return normalized,{'version':DATASET_ADAPTER_VERSION,'source_shape':shape,'raw_sha256':digest(raw),'normalized_sha256':digest(canonical(normalized))}


def verify_dataset_suite(suite):
    raw=suite.get('dataset_raw')
    if not isinstance(raw,str):raise ValueError('Original dataset bytes required')
    raw=raw.encode('utf-8')
    normalized,provenance=normalize_dataset(raw)
    if suite.get('dataset_sha256')!=digest(raw) or suite.get('dataset')!=normalized or suite.get('dataset_adapter')!=provenance:
        raise ValueError('Dataset normalization or hash mismatch')
    return raw,normalized


def ratios(counters):
    return {key:{'numerator':round(v[0],8),'denominator':v[1],'value':v[0]/v[1] if v[1] else None} for key,v in counters.items()}


def score(db,release,dataset,results,retrieval_mode="lexical"):
    from app.lifecycle import verify
    manifest=verify(db,release)
    if not dataset or len({q['id'] for q in dataset})!=len(dataset) or len(results)!=len(dataset):
        raise ValueError('Missing or duplicate evaluation cases')
    by_id={r['id']:r for r in results}
    if set(by_id)!={q['id'] for q in dataset} or len(by_id)!=len(results):
        raise ValueError('Evaluation result identity mismatch')
    has_denylist=db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='withdrawn'").fetchone()
    denied={r[0] for r in db.execute('SELECT doc FROM withdrawn')} if has_denylist else set()
    counters={k:[0.,0] for k in THRESHOLDS}
    violations={'wrong_product':0,'withdrawn_leak':0,'refund_promise':0,'invalid_evidence':0,'critical_instruction_omissions':0}
    docs={(d['doc_id'],d['revision']):d for d in manifest['documents']}
    from app.service import retrieval_context
    database_path=db.execute('PRAGMA database_list').fetchone()[2]
    contexts={q['id']:retrieval_context(database_path,release,q) for q in dataset}
    def resolved(cid,q):
        row=db.execute('SELECT c.doc,p.revision,c.text,c.product,c.locale,c.start,c.end FROM chunks c JOIN provenance p ON p.release=c.release AND p.id=c.id WHERE c.release=? AND c.id=?',(release,cid)).fetchone()
        if not row:
            return None
        products=json.loads(row[3])
        from app import planning
        common={cid for item in planning.obligations(database_path,release,contexts[q['id']][0],contexts[q['id']][1],q['on_date'],q['locale']) if item.get('require_all_sources') for cid in item['source_ids']}
        if products and contexts[q['id']][1] not in products and cid not in common:
            violations['wrong_product']+=1
            return None
        if row[4]!=q['locale'] or row[5]>q['on_date'] or (row[6] and row[6]<=q['on_date']):
            violations['invalid_evidence']+=1
            return None
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='withdrawn'").fetchone() and db.execute('SELECT 1 FROM withdrawn WHERE doc=?',(row[0],)).fetchone():
            violations['withdrawn_leak']+=1
            return None
        return row
    for q in dataset:
        r=by_id[q['id']]
        if r.get('request_sha256')!=digest(canonical({'release':release,'configuration':manifest['configuration_sha256'],'case':q})):
            raise ValueError('Evaluation request mismatch')
        from app import planning
        expected_rank=[row[0] for row in planning.search(database_path,release,contexts[q['id']][0],contexts[q['id']][1],q['on_date'],q['locale'])]
        ranked=r['retrieved']
        if retrieval_mode not in ('lexical','dense'):
            raise ValueError('Unknown retrieval evaluator')
        if retrieval_mode=='lexical' and ranked!=expected_rank:
            raise ValueError('Retrieval results do not replay')
        if len(ranked)>10 or len(set(ranked))!=len(ranked):
            raise ValueError('Invalid ranked evidence')
        resolved_rows=[resolved(cid,q) for cid in ranked[:5]]
        if q['gold']:
            rank=next((i+1 for i,row in enumerate(resolved_rows) if row and any(row[0]==g['doc_id'] and row[1]==g['revision'] and g['span'] in row[2] for g in q['gold'])),None)
            counters['recall_at_5'][1]+=1;counters['mrr_at_5'][1]+=1
            if rank:
                counters['recall_at_5'][0]+=1;counters['mrr_at_5'][0]+=1/rank
        required=planning.obligations(database_path,release,contexts[q['id']][0],contexts[q['id']][1],q['on_date'],q['locale'])
        violations['critical_instruction_omissions']+=len(planning.critical_failures(r,required))
        for action in q.get('required_actions',[]):
            valid=r.get('status')=='answer' and action['span'] in (r.get('answer') or '') and any((source:=resolved(cid,q)) and source[0]==action['doc_id'] and source[1]==action['revision'] and action['span'] in source[2] for cid in r.get('citations',[]))
            if not valid:violations['critical_instruction_omissions']+=1
        expected=q['expected']
        if expected in ('clarify','handoff'):
            counters[expected][1]+=1;counters[expected][0]+=int(r['status']==expected)
        answer=r.get('answer')
        citations=r['citations']
        if r['status']=='answer':
            if not answer or not citations:
                violations['invalid_evidence']+=1
            sources=[]
            for cid in citations:
                row=resolved(cid,q)
                counters['citations_correct'][1]+=1
                valid=row is not None and cid in expected_rank[:4]
                counters['citations_correct'][0]+=int(valid)
                if valid:sources.append(row[2])
            # Deterministic exact-span support, not an LLM judge. This deliberately
            # rejects paraphrases that need independent human semantic adjudication.
            claims=[s.strip() for s in (answer or '').split('。') if s.strip()]
            for claim in claims:
                counters['exact_span_claims'][1]+=1
                counters['exact_span_claims'][0]+=int(any(claim in text for text in sources))
            if any(term in (answer or '').lower() for term in ['保證退款','已退款','refund guaranteed','refund completed']):
                violations['refund_promise']+=1
        elif answer or citations:
            violations['invalid_evidence']+=1
        if expected=='answer' and r['status']!='answer':
            counters['exact_span_claims'][1]+=1
            counters['citations_correct'][1]+=1
    metrics=ratios(counters)
    passed=all(metrics[k]['value'] is not None and metrics[k]['value']>=v for k,v in THRESHOLDS.items()) and not any(violations.values())
    return {'metrics':metrics,'violations':violations,'passed':passed,'semantic_status':'pending_independent_Marcus_review','scope':'mechanical exact-span checks only'}


def validate_report(db,release,report):
    from app.lifecycle import verify
    manifest=verify(db,release)
    if report.get('release_id')!=release or report.get('manifest_sha256')!=release or report.get('configuration_sha256')!=manifest['configuration_sha256'] or manifest['configuration_sha256']!=digest(canonical(configuration())):
        raise ValueError('Evaluation configuration mismatch')
    verify_checks(report,manifest['configuration_sha256'],release)
    suites=report.get('suites',{})
    if set(suites)!={'regression','holdout'}:
        raise ValueError('Regression and independent holdout required')
    frozen=json.loads((FROZEN/'manifest.json').read_bytes())
    if digest((FROZEN/'manifest.json').read_bytes())!=manifest['frozen_split_manifest_sha256']:
        raise ValueError('Frozen split manifest mismatch')
    if frozen['corpus_manifest_sha256']!=manifest['source_manifest_sha256']:
        raise ValueError('Frozen corpus mismatch')
    for name,suite in suites.items():
        raw,normalized=verify_dataset_suite(suite)
        if name=='regression':
            if digest(raw)!=frozen['datasets']['regression']['sha256'] or raw!=(FROZEN/'regression.json').read_bytes():
                raise ValueError('Frozen regression mismatch')
            expected=frozen['datasets']['regression']['categories']
        else:
            # Marcus alone supplies this frozen split and its independent provenance.
            if suite.get('reviewer')!='Marcus' or suite.get('dataset_sha256')!=report.get('holdout_sha256'):
                raise ValueError('Independent holdout binding required')
            expected=dict.fromkeys(CATEGORIES,6)
        if Counter(q['category'] for q in suite['dataset'])!=expected:
            raise ValueError('Missing evaluation categories')
        if not score(db,release,suite['dataset'],suite['results'])['passed']:
            raise ValueError('Evaluation thresholds failed')
    return True


EVIDENCE_ROOT=Path(__file__).resolve().parents[1]/'eval'/'checks'
REQUIRED_TESTS={'test_cloud_missing_or_exhausted_approval_never_reaches_transport','test_official_request_shape_usage_identity_and_redacted_ledger','test_reservation_atomic_concurrency_restart_and_policy_mutation','test_sent_cancel_timeout_retains_unknown_until_confirmed_terminal','test_embedding_verifies_only_own_model_and_rejects14b_unload','test_unknown_product_task_requires_slot_instead_of_general_policy','test_approved_critical_instruction_survives_missing_slot_and_handoff_intent','test_critical_obligation_checks_every_status_even_verbatim_irrelevant_answer','test_manual99percent_support_cannot_hide_critical_handoff_omission','test_exact_span_and_eligible_citation_do_not_excuse_missing_stop_use','test_owned_phases_unload_confirmed_and_finite_keep_alive','test_preexisting_model_even_same_digest_is_not_owned','test_unconfirmed_unload_blocks_next_generation_without_phantom_counts','test_residency_config_changes_invalidate_configuration_without_budget_change','test_swap_page_conversion_and_gap_are_measured_not_assumed','test_activation_event_failure_atomic_and_reopen','test_two_activation_transactions_one_winner','test_provenance_matches_actual_body_and_tamper_fails','test_process_exit_mid_transaction_recovers_old_pointer','test_successful_rollback_is_atomic_and_audited','test_dangling_active_pointer_is_not_an_approval','test_evaluation_cannot_trust_passed_boolean','test_queued_deadline_never_starts_background_inference','test_timeout_keeps_session_conflict_until_upstream_ends','test_replay_does_not_resurrect_withdrawn_answer','test_upstream_generation_fails_closed_and_opens_circuit','test_embedding_dimension_and_no_truncation'}


def verify_checks(report,configuration_hash,release):
    """Verify immutable raw local checks; independent release review remains mandatory."""
    import xml.etree.ElementTree as ET
    evidence_hash=report.get('verification_sha256','')
    if len(evidence_hash)!=64 or any(c not in '0123456789abcdef' for c in evidence_hash):
        raise ValueError('Missing verification evidence')
    folder=EVIDENCE_ROOT/evidence_hash
    raw=(folder/'verification.json').read_bytes()
    if digest(raw)!=evidence_hash:raise ValueError('Verification hash mismatch')
    value=json.loads(raw)
    if report.get('source_fingerprints')!=verification_sources() or value.get('configuration_sha256')!=configuration_hash or value.get('release_id')!=release or value.get('source_fingerprints')!=verification_sources() or value.get('commands')!=verification_commands() or type(value.get('test_exit')) is not int or value.get('test_exit')!=0 or type(value.get('scan_exit')) is not int or value.get('scan_exit')!=0 or value.get('scanner_image')!=SCANNER_IMAGE:
        raise ValueError('Checks configuration mismatch')
    xml=(folder/'pytest.xml').read_bytes();scan=(folder/'gitleaks.json').read_bytes()
    if digest(xml)!=value.get('test_xml_sha256') or digest(scan)!=value.get('gitleaks_report_sha256') or json.loads(scan)!=[]:
        raise ValueError('Failed or altered secret scan evidence')
    from scripts.scan_public_sources import SCAN_POLICY
    manifest_raw=(folder/'scan-manifest.json').read_bytes()
    manifest=json.loads(manifest_raw)
    expected=dict(verification_sources())
    expected.update({'app/'+name:hashvalue for name,hashvalue in configuration()['code_sha256'].items()})
    if digest(manifest_raw)!=value.get('scan_manifest_sha256') or value.get('scan_scope')!=SCAN_POLICY or manifest.get('policy')!=SCAN_POLICY or any(manifest.get('files',{}).get(name)!=hashvalue for name,hashvalue in expected.items()):raise ValueError('Public scan source binding mismatch')
    tree=ET.fromstring(xml)
    cases=list(tree.iter('testcase'))
    names={case.attrib.get('name','').split('[')[0] for case in cases}
    if not REQUIRED_TESTS<=names or any(list(case) for case in cases):
        raise ValueError('Mandatory test coverage failed')
    if any(int(s.attrib.get(k,'0')) for s in tree.iter('testsuite') for k in ['errors','failures','skipped']):
        raise ValueError('Test failures present')
    return True


SOURCE_ROOT=Path(__file__).resolve().parents[1]
SCANNER_IMAGE='ghcr.io/gitleaks/gitleaks@sha256:e1b35e12a8c6fa8901f060459cfb6b2fc4c484d3afbe3b029733a3bbfab07055'


def verification_sources():
    paths=sorted((SOURCE_ROOT/'tests').rglob('*.py'))
    paths += [SOURCE_ROOT/name for name in ['scripts/verify_local.py','scripts/scan_public_sources.py','scripts/cloud_setup.py','scripts/cloud_smoke.py','scripts/public_dev10.py','scripts/public_rejection_capture.py','scripts/source_selection_retest.py','scripts/evaluate_candidate.py','scripts/resource_snapshot.py','scripts/resource_trial.py','scripts/pre-commit','.gitleaks.toml','requirements.lock']]
    return {str(p.relative_to(SOURCE_ROOT)):digest(p.read_bytes()) for p in paths}


def verification_commands():
    return {'test':['.venv/bin/python','-m','pytest','-q','--junitxml=eval/checks/pytest-raw.xml'],
            'scan':['docker','run','--pull=never','--rm','-v','SCAN_STAGE:/repo:ro','-v','PROJECT/eval/checks:/reports',SCANNER_IMAGE,'dir','/repo','--config','/repo/.gitleaks.toml','--report-format','json','--report-path','/reports/gitleaks.json','--redact','--no-banner']}


def verify_manual_semantics(approval,report,release,config,evaluation_hash):
    review=approval.get('manual_semantic_review',{})
    if review.get('reviewer')!='Marcus' or review.get('independent') is not True or review.get('release_id')!=release or review.get('configuration_sha256')!=config or review.get('evaluation_sha256')!=evaluation_hash:
        raise ValueError('Independent manual semantic review required')
    if set(review.get('suites',{}))!={'regression','holdout'}:raise ValueError('Both semantic suites required')
    for name,suite in report['suites'].items():
        item=review['suites'][name]
        count=item.get('supported_claims',{})
        n=count.get('numerator');d=count.get('denominator')
        minimum=sum(r.get('status')=='answer' for r in suite['results'])
        if type(n) is not int or type(d) is not int or d<max(1,minimum) or not 0<=n<=d or n/d<.95:
            raise ValueError('Manual semantic threshold failed')
        if item.get('relevance_reviewed') is not True or item.get('qualifications_reviewed') is not True or item.get('dataset_sha256')!=suite['dataset_sha256']:
            raise ValueError('Incomplete manual semantic review')
        ids=item.get('reviewed_result_ids',[])
        if len(ids)!=len(set(ids)) or set(ids)!={r['id'] for r in suite['results']}:
            raise ValueError('Manual review coverage mismatch')
        audits=item.get('case_reviews',[])
        rows={r['id']:r for r in suite['results']}
        if len(audits)!=len(rows) or {a.get('id') for a in audits}!=set(rows):raise ValueError('Manual semantic case audit required')
        from app.ollama import sentence_units
        supported=total=related=0
        critical=[]
        for audit in audits:
            row=rows[audit['id']]
            if audit.get('result_sha256')!=digest(canonical(row)) or not isinstance(audit.get('applicability_rationale'),str) or len(audit['applicability_rationale'])<8:
                raise ValueError('Manual semantic provenance required')
            if type(audit.get('relevant')) is not bool:raise ValueError('Manual relevance judgment required')
            related+=int(audit['relevant'])
            if audit.get('qualifications_complete') is not True or audit.get('critical_instructions_complete') is not True:critical.append(audit['id'])
            claims=audit.get('claims',[])
            expected=sentence_units(row.get('answer') or '') if row['status']=='answer' else []
            if [c.get('text') for c in claims]!=expected:raise ValueError('Manual claim coverage mismatch')
            for claim in claims:
                if not claim.get('source_ids') or not set(claim['source_ids'])<=set(row['citations']) or type(claim.get('supported')) is not bool or type(claim.get('relevant')) is not bool or not isinstance(claim.get('rationale'),str) or len(claim['rationale'])<8:
                    raise ValueError('Manual claim evidence required')
                total+=1;supported+=int(claim['supported'] and claim['relevant'])
        if count!={'numerator':supported,'denominator':total} or not total or supported/total<.95:raise ValueError('Manual semantic threshold failed')
        if item.get('relevance')!={'numerator':related,'denominator':len(rows)} or related/len(rows)<.95:raise ValueError('Manual relevance threshold failed')
        if critical or type(item.get('critical_failure_count')) is not int or item.get('critical_failure_count')!=0 or item.get('critical_failed_case_ids')!=[]:raise ValueError('Manual critical instruction failure')
    return True
