"""Offline approved-corpus evaluation; never activates or exposes a candidate."""
import argparse
import asyncio
import json
import time
from pathlib import Path
import numpy as np
from app import service,planning
from app.knowledge import ROOT,build,configuration,connect,digest,eligible,search_candidate
from app.lifecycle import canonical,corpus_identity,record,verify,schema
from app.evaluation import FROZEN,score,normalize_dataset,verification_sources,verify_checks
from app.main import TurnInput
from app.providers import adapter
from app.ollama import DependencyError,CapacityError,BudgetError,TurnTimeout


VERIFICATION_FILE=Path('eval/checks/verification.json')

def approved_corpus(db,release,approval_path):
    approval,_=record(approval_path);manifest=verify(db,release)
    docs=manifest['documents']
    if approval.get('manifest_sha256')!=manifest['source_manifest_sha256'] or approval.get('corpus_sha256')!=corpus_identity(docs) or approval.get('document_hashes')!={d['doc_id']+':'+d['revision']:d['content_sha256'] for d in docs}:
        raise ValueError('Candidate corpus is not independently approved')
    return manifest


def dataset_preflight(path,expected_hash):
    """Read-only metadata output for the dataset owner; never prints case content."""
    from collections import Counter
    raw=Path(path).read_bytes()
    if digest(raw)!=expected_hash:raise ValueError('Frozen raw dataset hash mismatch')
    normalized,proof=normalize_dataset(raw)
    counts=Counter(q['category'] for q in normalized)
    if counts!=dict.fromkeys(['normal','wrong_version','missing_info','no_answer','injection'],6):
        raise ValueError('Holdout categories mismatch')
    return dict(proof,case_count=len(normalized),category_counts=dict(counts))


async def run(args):
    import datetime
    if not getattr(args,'corpus_approval',None):raise ValueError('Corpus approval required')
    run_started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    if args.split=='holdout':
        if not getattr(args,'dataset_file',None) or not getattr(args,'output',None) or not getattr(args,'expected_dataset_sha256',None):
            raise ValueError('Private holdout requires explicit dataset, hash and private output')
        dataset_raw=Path(args.dataset_file).read_bytes()
        if digest(dataset_raw)!=args.expected_dataset_sha256:
            raise ValueError('Independent frozen holdout hash mismatch')
        dataset,dataset_adapter=normalize_dataset(dataset_raw)
        from collections import Counter
        if Counter(q['category'] for q in dataset)!=dict.fromkeys(['normal','wrong_version','missing_info','no_answer','injection'],6):
            raise ValueError('Holdout categories mismatch')
        output_path=Path(args.output)
        sealed=Path(args.dataset_file).resolve()
        if output_path.resolve()==sealed or output_path.with_suffix('.checkpoint.jsonl').resolve()==sealed or output_path.exists():
            raise ValueError('Private output must not overwrite sealed data or evidence')
    else:
        dataset_raw=(FROZEN/(args.split+'.json')).read_bytes();dataset,dataset_adapter=normalize_dataset(dataset_raw)
        frozen=json.loads((FROZEN/'manifest.json').read_bytes())
        if digest(dataset_raw)!=frozen['datasets'][args.split]['sha256']:
            raise ValueError('Frozen dataset modified')
        output_path=Path('eval/'+args.split+'-actual.json')
    for q in dataset:
        TurnInput(client_turn_id=digest(canonical([q['id'],'current']))[:32],message=q['message'],product_id=q['product_id'])
        if len(q.get('prior_turns',[]))>19:raise ValueError('Prior turn limit exceeded')
        for i,prior in enumerate(q.get('prior_turns',[])):
            if not isinstance(prior,dict):raise ValueError('Unsupported prior turn shape')
            TurnInput(client_turn_id=digest(canonical([q['id'],i]))[:32],message=prior.get('message'),product_id=prior.get('product_id',q['product_id']))
    frozen=json.loads((FROZEN/'manifest.json').read_bytes())
    path=ROOT/'runtime.sqlite3';release=getattr(args,'candidate',None) or build(path);db=connect(path)
    manifest=approved_corpus(db,release,args.corpus_approval)
    schema(db)
    if manifest['source_manifest_sha256']!=frozen['corpus_manifest_sha256']:
        raise ValueError('Frozen corpus changed')
    if manifest['configuration_sha256']!=digest(canonical(configuration())):
        raise ValueError('Candidate configuration is stale')
    run_sources=verification_sources()
    verify_checks({'verification_sha256':digest(VERIFICATION_FILE.read_bytes()),'source_fingerprints':run_sources},manifest['configuration_sha256'],release)
    service.DB=path
    # No activation/event is written; offline=True only resolves this candidate.
    results=[];dense_results=[];vectors={};generation=[]
    checkpoint=output_path.with_suffix('.checkpoint.jsonl') if args.split=='holdout' else Path('eval/'+args.split+'-checkpoint.jsonl')
    if checkpoint.exists():raise ValueError('Refusing to overwrite checkpoint; archive prior run explicitly')
    adapter.preflight('public_eval' if args.split!='holdout' else 'holdout')
    await adapter.begin_owned_batch(exclusive_slot=getattr(args,'exclusive_model_slot',False))
    checkpoint.write_text('')
    try:
        phase_started=time.monotonic()
        contexts=[service.retrieval_context(path,release,q) for q in dataset]
        query_vectors=await adapter.embed(['Instruct: Retrieve supporting customer support evidence\nQuery: '+context[0] for context in contexts])
        for q,context in zip(dataset,contexts):
            key=planning.pool_key(context[0],context[2],q['on_date'],q['locale'])
            if key not in vectors:
                pool=planning.pool(path,release,context[0],context[2],q['on_date'],q['locale'])
                vectors[key]=(pool,await adapter.embed([r[3] for r in pool]) if pool else np.empty((0,1024)))
        embedding_seconds=time.monotonic()-phase_started
        phase_started=time.monotonic()
        await adapter.unload_owned('qwen3-embedding:0.6b')
        phase_unload_seconds=time.monotonic()-phase_started
        for index,q in enumerate(dataset):
            start=time.monotonic()
            context=contexts[index]
            rows=planning.search(path,release,context[0],context[2],q['on_date'],q['locale'])
            retrieval_seconds=time.monotonic()-start
            pool=planning.pool(path,release,context[0],context[2],q['on_date'],q['locale'])
            key=planning.pool_key(context[0],context[2],q['on_date'],q['locale'])
            if key not in vectors:raise ValueError('Embedding phase is incomplete')
            pool,matrix=vectors[key]
            query=query_vectors[index]
            scores=np.sum(matrix*query,axis=1)
            if not np.isfinite(scores).all():raise ValueError('Invalid dense cosine')
            dense_ids=[pool[int(i)][0] for i in np.argsort(-scores)[:10]]
            row={'id':q['id'],'retrieved':[r[0] for r in rows],'citations':[],'status':'unavailable','answer':None,'retrieval_seconds':retrieval_seconds}
            session={'product':None,'history':[],'clarifications':0}
            for i,prior in enumerate(q.get('prior_turns',[])):
                previous=TurnInput(client_turn_id=digest(canonical([q['id'],i]))[:32],message=prior['message'],product_id=prior.get('product_id',q['product_id']))
                await service.respond(session,previous,'offline-prior',release,offline=True,on_date=q['on_date'])
            payload=TurnInput(client_turn_id=digest(canonical([q['id'],'current']))[:32],message=q['message'],product_id=q['product_id'])
            started=time.monotonic()
            try:
                response=await service.respond(session,payload,'offline-'+q['id'],release,offline=True,on_date=q['on_date'])
                row.update(status=response['status'],answer=response['answer'],citations=[c['source_id'][64:] for c in response['citations']])
            except (DependencyError,CapacityError,BudgetError,TurnTimeout,asyncio.TimeoutError) as exc:
                row['error']=type(exc).__name__
                row['error_code']=getattr(exc,'code','dependency_failure')
            row['response_plan']=session.get('response_plan')
            row['quality_code']=session.get('quality_code')
            row['generation_metrics']=session.get('generation_metrics')
            row['generation_and_policy_seconds']=time.monotonic()-started
            # Request binding prevents accidental mixing of runs/configs/datasets.
            row['request_sha256']=digest(canonical({'release':release,'configuration':manifest['configuration_sha256'],'case':q}))
            with checkpoint.open('a') as output:
                output.write(json.dumps({'release_id':release,'case':q['id'],'result':row,'dense_retrieved':dense_ids},ensure_ascii=False)+'\n')
            results.append(row)
            dense_results.append(dict(row,retrieved=dense_ids,citations=[],answer=None,status='unavailable'))
            if args.split!='holdout':
                print(json.dumps({'case':q['id'],'status':row['status'],'error':row.get('error')}),flush=True)
        await adapter.finish_owned_batch()
        if run_sources!=verification_sources():raise ValueError('Evaluation source changed during run')
        metrics=score(db,release,dataset,results)
        dense_score=score(db,release,dataset,dense_results,retrieval_mode='dense')
        report={'model_phases':{'embedding_seconds':embedding_seconds,'embedding_unload_seconds':phase_unload_seconds,'lifecycle_events':adapter.lifecycle_events,'lifecycle_failures':dict(adapter.lifecycle_failures)},'schema_version':1,'public_suite_revision':frozen.get('suite_label','public-v1'),'source_fingerprints':run_sources,'verification_sha256':digest(VERIFICATION_FILE.read_bytes()),'started_at':run_started,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'embedding_tag_snapshot':getattr(adapter,'last_embedding_tags',None),'model_digests':{'qwen3-embedding:0.6b':__import__('app.ollama',fromlist=['MODELS']).MODELS['qwen3-embedding:0.6b']},'generation_provider':__import__('app.cloud_config',fromlist=['POLICY']).POLICY,'scope':'offline candidate; mechanical exact-span checks only; independent manual semantic QA required','release_id':release,'manifest_sha256':release,'configuration_sha256':manifest['configuration_sha256'],'corpus_approval_sha256':digest(Path(args.corpus_approval).read_bytes()),'suites':{args.split:{'dataset':dataset,'dataset_sha256':digest(dataset_raw),'dataset_raw':dataset_raw.decode('utf-8'),'dataset_adapter':dataset_adapter,'results':results}},'computed':metrics,'dense_retrieval':{k:dense_score['metrics'][k] for k in ['recall_at_5','mrr_at_5']},'quality_counts':dict(adapter.quality_counts),'dependency_counts':dict(adapter.dependency_counts),'upstream_pending':adapter.pending,'upstream_uncertain':adapter.uncertain,'selection':'lexical; dense must improve beyond dev baseline before adoption; thresholds unchanged'}
        if args.split=='holdout':
            report['suites']['holdout'].update(dataset_raw=dataset_raw.decode('utf-8'),reviewer='Marcus')
            report['holdout_sha256']=digest(dataset_raw)
        output_path.write_bytes(canonical(report))
        print(json.dumps({'mechanical_metrics':metrics,'dense':report['dense_retrieval']}),flush=True)
        return metrics['passed']
    finally:
        if adapter.exclusive_batch and adapter.idle_for_maintenance():await adapter.finish_owned_batch()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','regression','holdout'],required=True);p.add_argument('--corpus-approval');p.add_argument('--candidate');p.add_argument('--dataset-file');p.add_argument('--expected-dataset-sha256');p.add_argument('--output');p.add_argument('--validate-dataset-only',action='store_true');p.add_argument('--exclusive-model-slot',action='store_true',help='Assert an operational exclusive slot; batch additionally requires empty /api/ps');args=p.parse_args()
    try:
        if args.validate_dataset_only:
            if args.split!='holdout' or not args.dataset_file or not args.expected_dataset_sha256:raise ValueError('Dataset preflight arguments required')
            print(json.dumps(dataset_preflight(args.dataset_file,args.expected_dataset_sha256)))
            raise SystemExit(0)
        passed=asyncio.run(run(args))
    except Exception:
        print(json.dumps({'error':'evaluation_input_or_gate_failed'}))
        raise SystemExit(2)
    raise SystemExit(0 if passed else 3)
