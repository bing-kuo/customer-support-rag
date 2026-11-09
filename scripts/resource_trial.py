"""Bounded public dev residency comparison; never a release or HTTP benchmark."""
import argparse
import asyncio
import json
import threading
import time
from pathlib import Path
import numpy as np
from app import service,planning
from app.knowledge import ROOT,configuration,connect,digest,eligible,search_candidate
from app.lifecycle import canonical
from app.evaluation import FROZEN,normalize_dataset,score,verification_sources,verify_checks
from app.main import TurnInput
from app.ollama import Adapter,MODELS,RESIDENCY
from scripts.evaluate_candidate import approved_corpus
from scripts.resource_snapshot import snapshot,REVISION

INTERVAL=2
IDLE_SECONDS=20


def frozen_subset():
    raw=(FROZEN/'dev.json').read_bytes()
    manifest=json.loads((FROZEN/'manifest.json').read_bytes())
    if digest(raw)!=manifest['datasets']['dev']['sha256']:raise ValueError('Frozen dev mismatch')
    cases,_=normalize_dataset(raw)
    selected=[q for q in cases if q['expected']=='answer' and not q.get('prior_turns')][:21]
    if len(selected)!=21:raise ValueError('Cold plus twenty warm cases required')
    return {'selection':'first21 expected=answer without prior turns in frozen public dev order; one cold plus20 warm; identical both arms','dev_raw_sha256':digest(raw),'frozen_manifest_sha256':digest((FROZEN/'manifest.json').read_bytes()),'cases':selected}


class Samples:
    def __init__(self,folder):
        self.path=folder/'samples.jsonl';self.previous=None;self.phase='startup';self.rows=[];self.errors=[]
        self.lock=threading.Lock();self.stop=threading.Event()
    def capture(self):
        with self.lock:
            row=snapshot(self.phase,self.previous,INTERVAL)
            self.previous=row;self.rows.append(row)
            with self.path.open('a') as out:out.write(json.dumps(row)+'\n')
            return row
    def mark(self,phase):
        self.phase=phase;return self.capture()
    def monitor(self):
        while not self.stop.wait(INTERVAL):
            try:self.capture()
            except Exception as exc:self.errors.append({'phase':self.phase,'error':type(exc).__name__,'monotonic':time.monotonic()})


def timing(values):
    return {'count':len(values),'p50_seconds':float(np.percentile(values,50)),'p95_seconds':float(np.percentile(values,95)),'max_seconds':max(values)}


async def arm(name,folder,subset,release,manifest,exclusive):
    folder.mkdir();samples=Samples(folder);startup=samples.mark('startup')
    if startup['models']:raise ValueError('Unexpected preexisting residency; no model stopped')
    monitor=threading.Thread(target=samples.monitor,daemon=True);monitor.start()
    a=Adapter();service.adapter=a;service.DB=ROOT/'runtime.sqlite3'
    results=[];dense=[];phases={};failure=None
    try:
        await a.begin_owned_batch(exclusive_slot=exclusive)
        samples.mark('no-inference-baseline');await asyncio.sleep(IDLE_SECONDS)
        cases=subset['cases'];contexts=[service.retrieval_context(service.DB,release,q) for q in cases]
        samples.mark('embedding-only');started=time.monotonic()
        query_vectors=await a.embed(['Instruct: Retrieve supporting customer support evidence\nQuery: '+c[0] for c in contexts])
        pools={}
        for q,c in zip(cases,contexts):
            key=planning.pool_key(c[0],c[2],q['on_date'],q['locale'])
            if key not in pools:
                pool=planning.pool(service.DB,release,c[0],c[2],q['on_date'],q['locale'])
                pools[key]=(pool,await a.embed([r[3] for r in pool]) if pool else np.empty((0,1024)))
        phases['embedding_seconds']=time.monotonic()-started
        samples.mark('embedding-phase-complete');started=time.monotonic()
        if name=='phased':await a.unload_owned('qwen3-embedding:0.6b')
        phases['embedding_phase_unload_seconds']=time.monotonic()-started
        samples.mark('generation-cold')
        for index,(q,c) in enumerate(zip(cases,contexts)):
            if index==1:samples.mark('generation-warm20')
            rows=planning.search(service.DB,release,c[0],c[2],q['on_date'],q['locale'])
            pool,vectors=pools[planning.pool_key(c[0],c[2],q['on_date'],q['locale'])]
            scores=np.sum(vectors*query_vectors[index],axis=1)
            if not np.isfinite(scores).all():raise ValueError('Dense scores invalid')
            dense_ids=[pool[int(i)][0] for i in np.argsort(-scores)[:10]]
            session={'product':None,'history':[],'clarifications':0}
            payload=TurnInput(client_turn_id=digest(canonical([name,q['id']]))[:32],message=q['message'],product_id=q['product_id'])
            started=time.monotonic()
            row={'id':q['id'],'retrieved':[r[0] for r in rows],'citations':[],'status':'unavailable','answer':None}
            try:
                response=await service.respond(session,payload,'resource-'+str(index),release,offline=True,on_date=q['on_date'])
                row.update(status=response['status'],answer=response['answer'],citations=[v['source_id'][64:] for v in response['citations']])
            except Exception as exc:
                row.update(error=type(exc).__name__,error_code=getattr(exc,'code','safe_unspecified'))
                raise
            finally:
                row.update(seconds=time.monotonic()-started,generation_metrics=session.get('generation_metrics'),quality_code=session.get('quality_code'),request_sha256=digest(canonical({'release':release,'configuration':manifest['configuration_sha256'],'case':q})))
                results.append(row)
                with (folder/'checkpoint.jsonl').open('a') as out:out.write(json.dumps(row,ensure_ascii=False)+'\n')
            dense.append(dict(row,retrieved=dense_ids,citations=[],answer=None,status='unavailable'))
            print(json.dumps({'arm':name,'round':index+1,'status':row['status']}),flush=True)
        phases['cold_seconds']=results[0]['seconds'];phases['warm20']=timing([r['seconds'] for r in results[1:]])
        samples.mark('batch-unload');started=time.monotonic();await a.finish_owned_batch()
        phases['final_unload_seconds']=time.monotonic()-started
        samples.mark('post-unload-recovery');await asyncio.sleep(IDLE_SECONDS);samples.mark('recovery-end')
    except Exception as exc:
        failure={'error':type(exc).__name__,'code':getattr(exc,'code','safe_unspecified')}
        if a.exclusive_batch and a.idle_for_maintenance():
            try:await a.finish_owned_batch()
            except Exception as cleanup:failure['cleanup_error']=getattr(cleanup,'code',type(cleanup).__name__)
        samples.mark('failed-end')
    finally:
        samples.stop.set();monitor.join(timeout=3)
    db=connect(service.DB)
    metrics=score(db,release,subset['cases'],results) if len(results)==21 else None
    dense_metrics=score(db,release,subset['cases'],dense,retrieval_mode='dense') if len(dense)==21 else None
    db.close()
    first,last=samples.rows[0],samples.rows[-1]
    report={'scope':'offline candidate service cold+20warm public dev subset; excludes HTTP/UI; no blind or full-suite acceptance','arm':name,'diagnostic_override':{'embedding_phase_unload':name=='phased'},'arm_configuration_sha256':digest(canonical({'candidate_configuration':manifest['configuration_sha256'],'embedding_phase_unload':name=='phased','residency':RESIDENCY})),'release_id':release,'configuration_sha256':manifest['configuration_sha256'],'subset_sha256':digest(canonical(subset)),'sampling_revision':REVISION,'sample_interval_target_seconds':INTERVAL,'idle_and_recovery_seconds_each':IDLE_SECONDS,'results':results,'phases':phases,'mechanical_subset':metrics,'dense_subset':dense_metrics,'error':failure,'model_digests':MODELS,'residency':dict(RESIDENCY),'lifecycle_events':a.lifecycle_events,'lifecycle_failures':dict(a.lifecycle_failures),'quality_counts':dict(a.quality_counts),'dependency_counts':dict(a.dependency_counts),'upstream_pending':a.pending,'embedding_pending':a.embedding_pending,'upstream_uncertain':a.uncertain,'residency_uncertain':a.residency_uncertain,'sample_errors':samples.errors,'sample_count':len(samples.rows),'late_samples':sum(r.get('late_sample',False) for r in samples.rows),'baseline':first,'after':last,'peak_rss_kib':max(r['ollama_aggregate_rss_kib'] for r in samples.rows),'pressure_levels':dict(__import__('collections').Counter(r['kernel_pressure_level'] for r in samples.rows)),'swapout_delta_pages':last['vm_counters']['Swapouts']-first['vm_counters']['Swapouts'],'swapout_delta_mib':(last['vm_counters']['Swapouts']-first['vm_counters']['Swapouts'])*first['page_size_bytes']/1048576,'samples_sha256':digest(samples.path.read_bytes())}
    (folder/'report.json').write_bytes(canonical(report))
    if failure or a.pending or a.uncertain or a.residency_uncertain:raise ValueError('Bounded trial failed; do not run another arm')
    return report


async def run(args):
    if not args.exclusive_model_slot:raise ValueError('Exclusive model slot required')
    folder=Path(args.output_dir)
    if folder.exists():raise ValueError('Never overwrite trial evidence')
    folder.mkdir(parents=True)
    subset=frozen_subset();(folder/'subset.json').write_bytes(canonical(subset))
    db=connect(ROOT/'runtime.sqlite3')
    manifest=approved_corpus(db,args.candidate,args.corpus_approval);db.close()
    if manifest['configuration_sha256']!=digest(canonical(configuration())):raise ValueError('Stale candidate')
    sources=verification_sources();checks=digest(Path('eval/checks/verification.json').read_bytes())
    verify_checks({'verification_sha256':checks,'source_fingerprints':sources},manifest['configuration_sha256'],args.candidate)
    provenance={'candidate':args.candidate,'configuration_sha256':manifest['configuration_sha256'],'checks_sha256':checks,'sources':sources,'subset_sha256':digest(canonical(subset)),'corpus_approval_sha256':digest(Path(args.corpus_approval).read_bytes()),'one_factor':'both arms finite60s keepalive; only unload embedding before generation differs; baseline embedding may expire naturally during generation','ordering_limit':'retained then phased; no OS cache flush, no swap reset; system workload uncontrolled; model-empty startup confirmed each arm'}
    (folder/'provenance.json').write_bytes(canonical(provenance))
    for name in ['retained','phased']:
        report=await arm(name,folder/name,subset,args.candidate,manifest,args.exclusive_model_slot)
        if sources!=verification_sources() or manifest['configuration_sha256']!=digest(canonical(configuration())):raise ValueError('Sources changed during trial')
        print(json.dumps({'arm':name,'phases':report['phases'],'swapout_delta_mib':report['swapout_delta_mib'],'pressure':report['pressure_levels']}),flush=True)

if __name__=='__main__':
    raise SystemExit('Retired14B resource trial; cloud stage requires separate authorised protocol')
