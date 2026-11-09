"""Frozen public lexical candidate stage; maximum ten calls, never activates releases."""
import argparse,asyncio,json,time
from pathlib import Path
from app import service
from app.cloud_config import SetupUnavailable
from app.providers import ProviderAdapter,_request
from app.main import TurnInput
from app.knowledge import ROOT,connect,configuration,digest
from app.lifecycle import canonical,verify
from app.evaluation import verify_checks,verification_sources
from scripts.evaluate_candidate import approved_corpus

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'eval'/'cloud-public-dev10-v1'
FROZEN=BASE/'eval'/'frozen-v4'
IDS=('dev-normal-00','dev-normal-01','dev-public-v2-02','dev-missing_info-00','dev-normal-07','dev-public-v2-04','dev-no_answer-00','dev-injection-00','dev-public-v2-06','dev-public-v2-07')
LIMIT=10


def selected_manifest():
    raw=(FROZEN/'dev.json').read_bytes();frozen=json.loads((FROZEN/'manifest.json').read_bytes())
    if digest(raw)!=frozen['datasets']['dev']['sha256']:raise ValueError('Frozen public dataset changed')
    cases=json.loads(raw);by_id={q['id']:q for q in cases}
    return {'label':'public-dev10-v1-known-not-blind','dataset_sha256':digest(raw),'frozen_manifest_sha256':digest((FROZEN/'manifest.json').read_bytes()),'maximum_provider_calls':LIMIT,'selection_ids':list(IDS),'cases':[by_id[key] for key in IDS],'gold_changed':False,'retrieval':'existing release-isolated lexical; no embedding calls in this bounded product stage'}


def freeze():
    value=selected_manifest();raw=canonical(value);OUT.mkdir(exist_ok=True)
    path=OUT/'manifest.json'
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('Frozen stage manifest conflict')
    else:
        with path.open('xb') as stream:stream.write(raw)
    return value,digest(raw)


class BoundedPublicAdapter(ProviderAdapter):
    def __init__(self):super().__init__();self.calls=[];self.case=None;self.turn=None
    async def generate(self,message,evidence,history):
        if self.generation_scope!='public_eval' or len(self.calls)>=LIMIT:raise SetupUnavailable('stage_request_limit')
        request=_request.get()
        self.calls.append({'index':len(self.calls)+1,'case':self.case,'turn':self.turn,'local_request_id':request[0] if request else None})
        return await super().generate(message,evidence,history)


async def execute_cases(cases,release,manifest_hash,adapter,checkpoint):
    results=[];stop=False
    for q in cases:
        session={'product':None,'history':[],'clarifications':0,'_generation_namespace':digest(canonical(['public-dev10-v1',manifest_hash,q['id']]))}
        row={'id':q['id'],'expected':q['expected'],'turns':[]};adapter.case=q['id']
        turns=list(q.get('prior_turns',[]))+[q]
        for index,turn in enumerate(turns):
            adapter.turn=index;before=len(adapter.calls);start=time.monotonic()
            current=index==len(turns)-1
            payload=TurnInput(client_turn_id=digest(canonical([q['id'],index]))[:32],message=turn['message'],product_id=turn.get('product_id',q['product_id']))
            item={'index':index,'current':current,'input':payload.model_dump()}
            try:
                response=await service.respond(session,payload,'stage-'+q['id']+'-'+str(index),release,offline=True,on_date=q['on_date'])
                item.update(response=response,quality_code=session.get('quality_code'),response_plan=session.get('response_plan'),generation_metrics=session.get('generation_metrics'),product_slot=session.get('product'))
                if current:
                    row['status_matches']=response['status']==q['expected']
                    row['all_frozen_gold_spans_present_proxy']=all(g['span'] in (response['answer'] or '') for g in q['gold']) if q['gold'] else None
                if len(adapter.calls)>before and session.get('quality_code') and not (q['expected']=='handoff' and session['quality_code']=='policy_not_supported'):stop=True
            except Exception as exc:
                item.update(error_class=type(exc).__name__,error_code=getattr(exc,'code','candidate_stage_failure'));stop=True
            if adapter.uncertain or (len(adapter.calls)>before and item.get('response',{}).get('status')=='unavailable'):stop=True
            item.update(provider_admission_attempts=len(adapter.calls)-before,elapsed_seconds=time.monotonic()-start)
            row['turns'].append(item)
            with checkpoint.open('a') as stream:stream.write(json.dumps({'case':q['id'],'turn':item},ensure_ascii=False)+'\n')
            if stop:break
        results.append(row)
        print(json.dumps({'case':q['id'],'status':row['turns'][-1].get('response',{}).get('status'),'new_provider_admissions':len(adapter.calls),'stopped':stop}),flush=True)
        if stop:break
    return results,stop


def safe_accounting(ledger,ids):
    result=[]
    with ledger.db() as db:
        for request in ids:
            row=db.execute('SELECT state,requested_model,returned_model,used_input,used_output,latency FROM requests WHERE id=?',(request,)).fetchone()
            if row:
                result.append(dict(local_request_id=request,state=row[0],requested_model=row[1],returned_model=row[2],input_tokens=row[3],output_tokens=row[4],latency_seconds=row[5],sent_attempts=db.execute("SELECT count(*) FROM events WHERE id=? AND event='sent'",(request,)).fetchone()[0]))
    return result


def budget_preflight(cfg,ledger):
    # Whole maximum fits before starting; each actual request is then durably
    # reserved by the existing transactional guard (no new financial framework).
    cfg.validate('public_eval')
    with ledger.db() as db:
        counts=db.execute("SELECT count(*),coalesce(sum(coalesce(used_input,res_input)),0),coalesce(sum(coalesce(used_output,res_output)),0),coalesce(sum(coalesce(used_cost,res_cost)),0) FROM requests WHERE state!='released'").fetchone()
        no_pending=db.execute("SELECT count(*) FROM requests WHERE state IN ('sent','reserved')").fetchone()[0]==0
    return no_pending and not ledger.status()['unknown_or_sent'] and counts[0]+LIMIT<=cfg.max_requests and counts[1]+LIMIT*3584<=cfg.max_input_tokens and counts[2]+LIMIT*512<=cfg.max_output_tokens and counts[1]+counts[2]+LIMIT*4096<=cfg.max_total_tokens and counts[3]+LIMIT*cfg.price(3584,512)<cfg.cap_units


async def run(candidate,corpus_approval):
    manifest,manifest_hash=freeze()
    if (OUT/'started.json').exists():raise ValueError('Stage already attempted; no replay/resume')
    dbpath=ROOT/'runtime.sqlite3';db=connect(dbpath)
    try:
        release=approved_corpus(db,candidate,corpus_approval)
        if release['configuration_sha256']!=digest(canonical(configuration())):raise ValueError('Stale configuration')
        sources=verification_sources();raw=(BASE/'eval/checks/verification.json').read_bytes();checks=json.loads(raw)
        verify_checks({'verification_sha256':digest(raw),'source_fingerprints':sources},release['configuration_sha256'],candidate)
    finally:db.close()
    adapter=BoundedPublicAdapter();cfg=adapter.preflight('public_eval');ledger=adapter.generation.ledger
    fits=budget_preflight(cfg,ledger)
    if not fits:raise SetupUnavailable('stage_budget_or_pending_block')
    checkpoint=OUT/'checkpoint.jsonl'
    if checkpoint.exists():raise ValueError('Checkpoint already exists')
    with (OUT/'started.json').open('xb') as stream:stream.write(canonical({'candidate':candidate,'configuration_sha256':release['configuration_sha256'],'manifest_sha256':manifest_hash,'maximum_calls':LIMIT,'maximum_stage_reservation_fits':fits,'scope':'public_eval; fixed application stage only'}))
    service.DB=dbpath;service.adapter=adapter
    results,stopped=await execute_cases(manifest['cases'],candidate,manifest_hash,adapter,checkpoint)
    if sources!=verification_sources():raise ValueError('Stage source drift')
    account=safe_accounting(ledger,[c['local_request_id'] for c in adapter.calls if c['local_request_id']])
    report={'scope':'public-dev10-v1-known-not-blind; candidate service only, no active release, no HTTP/UI/replay acceptance','candidate':candidate,'configuration_sha256':release['configuration_sha256'],'verification_sha256':digest(raw),'manifest_sha256':manifest_hash,'corpus_approval_sha256':digest(Path(corpus_approval).read_bytes()),'source_fingerprints':sources,'results':results,'generation_admission_attempts':len(adapter.calls),'maximum_calls':LIMIT,'provider_call_mapping':adapter.calls,'accounting':account,'actual_sent_attempts':sum(r['sent_attempts'] for r in account),'confirmed_input_tokens':sum(r['input_tokens'] or 0 for r in account),'confirmed_output_tokens':sum(r['output_tokens'] or 0 for r in account),'stopped':stopped,'upstream_pending':adapter.pending,'upstream_uncertain':adapter.uncertain,'paid_block':ledger.status()['unknown_or_sent'],'maximum_stage_preflight_fits':fits,'remaining_budget_positive':None,'manual_semantic_review_required':True,'gold_or_thresholds_changed':False,'Ollama_or_embedding_calls':0,'replay_scope':'not exercised on real serving endpoint; existing mock request-boundary tests only'}
    with ledger.db() as db:report['remaining_budget_positive']=db.execute("SELECT coalesce(sum(coalesce(used_cost,res_cost)),0) FROM requests WHERE state!='released'").fetchone()[0]<cfg.cap_units
    (OUT/'results.json').write_bytes(canonical(report))
    print(json.dumps({key:report[key] for key in ['generation_admission_attempts','actual_sent_attempts','confirmed_input_tokens','confirmed_output_tokens','stopped','upstream_pending','upstream_uncertain','paid_block']}))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze-only',action='store_true');parser.add_argument('--candidate');parser.add_argument('--corpus-approval');args=parser.parse_args()
    try:
        if args.freeze_only:
            manifest,sha=freeze();print(json.dumps({'manifest_sha256':sha,'cases':len(manifest['cases']),'maximum_calls':LIMIT,'network_calls':0}))
        else:
            if not args.candidate or not args.corpus_approval:raise ValueError('Candidate and corpus approval required')
            report=asyncio.run(run(args.candidate,args.corpus_approval));raise SystemExit(3 if report['stopped'] else 0)
    except Exception:
        print(json.dumps({'status':'stage_stopped_no_automatic_retry'}));raise SystemExit(2)
