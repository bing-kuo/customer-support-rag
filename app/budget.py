"""Durable conservative reservations; no credentials, messages or results stored."""
import sqlite3,time,uuid,re,json,math
from pathlib import Path
from contextlib import contextmanager
from app.cloud_config import SetupUnavailable,MODEL,fingerprint

STAGES={'sent','http_received','response_identity_recorded','usage_validated','accounting_settled','application_completed','application_rejected','accounting_or_dependency_rejected','dependency_rejected','logical_timeout','caller_cancelled','unknown'}
CODES={'none','unclassified_failure','usage_missing','usage_shape_invalid','usage_totals_invalid','known_usage_extension_invalid','unknown_billing_component','unknown_usage','unverified_service_tier','unverified_returned_model','unknown_response_terminal_state','usage_exceeds_reservation','ledger_schema_or_storage_unavailable','settlement_storage_failed','diagnostic_storage_failed','provider_transport_error','provider_service_unavailable','provider_authentication_or_access_unavailable','provider_model_or_endpoint_unavailable','provider_capacity','unbounded_response','logical_timeout','caller_cancelled','invalid_model_schema','unknown_citation','non_verbatim_sentence','missing_qualifier','approval_changed_before_send','output_refusal'}
USAGE_KEYS={'input_tokens','output_tokens','total_tokens','cached_tokens','reasoning_tokens','cache_write_tokens','ttl_30m_tokens','ttl_12h_tokens'}


def safe_diagnostic(stage,code,metadata):
    value={'stage':stage if isinstance(stage,str) and stage in STAGES else 'unknown','code':code if isinstance(code,str) and code in CODES else 'unclassified_failure'}
    for key in ('usage_present','terminal_verified','settled','unknown_usage_component'):
        if type(metadata.get(key)) is bool:value[key]=metadata[key]
    status=metadata.get('provider_status')
    if 'provider_status' in metadata:value['provider_status']=status if isinstance(status,str) and status in {'completed','incomplete','failed','cancelled','queued','in_progress'} else 'unknown'
    tier=metadata.get('service_tier')
    if 'service_tier' in metadata:value['service_tier']=tier if isinstance(tier,str) and tier in {'default','auto','flex','fast','priority','omitted'} else 'unknown'
    aliases=metadata.get('candidate_aliases')
    if isinstance(aliases,list):value['candidate_aliases']=[a for a in aliases if isinstance(a,str) and re.fullmatch('e[0-3]',a)]
    if metadata.get('render_mode')=='source_selection_server_render':value['render_mode']='source_selection_server_render'
    fields=metadata.get('usage_fields')
    allowed={'input_tokens','output_tokens','total_tokens','input_tokens_details','output_tokens_details','cached_tokens','reasoning_tokens','cache_write_tokens','cache_write_tokens_details','ttl_30m_tokens','ttl_12h_tokens'}
    if isinstance(fields,list):value['usage_fields']=sorted({k for k in fields if isinstance(k,str) and k in allowed})
    number=metadata.get('http_status')
    if type(number) is int and 100<=number<=599:value['http_status']=number
    elapsed=metadata.get('elapsed_seconds')
    if type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<=86400:value['elapsed_seconds']=elapsed
    usage=metadata.get('usage')
    if isinstance(usage,dict):value['usage']={k:v for k,v in usage.items() if k in USAGE_KEYS and (v is None or type(v) is int and 0<=v<=2**31)}
    return value

class BudgetUnavailable(SetupUnavailable):
    def __init__(self,code='budget_unavailable'):super().__init__(code)

class Ledger:
    # A disposition never changes unknown usage into provider-settled accounting.
    debit_match="EXISTS (SELECT 1 FROM conservative_debits c WHERE c.id=requests.id AND c.budget=requests.budget AND c.body_hash=requests.body_hash AND c.pricing_hash=requests.pricing_hash AND c.res_cost=requests.res_cost AND c.res_input=requests.res_input AND c.res_output=requests.res_output)"
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.owner=uuid.uuid4().hex
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS policy_guard(singleton INTEGER PRIMARY KEY CHECK(singleton=1),cap_fingerprint TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS budgets(id TEXT PRIMARY KEY,binding TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY,budget TEXT,body_hash TEXT,owner TEXT,scope TEXT,state TEXT,res_input INTEGER,res_output INTEGER,res_cost INTEGER,used_input INTEGER,used_output INTEGER,used_cost INTEGER,pricing_hash TEXT,requested_model TEXT,returned_model TEXT,provider_request_id TEXT,provider_response_id TEXT,cached_tokens INTEGER,reasoning_tokens INTEGER,created REAL,latency REAL)')
            db.execute('CREATE TABLE IF NOT EXISTS conservative_debits(id TEXT PRIMARY KEY,budget TEXT,body_hash TEXT,pricing_hash TEXT,res_cost INTEGER,res_input INTEGER,res_output INTEGER,decision_hash TEXT,review_hash TEXT,ceiling_fingerprint TEXT,at REAL,decision_metadata TEXT)')
            if 'decision_metadata' not in {r[1] for r in db.execute('PRAGMA table_info(conservative_debits)')}:db.execute('ALTER TABLE conservative_debits ADD COLUMN decision_metadata TEXT')
            db.execute('CREATE TABLE IF NOT EXISTS diagnostics(id TEXT,metadata TEXT NOT NULL,at REAL)')
            db.execute('CREATE TABLE IF NOT EXISTS events(id TEXT,event TEXT,at REAL)')
            required={'id','budget','body_hash','owner','scope','state','res_input','res_output','res_cost','used_input','used_output','used_cost','pricing_hash','requested_model','returned_model','provider_request_id','provider_response_id','cached_tokens','reasoning_tokens','created','latency'}
            if not required<={r[1] for r in db.execute('PRAGMA table_info(requests)')}:raise BudgetUnavailable('ledger_schema_incompatible')
    @contextmanager
    def db(self):
        db=sqlite3.connect(self.path,timeout=2,isolation_level=None)
        try:
            db.execute('PRAGMA synchronous=FULL');db.execute('BEGIN IMMEDIATE');yield db;db.commit()
        except sqlite3.DatabaseError:
            db.rollback();raise BudgetUnavailable('ledger_schema_or_storage_unavailable') from None
        except BaseException:db.rollback();raise
        finally:db.close()
    def event(self,db,request_id,event):db.execute('INSERT INTO events VALUES(?,?,?)',(request_id,event,time.time()))
    def reserve(self,cfg,request_id,body_hash,input_tokens,output_tokens,scope):
        cfg.validate(scope)
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',request_id) or not re.fullmatch(r'[a-f0-9]{64}',body_hash) or cfg.key in request_id or cfg.key in body_hash:raise BudgetUnavailable()
        cost=cfg.price(input_tokens,output_tokens)
        if type(input_tokens) is not int or type(output_tokens) is not int or input_tokens<1 or output_tokens!=512 or input_tokens+output_tokens>4096 or cost>2**60:raise BudgetUnavailable()
        with self.db() as db:
            guard=db.execute('SELECT cap_fingerprint FROM policy_guard WHERE singleton=1').fetchone()
            if guard and fingerprint({'ceiling':cfg.cap_units})!=guard[0]:raise BudgetUnavailable('cap_change_requires_authorization')
            if not guard:db.execute('INSERT INTO policy_guard VALUES(1,?)',(fingerprint({'ceiling':cfg.cap_units}),))
            old=db.execute('SELECT body_hash,state FROM requests WHERE id=?',(request_id,)).fetchone()
            if old and (old[0]!=body_hash or old[1]!='released'):raise BudgetUnavailable('duplicate_or_conflicting_request')
            if db.execute('SELECT 1 FROM conservative_debits WHERE body_hash=?',(body_hash,)).fetchone():raise BudgetUnavailable('permanent_original_body_replay')
            if db.execute("SELECT 1 FROM requests WHERE (state='unknown' AND NOT "+self.debit_match+") OR (state IN ('sent','reserved') AND owner!=?)",(self.owner,)).fetchone():raise BudgetUnavailable('unreconciled_or_foreign_reservation')
            if db.execute('SELECT 1 FROM budgets WHERE id!=?',(cfg.budget_id,)).fetchone():raise BudgetUnavailable('budget_identity_change_requires_reconciliation')
            binding=db.execute('SELECT binding FROM budgets WHERE id=?',(cfg.budget_id,)).fetchone()
            if binding and binding[0]!=cfg.binding():raise BudgetUnavailable('budget_configuration_changed')
            if not binding:db.execute('INSERT INTO budgets VALUES(?,?)',(cfg.budget_id,cfg.binding()))
            used=db.execute("SELECT count(*),coalesce(sum(coalesce(used_input,res_input)),0),coalesce(sum(coalesce(used_output,res_output)),0),coalesce(sum(coalesce(used_cost,res_cost)),0) FROM requests WHERE budget=? AND state!='released'",(cfg.budget_id,)).fetchone()
            if used[0]+1>cfg.max_requests or used[1]+input_tokens>cfg.max_input_tokens or used[2]+output_tokens>cfg.max_output_tokens or used[1]+used[2]+input_tokens+output_tokens>cfg.max_total_tokens or used[3]+cost>=cfg.cap_units:raise BudgetUnavailable('budget_exhausted')
            if db.execute("SELECT count(*) FROM requests WHERE state IN ('reserved','sent')").fetchone()[0]>=3:raise BudgetUnavailable('reservation_capacity')
            if old:db.execute('DELETE FROM requests WHERE id=?',(request_id,))
            db.execute('INSERT INTO requests(id,budget,body_hash,owner,scope,state,res_input,res_output,res_cost,pricing_hash,requested_model,created) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(request_id,cfg.budget_id,body_hash,self.owner,scope,'reserved',input_tokens,output_tokens,cost,cfg.pricing_hash(),MODEL,time.time()))
            self.event(db,request_id,'reserved')
    def sent(self,request_id):
        with self.db() as db:
            if db.execute("SELECT 1 FROM requests WHERE state='sent' OR (state='unknown' AND NOT "+self.debit_match+")").fetchone():raise BudgetUnavailable('global_upstream_pending')
            if db.execute("UPDATE requests SET state='sent' WHERE id=? AND owner=? AND state='reserved'",(request_id,self.owner)).rowcount!=1:raise BudgetUnavailable()
            self.event(db,request_id,'sent')
    def release(self,request_id):
        with self.db() as db:
            if db.execute("UPDATE requests SET state='released' WHERE id=? AND owner=? AND state='reserved'",(request_id,self.owner)).rowcount:self.event(db,request_id,'released_before_send')
    def unknown(self,request_id):
        with self.db() as db:
            if db.execute("UPDATE requests SET state='unknown' WHERE id=? AND owner=? AND state='sent'",(request_id,self.owner)).rowcount:self.event(db,request_id,'unknown_cost_retained')
            return bool(db.execute("SELECT 1 FROM requests WHERE id=? AND state IN ('sent','unknown')",(request_id,)).fetchone())
    def provider_identity(self,request_id,provider_request_id=None,provider_response_id=None,returned_model=None):
        with self.db() as db:
            db.execute('UPDATE requests SET provider_request_id=coalesce(?,provider_request_id),provider_response_id=coalesce(?,provider_response_id),returned_model=coalesce(?,returned_model) WHERE id=? AND owner=?',(provider_request_id,provider_response_id,returned_model,request_id,self.owner))
    def settle(self,cfg,request_id,usage,request_identity,response_identity,latency):
        i,o,cached,reasoning=usage
        with self.db() as db:
            if db.execute('SELECT 1 FROM conservative_debits WHERE id=?',(request_id,)).fetchone():raise BudgetUnavailable('permanent_conservative_debit_no_settlement')
            row=db.execute('SELECT res_input,res_output,pricing_hash,state FROM requests WHERE id=? AND owner=?',(request_id,self.owner)).fetchone()
            if not row or row[3] not in ('sent','unknown') or row[2]!=cfg.pricing_hash() or i>row[0] or o>row[1]:raise BudgetUnavailable('usage_exceeds_reservation')
            db.execute("UPDATE requests SET state='settled',used_input=?,used_output=?,used_cost=?,cached_tokens=?,reasoning_tokens=?,returned_model=?,provider_request_id=?,provider_response_id=?,latency=? WHERE id=?",(i,o,cfg.price(i,o),cached,reasoning,MODEL,request_identity,response_identity,latency,request_id))
            if hasattr(usage,'metadata'):
                safe=safe_diagnostic('accounting_settled','none',{'usage':usage.metadata(),'settled':True,'terminal_verified':True,'elapsed_seconds':latency})
                db.execute('INSERT INTO diagnostics VALUES(?,?,?)',(request_id,json.dumps(safe,sort_keys=True,separators=(',',':')),time.time()))
            self.event(db,request_id,'settled_usage_upper_bound')
    def diagnostic(self,request_id,value):
        safe=safe_diagnostic(value.get('stage'),value.get('code'),value)
        with self.db() as db:
            if not db.execute('SELECT 1 FROM requests WHERE id=? AND owner=?',(request_id,self.owner)).fetchone():raise BudgetUnavailable('diagnostic_owner_mismatch')
            db.execute('INSERT INTO diagnostics VALUES(?,?,?)',(request_id,json.dumps(safe,sort_keys=True,separators=(',',':')),time.time()))
    def diagnostics(self,request_id):
        with self.db() as db:
            return [json.loads(row[0]) for row in db.execute('SELECT metadata FROM diagnostics WHERE id=? ORDER BY rowid',(request_id,))]
    def status(self):
        with self.db() as db:
            rows=db.execute('SELECT state,count(*) FROM requests GROUP BY state').fetchall()
            usage=db.execute("SELECT coalesce(sum(used_input),0),coalesce(sum(used_output),0) FROM requests WHERE state='settled'").fetchone()
            return {'states':dict(rows),'input_tokens':usage[0],'output_tokens':usage[1],'total_tokens':usage[0]+usage[1],'unknown_or_sent':bool(db.execute("SELECT 1 FROM requests WHERE state='sent' OR (state='unknown' AND NOT "+self.debit_match+")").fetchone())}

    def conservative_debit(self,cfg,request_id,decision,review,*,dry_run=True):
        """Reviewed manual operator boundary, not a public/automatic reconciliation API.

        Attestation hashes bind independent review; code cannot authenticate provider
        evidence from a reviewer label. Actual operation needs a separately verified
        evidence chain and authorised operator. This task exercises temp fixtures only.
        """
        from app.cloud_config import timestamp
        def reject():raise BudgetUnavailable('conservative_debit_prerequisites_unverified')
        with self.db() as db:
            columns=['id','budget','body_hash','pricing_hash','res_cost','res_input','res_output','state','provider_request_id','provider_response_id','requested_model','returned_model','created','used_input','used_output','used_cost']
            record=db.execute('SELECT '+','.join(columns)+' FROM requests WHERE id=?',(request_id,)).fetchone()
            if not record:reject()
            row=dict(zip(columns,record))
            binding=db.execute('SELECT binding FROM budgets WHERE id=?',(cfg.budget_id,)).fetchone()
            ceiling=db.execute('SELECT cap_fingerprint FROM policy_guard WHERE singleton=1').fetchone()
            try:
                risk='user_risk_acceptance' in decision
                expected={'local_request_id','project','body_sha256','pricing_hash','approval_binding','configuration_sha256','ceiling_fingerprint','reservation_sha256','completion_evidence','bound_evidence'}
                if risk:expected=(expected-{'completion_evidence','bound_evidence'})|{'user_risk_acceptance'}
                if set(decision)!=expected or not binding or binding[0]!=cfg.binding() or not ceiling or ceiling[0]!=fingerprint({'ceiling':cfg.cap_units}):reject()
                match={'local_request_id':row['id'],'project':row['budget'],'body_sha256':row['body_hash'],'pricing_hash':row['pricing_hash'],'approval_binding':binding[0],'ceiling_fingerprint':ceiling[0],'reservation_sha256':fingerprint({k:row[k] for k in ('res_cost','res_input','res_output')})}
                if any(decision[k]!=v for k,v in match.items()) or row['budget']!=cfg.budget_id or (not risk and row['pricing_hash']!=cfg.pricing_hash()) or row['state']!='unknown' or any(row[k] is not None for k in ('used_input','used_output','used_cost')):reject()
                if row['requested_model']!=MODEL or row['returned_model']!=MODEL:reject()
                if any(type(row[k]) is not int or row[k]<0 for k in ('res_cost','res_input','res_output')):reject()
                if risk:
                    acceptance=decision['user_risk_acceptance']
                    fields={'decision_id','source','source_reference','accept_original_cost_uncertainty','preserve_original_full_reservation','actual_usage','provider_verified','user_reported_total_tokens'}
                    if set(acceptance)!=fields or acceptance['source']!='explicit_user_decision' or not re.fullmatch('[A-Za-z0-9_-]{1,100}',acceptance['decision_id']) or not re.fullmatch('orca-task:task_[a-f0-9]+',acceptance['source_reference']) or acceptance['accept_original_cost_uncertainty'] is not True or acceptance['preserve_original_full_reservation'] is not True or acceptance['actual_usage']!='unknown' or acceptance['provider_verified'] is not False or type(acceptance['user_reported_total_tokens']) is not int or not 0<=acceptance['user_reported_total_tokens']<=2**31:reject()
                    if set(review)!={'reviewer','decision_sha256','configuration_sha256','user_decision_id'} or review['reviewer'] not in ('Dana','Marcus') or review['decision_sha256']!=fingerprint(decision) or review['configuration_sha256']!=decision['configuration_sha256'] or review['user_decision_id']!=acceptance['decision_id']:reject()
                    if not isinstance(decision['configuration_sha256'],str) or not re.fullmatch('[a-f0-9]{64}',decision['configuration_sha256']):reject()
                else:
                    complete=decision['completion_evidence'];bound=decision['bound_evidence']
                    if set(complete)!={'artifact_sha256','kind','provider_request_id','provider_response_id','status','terminal_at'} or complete['kind']!='authenticated_same_request_terminal' or complete['provider_request_id']!=row['provider_request_id'] or complete['provider_response_id']!=row['provider_response_id'] or not row['provider_request_id'] or not row['provider_response_id'] or complete['status'] not in ('completed','incomplete','failed','cancelled') or timestamp(complete['terminal_at']).timestamp()<row['created']:reject()
                    if set(bound)!={'artifact_sha256','kind','request_sha256','provider_request_id','model','service_tier','pricing_hash','all_components_bounded'} or bound['kind']!='independently_reviewed_applicable_fee_bound' or bound['request_sha256']!=row['body_hash'] or bound['provider_request_id']!=row['provider_request_id'] or bound['model']!=MODEL or bound['service_tier']!='default' or bound['pricing_hash']!=row['pricing_hash'] or bound['all_components_bounded'] is not True:reject()
                    if cfg.price(row['res_input'],row['res_output'])>row['res_cost']:reject()
                    if set(review)!={'reviewer','decision_sha256','configuration_sha256','completion_artifact_sha256','bound_artifact_sha256'} or review['reviewer'] not in ('Dana','Marcus') or review['decision_sha256']!=fingerprint(decision) or review['configuration_sha256']!=decision['configuration_sha256'] or review['completion_artifact_sha256']!=complete['artifact_sha256'] or review['bound_artifact_sha256']!=bound['artifact_sha256']:reject()
                    for value in (decision['configuration_sha256'],complete['artifact_sha256'],bound['artifact_sha256']):
                        if not isinstance(value,str) or not re.fullmatch('[a-f0-9]{64}',value):reject()
            except (KeyError,TypeError,ValueError):reject()
            if db.execute("SELECT 1 FROM requests WHERE state IN ('sent','reserved')").fetchone():reject()
            old=db.execute('SELECT decision_hash,review_hash FROM conservative_debits WHERE id=?',(request_id,)).fetchone()
            hashes=(fingerprint(decision),fingerprint(review))
            if old and old!=hashes:reject()
            remaining=db.execute("SELECT coalesce(sum(coalesce(used_cost,res_cost)),0) FROM requests WHERE state!='released'").fetchone()[0]<cfg.cap_units
            result={'dry_run':dry_run,'already_recorded':bool(old),'actual_usage':'unknown','provider_settled':False,'reservation_permanently_counted':bool(old) or not dry_run,'remaining_budget_positive':remaining,'external_evidence_authentication_required':not risk,'decision_basis':'explicit_user_risk_acceptance' if risk else 'verified_provider_evidence','provider_verified':False if risk else None}
            if dry_run or old:return result
            db.execute('INSERT INTO conservative_debits VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(row['id'],row['budget'],row['body_hash'],row['pricing_hash'],row['res_cost'],row['res_input'],row['res_output'],hashes[0],hashes[1],ceiling[0],time.time(),json.dumps(decision,sort_keys=True,separators=(',',':'))))
            self.event(db,request_id,'full_reservation_permanently_consumed_unknown_usage')
            return result

    def migrate_policy(self,cfg):
        """Explicit offline adoption of a reviewed policy; never reclaim pending usage."""
        cfg.validate(cfg.scopes[0])
        with self.db() as db:
            if db.execute("SELECT 1 FROM requests WHERE state IS NULL OR state NOT IN ('settled','released')").fetchone():raise BudgetUnavailable('migration_pending_reconciliation')
            if db.execute("SELECT 1 FROM requests WHERE state='settled' AND (used_cost IS NULL OR used_input IS NULL OR used_output IS NULL OR requested_model IS NULL OR requested_model!=? OR typeof(used_cost)!='integer' OR typeof(used_input)!='integer' OR typeof(used_output)!='integer' OR used_cost<0 OR used_input<0 OR used_output<0)",(MODEL,)).fetchone():raise BudgetUnavailable('migration_incompatible_usage')
            guard=db.execute('SELECT cap_fingerprint FROM policy_guard WHERE singleton=1').fetchone()
            if guard and fingerprint({'ceiling':cfg.cap_units})!=guard[0]:raise BudgetUnavailable('cap_change_requires_authorization')
            rows=db.execute("SELECT used_input,used_output,used_cost FROM requests WHERE state='settled'").fetchall()
            # Retain higher historical accounting, never discount/reprice downward.
            total=sum(max(cost,cfg.price(i,o)) for i,o,cost in rows)
            if total>=cfg.cap_units or len(rows)>=cfg.max_requests or sum(r[0] for r in rows)>cfg.max_input_tokens or sum(r[1] for r in rows)>cfg.max_output_tokens or sum(r[0]+r[1] for r in rows)>cfg.max_total_tokens:raise BudgetUnavailable('migration_budget_exhausted')
            for request_id,i,o,cost in db.execute("SELECT id,used_input,used_output,used_cost FROM requests WHERE state='settled'").fetchall():
                db.execute('UPDATE requests SET used_cost=? WHERE id=?',(max(cost,cfg.price(i,o)),request_id))
            db.execute('UPDATE requests SET budget=?',(cfg.budget_id,))
            db.execute('DELETE FROM budgets')
            db.execute('INSERT INTO budgets VALUES(?,?)',(cfg.budget_id,cfg.binding()))
            db.execute('INSERT OR REPLACE INTO policy_guard VALUES(1,?)',(fingerprint({'ceiling':cfg.cap_units}),))
            self.event(db,'policy-migration','reviewed_policy_adopted_preserving_usage')
