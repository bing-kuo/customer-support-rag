"""Pinned-release conversation with memory-only session state."""
import asyncio
import json
from datetime import date
from app.knowledge import ROOT, connect, search_candidate
from app.lifecycle import pin, verify
from app import planning
from app.providers import adapter
from app.ollama import DependencyError, CircuitOpen, QualityRejection, sentence_units

DB = ROOT / 'runtime.sqlite3'


def metadata(release):
    db=connect(DB)
    try:
        return verify(db,release)
    finally:
        db.close()


def withdrawn(doc):
    db=connect(DB)
    try:
        return bool(db.execute('SELECT 1 FROM withdrawn WHERE doc=?',(doc,)).fetchone())
    finally:
        db.close()


def source_value(release, chunk, offline=False):
    db=connect(DB)
    try:
        manifest=verify(db,release)
        row=db.execute('SELECT c.doc,c.section,c.text,c.start,c.end,p.revision FROM chunks c JOIN provenance p ON p.release=c.release AND p.id=c.id WHERE c.release=? AND c.id=?',(release,chunk)).fetchone()
        if not row or (not offline and not db.execute('SELECT 1 FROM events WHERE target=?',(release,)).fetchone()):
            return None
        d=next(d for d in manifest['documents'] if d['doc_id']==row[0] and d['revision']==row[5])
        gone=withdrawn(row[0])
        return {'source_id':release+chunk,'release_id':release,'doc_id':row[0],'revision':d['revision'],'title':'Withdrawn source' if gone else d['title'],'section':'' if gone else row[1],'text':None if gone else row[2],'status':'withdrawn' if gone else 'active' if (offline or pin(DB)==release) and (row[4] is None or row[4]>date.today().isoformat()) else 'superseded'}
    finally:
        db.close()


def policy_intent(message):
    return planning.classify(message).kind=='policy'


def troubleshooting(message):
    return planning.classify(message).product_required


def refund_action(message):
    text=message.casefold()
    if not any(term in text for term in ['退款','refund']):return False
    explanation=any(term in text for term in ['政策','規定','流程','條件','policy','review process','審查','rules','conditions','review'])
    request=any(term in text for term in ['我要退款','幫我退款','替我退款','退給我','執行','立即','保證','承諾','已退款','refund now','want a refund','guarantee','approve my','process my refund','issue a refund','give me a refund'])
    return request or not explanation


def prepare_input(session,message,incoming):
    model_reply=message.strip().casefold() in ('hp-a','hp-b')
    if incoming is None and model_reply:incoming=message.strip().casefold()
    if incoming is not None and session.get('product')!=incoming:
        pending=session.get('pending_issue') if session.get('product') is None else None
        if session.get('product') is not None:
            session['user_statements']=[];session.pop('active_safety_query',None)
        session.update(product=incoming,history=[],clarifications=0,attempted_steps=[])
        if pending is None:
            session.pop('pending_issue',None);session.pop('issue',None)
    query=session.get('pending_issue',session.get('issue',message)) if model_reply else message
    return query,session.get('product'),model_reply


def effective_safety_query(session,query):
    if planning.classify(query).kind=='safety':session['active_safety_query']=query
    return session.get('active_safety_query',query)


def retrieval_context(path,release,case):
    """Replay explicit slot inputs without reading model output or trusting reports."""
    session={'product':None,'history':[],'clarifications':0}
    for turn in case.get('prior_turns',[])+[case]:
        query,product,is_model_reply=prepare_input(session,turn['message'],turn.get('product_id'))
        if not is_model_reply:session['issue']=query
        query=effective_safety_query(session,query)
        plan=planning.classify(query)
        scope=None if plan.kind=='policy' else product
        if product is None and plan.product_required:session['pending_issue']=query
        else:session.pop('pending_issue',None)
    return query,product,scope


async def respond(session,payload,request_id,release,offline=False,on_date=None):
    for key in ['quality_code','generation_metrics']:session.pop(key,None)
    query,product,model_reply=prepare_input(session,payload.message,payload.product_id)
    if not model_reply:session['issue']=query
    session['user_statements']=(session.get('user_statements',[])+[payload.message])[-3:]
    if any(term in payload.message.casefold() for term in ['試過','已試','已經試','已經重設','已經重新配對','已經清潔','tried','already tried']):
        session['attempted_steps']=(session.get('attempted_steps',[])+[payload.message])[-3:]
    user_query=query
    query=effective_safety_query(session,query)
    effective=on_date or date.today().isoformat()
    plan=planning.classify(query)
    rows=planning.search(DB,release,query,product,effective)
    allowed={r[0] for r in planning.pool(DB,release,query,product,effective)}
    rows=[r for r in rows if r[0] in allowed]
    required=planning.obligations(DB,release,query,product,effective)
    session['response_plan']={'intent':plan.kind,'user_intent':planning.classify(user_query).kind,'operation':plan.operation,'topics':list(plan.topics),'product_required':plan.product_required,'critical_obligations':required}
    result=dict(status='unavailable',answer=None,clarification=None,citations=[],handoff_summary=None,release_id=release,request_id=request_id)
    def citation(sid):
        value=source_value(release,sid,offline=offline)
        if not value or value['status']=='withdrawn':raise DependencyError('Evidence withdrawn')
        return {key:value[key] for key in ['source_id','doc_id','revision','section']} | {'excerpt':value['text'][:600],'url':'/api/v1/sources/'+value['source_id']}
    def finish():
        if required:
            session.pop('pending_issue',None)
            fresh=planning.obligations(DB,release,query,product,effective)
            if fresh!=required:raise DependencyError('Safety evidence changed')
            ids=list(dict.fromkeys(sid for item in required for sid in item['source_ids']))
            result.update(status='answer',answer='\n'.join(item['text'] for item in required),clarification=None,handoff_summary=None,citations=[citation(sid) for sid in ids])
            session['response_plan']['render_mode']='approved_critical_instruction_quote'
        if planning.critical_failures({'status':result['status'],'answer':result['answer'],'citations':[c['source_id'][64:] for c in result['citations']]},required):
            raise DependencyError('Critical instruction missing')
        from app.contracts import Turn
        Turn.model_validate(result)
        return result
    def handoff(reason):
        result.update(status='handoff',handoff_summary={'issue':'User statement: '+session.get('issue',query),'confirmed_product':product,'attempted_steps':session.get('attempted_steps',[]),'missing_fields':['product'] if not product and troubleshooting(query) else [],'reason':reason,'source_ids':[],'user_statements':session['user_statements']})
        return finish()
    if required:return finish()
    if model_reply and not session.get('pending_issue') and not session.get('issue'):
        if session['clarifications']>=2:return handoff('clarification_limit')
        session['clarifications']+=1
        result.update(status='clarify',clarification={'question':'請說明耳機遇到的問題。','options':[]})
        return finish()
    if refund_action(query):return handoff('refund_request')
    if product is None and (plan.product_required or plan.kind=='safety'):
        session['pending_issue']=query
        if session['clarifications']<2:
            session['clarifications']+=1
            result.update(status='clarify',clarification={'question':'請問耳機是哪個型號？','options':['hp-a','hp-b']})
            return finish()
        return handoff('clarification_limit')
    if plan.kind=='task' and not plan.topics:
        if session['clarifications']>=2:return handoff('clarification_limit')
        session['clarifications']+=1
        result.update(status='clarify',clarification={'question':'請說明耳機遇到的問題。','options':[]})
        return finish()
    if plan.kind=='ambiguous':
        if session['clarifications']>=2:return handoff('clarification_limit')
        session['clarifications']+=1
        result.update(status='clarify',clarification={'question':'請確認要排查耳機問題，還是詢問一般政策？','options':[]})
        return finish()
    general_policy=plan.kind=='policy'
    if not rows:return handoff('no_evidence')
    session.pop('pending_issue',None)
    docs={(d['doc_id'],d['revision']):d for d in metadata(release)['documents']}
    evidence=[]
    for row in rows[:4]:
        value=source_value(release,row[0],offline=offline)
        if value and value['status']!='withdrawn':
            doc=docs[(value['doc_id'],value['revision'])]
            evidence.append({'source_id':value['source_id'],'text':value['text'],'required_sentences':sentence_units(value['text']) if not doc['product_ids'] else [],'policy_block':general_policy})
    # Shield preserves the generation slot until upstream finishes after client timeout.
    token=None
    if hasattr(adapter,'bind_request'):
        import hashlib,uuid
        namespace=session.setdefault('_generation_namespace',uuid.uuid4().hex)
        identity=hashlib.sha256(json.dumps([namespace,payload.client_turn_id],separators=(',',':')).encode()).hexdigest()
        token=adapter.bind_request(identity)
    task=asyncio.create_task(adapter.generate(query,evidence,session['history']))
    if token is not None:adapter.reset_request(token)
    session['background']=task
    def consume(future):
        if not future.cancelled():
            future.exception()
    task.add_done_callback(consume)
    try:
        generated,generation_raw=await asyncio.wait_for(asyncio.shield(task),timeout=75)
    except CircuitOpen:
        return finish()
    except QualityRejection as exc:
        session['quality_code']=exc.code
        return handoff('no_evidence')
    except (asyncio.CancelledError,asyncio.TimeoutError):
        # Deliver cancellation through the outer shield. Adapter cancellation only
        # cancels work before the actual upstream request has started.
        task.cancel()
        try:await task
        except (asyncio.CancelledError,Exception):pass
        upstream=getattr(task,'upstream',None)
        if upstream is not None:
            if not getattr(upstream,'upstream_started',False):
                upstream.cancel()
                await asyncio.gather(upstream,return_exceptions=True)
            session['background']=upstream
        raise
    session['generation_metrics']={k:generation_raw.get(k) for k in ['provider','requested_model','observed_at','input_tokens','output_tokens','total_tokens','cached_tokens','reasoning_tokens','cache_write_tokens','ttl_30m_tokens','ttl_12h_tokens','pricing_hash','approval_binding','provider_request_id','provider_response_id','latency_seconds','model','total_duration','load_duration','prompt_eval_count','prompt_eval_duration','eval_count','eval_duration','done_reason','verified_model_tags','requested_model_digest','render_mode','model_output_kind','answer_origin']}
    if not generated.answer:
        session['quality_code']='policy_not_supported' if generation_raw.get('render_mode')=='approved_policy_block_quote' else 'source_not_supported' if generation_raw.get('render_mode')=='approved_procedural_source_quote' else 'empty_model_answer'
        return handoff('no_evidence')
    citations=[]
    for sid in dict.fromkeys(generated.source_ids):
        if sid not in {item['source_id'] for item in evidence}:
            raise DependencyError('Ineligible source')
        value=source_value(release,sid[64:],offline=offline)
        if not value or value['status']=='withdrawn':
            raise DependencyError('Evidence withdrawn')
        citations.append({key:value[key] for key in ['source_id','doc_id','revision','section'] } | {'excerpt':value['text'][:600],'url':'/api/v1/sources/'+sid})
    allowed_sentences={sentence for item in evidence if item['source_id'] in generated.source_ids for sentence in sentence_units(item['text'])}
    if any(sentence not in allowed_sentences for sentence in sentence_units(generated.answer)):
        session['quality_code']='non_verbatim_sentence'
        return handoff('no_evidence')
    fresh_allowed={r[0] for r in planning.pool(DB,release,query,product,effective)}
    if any(c['source_id'][64:] not in {r[0] for r in rows[:4]} & fresh_allowed for c in citations):return handoff('no_evidence')
    if any(required_sentence not in sentence_units(generated.answer) for item in evidence if item['source_id'] in generated.source_ids for required_sentence in item['required_sentences']):
        session['quality_code']='missing_qualifier'
        return handoff('no_evidence')
    result.update(status='answer',answer=generated.answer,citations=citations)
    # Store bounded slots only; never silently truncate history to fit a model request.
    session['history']=(session['history']+[{'product':product,'issue':query}])[-3:]
    return finish()
