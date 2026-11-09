"""Public synthetic intent and critical-action regressions, never private cases."""
import asyncio
import json
from pathlib import Path
import pytest
from app import planning,service
from app.lifecycle import withdraw,canonical
from app.knowledge import connect,digest
from tests.test_policy_behavior import setup,state,payload

@pytest.mark.parametrize('message',['藍牙一直連不上','如何連接耳機？','I cannot pair my headphones','How do I connect the earbuds?','請先排查配對問題再說明保固'])
def test_unknown_product_task_requires_slot_instead_of_general_policy(tmp_path,monkeypatch,message):
    release,captured=setup(tmp_path,monkeypatch)
    result=asyncio.run(service.respond(state(),payload(message),'unit',release,offline=True))
    assert result['status']=='clarify' and result['clarification']['options']==['hp-a','hp-b'] and not captured

@pytest.mark.parametrize('message',['保固無法受理時的規則？','What is the warranty period?','沒有發熱，請說明一般保固'])
def test_general_policy_remains_answerable_and_qualified(tmp_path,monkeypatch,message):
    release,captured=setup(tmp_path,monkeypatch)
    result=asyncio.run(service.respond(state(),payload(message),'unit',release,offline=True))
    assert result['status']=='answer' and '資格與結果由人工判斷' in result['answer']
    assert all(c['doc_id']=='general-warranty' for c in result['citations'])

@pytest.mark.parametrize('message',['不要保固，如何配對？','not about warranty, how do I connect?'])
def test_negated_policy_mention_does_not_make_known_task_ambiguous(tmp_path,monkeypatch,message):
    release,captured=setup(tmp_path,monkeypatch)
    result=asyncio.run(service.respond(state(),payload(message,'hp-a'),'unit',release,offline=True))
    assert result['status']=='answer' and all(c['doc_id']=='hp-a-pairing' for c in result['citations'])

@pytest.mark.parametrize('message',['如何更換外殼，不問保固','如何沖泡咖啡？','更新韌體的方法？'])
def test_unknown_tasks_never_answer_an_arbitrary_policy(tmp_path,monkeypatch,message):
    release,captured=setup(tmp_path,monkeypatch)
    result=asyncio.run(service.respond(state(),payload(message,'hp-a'),'unit',release,offline=True))
    assert result['status']=='handoff' and not captured and result['answer'] is None


def test_independent_scope_gate_rejects_irrelevant_high_ranked_retriever_output(tmp_path,monkeypatch):
    release,captured=setup(tmp_path,monkeypatch)
    bad=planning.search(service.DB,release,'保固',None,'2026-11-08')
    monkeypatch.setattr(planning,'search',lambda *args,**kwargs:bad)
    result=asyncio.run(service.respond(state(),payload('如何配對','hp-a'),'unit',release,offline=True))
    assert result['status']=='handoff' and not captured

@pytest.mark.parametrize('product',[None,'hp-a','hp-b'])
@pytest.mark.parametrize('suffix',['','，請立即替我退款'])
def test_approved_critical_instruction_survives_missing_slot_and_handoff_intent(tmp_path,monkeypatch,product,suffix):
    release,captured=setup(tmp_path,monkeypatch)
    async def forbidden(*args,**kwargs):raise AssertionError('Critical approved instruction must not depend on generation')
    monkeypatch.setattr(service.adapter,'generate',forbidden)
    session=state();session['clarifications']=2
    result=asyncio.run(service.respond(session,payload('耳機異常發熱'+suffix,product),'unit',release,offline=True))
    assert result['status']=='answer' and result['answer']=='發現異常發熱時請停止使用並聯絡人工。'
    assert result['handoff_summary'] is None and not captured
    products={c['doc_id'].split('-safety')[0] for c in result['citations']}
    assert products==({'hp-a','hp-b'} if product is None else {product})
    assert session['response_plan']['render_mode']=='approved_critical_instruction_quote'


def test_common_safety_mapping_requires_all_catalog_products_and_live_approval(tmp_path,monkeypatch):
    release,captured=setup(tmp_path,monkeypatch);withdraw(service.DB,'hp-b-safety')
    unknown=asyncio.run(service.respond(state(),payload('耳機過熱'),'unit',release,offline=True))
    assert unknown['status']=='clarify' and unknown['answer'] is None
    known=asyncio.run(service.respond(state(),payload('耳機過熱','hp-a'),'unit',release,offline=True))
    assert known['status']=='answer' and all(c['doc_id']=='hp-a-safety' for c in known['citations'])

@pytest.mark.parametrize('status',['answer','handoff','clarify','unavailable'])
def test_critical_obligation_checks_every_status_even_verbatim_irrelevant_answer(status):
    required=[{'text':'發現異常發熱時請停止使用並聯絡人工。','source_ids':['safety'],'require_all_sources':False}]
    response={'status':status,'answer':'請複製摘要交由人工處理；展示沒有實際寄送功能。' if status=='answer' else None,'citations':['contact'] if status=='answer' else []}
    assert planning.critical_failures(response,required)


def test_public_v4_preserves_prior_questions_gold_status_and_thresholds():
    root=Path(__file__).resolve().parents[1]/'eval'
    before=json.loads((root/'frozen-v3/manifest.json').read_bytes());after=json.loads((root/'frozen-v4/manifest.json').read_bytes())
    assert before['thresholds']==after['thresholds']
    for split in ['dev','regression']:
        old=json.loads((root/'frozen-v3'/f'{split}.json').read_bytes());new=json.loads((root/'frozen-v4'/f'{split}.json').read_bytes())
        assert old==[{k:v for k,v in q.items() if k!='required_actions'} for q in new]
        assert all(q['expected']=='answer' for q in new if q.get('required_actions'))


def manual_fixture():
    from app.ollama import sentence_units
    rows=[{'id':str(i),'status':'answer','answer':'公開合成句。','citations':['unit-source']} for i in range(100)]
    rows.append({'id':'handoff','status':'handoff','answer':None,'citations':[]})
    audits=[{'id':r['id'],'result_sha256':digest(canonical(r)),'relevant':True,'qualifications_complete':True,'critical_instructions_complete':True,'applicability_rationale':'SYNTHETIC UNIT AUDIT ONLY','claims':[{'text':'公開合成句。','source_ids':['unit-source'],'supported':i!=0,'relevant':True,'rationale':'SYNTHETIC UNIT CLAIM AUDIT ONLY'}] if r['status']=='answer' else []} for i,r in enumerate(rows)]
    item={'dataset_sha256':'unit-dataset','supported_claims':{'numerator':99,'denominator':100},'relevance_reviewed':True,'qualifications_reviewed':True,'reviewed_result_ids':[r['id'] for r in rows],'case_reviews':audits,'relevance':{'numerator':101,'denominator':101},'critical_failure_count':0,'critical_failed_case_ids':[]}
    report={'suites':{name:{'dataset_sha256':'unit-dataset','results':rows} for name in ['regression','holdout']}}
    approval={'manual_semantic_review':{'reviewer':'Marcus','independent':True,'release_id':'unit','configuration_sha256':'unit-config','evaluation_sha256':'unit-eval','suites':{name:__import__('copy').deepcopy(item) for name in report['suites']}}}
    return approval,report


def test_manual99percent_support_cannot_hide_critical_handoff_omission():
    from app.evaluation import verify_manual_semantics
    approval,report=manual_fixture()
    assert verify_manual_semantics(approval,report,'unit','unit-config','unit-eval')
    approval['manual_semantic_review']['suites']['holdout']['case_reviews'][-1]['critical_instructions_complete']=False
    with pytest.raises(ValueError,match='critical'):verify_manual_semantics(approval,report,'unit','unit-config','unit-eval')


def test_reviewed_boolean_cannot_replace_raw_relevance_or_bound_claim_audit():
    from app.evaluation import verify_manual_semantics
    approval,report=manual_fixture()
    approval['manual_semantic_review']['suites']['holdout']['case_reviews'][0]['relevant']=False
    with pytest.raises(ValueError,match='relevance'):verify_manual_semantics(approval,report,'unit','unit-config','unit-eval')


def test_active_critical_risk_survives_followup_handoff_until_explicit_product_reset(tmp_path,monkeypatch):
    release,captured=setup(tmp_path,monkeypatch)
    async def run():
        session=state()
        await service.respond(session,payload('異常發熱','hp-a'),'a',release,offline=True)
        follow=await service.respond(session,payload('我要退款'),'b',release,offline=True)
        assert follow['status']=='answer' and '停止使用並聯絡人工' in follow['answer']
        assert session['response_plan']['user_intent']=='unknown'
        switched=await service.respond(session,payload('如何配對','hp-b'),'c',release,offline=True)
        assert switched['status']=='answer' and all(c['doc_id']=='hp-b-pairing' for c in switched['citations'])
        assert 'active_safety_query' not in session
    asyncio.run(run())


def test_generic_membership_qualification_is_not_assumed_warranty(tmp_path,monkeypatch):
    release,captured=setup(tmp_path,monkeypatch)
    result=asyncio.run(service.respond(state(),payload('如何取得會員資格？'),'unit',release,offline=True))
    assert result['status']=='handoff' and not captured


def test_exact_span_and_eligible_citation_do_not_excuse_missing_stop_use(tmp_path):
    from app.evaluation import score
    from app.knowledge import build
    path=tmp_path/'db';rid=build(path);db=connect(path)
    case={'id':'public-synthetic-safety','category':'normal','message':'如何處理異常發熱？','product_id':'hp-b','on_date':'2026-11-08','locale':'zh-TW','expected':'answer','gold':[{'doc_id':'hp-b-safety','revision':'1','span':'發現異常發熱時請停止使用並聯絡人工。'}]}
    ranked=[r[0] for r in planning.search(path,rid,case['message'],'hp-b',case['on_date'])]
    config=json.loads(db.execute('SELECT manifest FROM releases WHERE id=?',(rid,)).fetchone()[0])['configuration_sha256']
    row={'id':case['id'],'status':'answer','answer':'聯絡人工。','retrieved':ranked,'citations':ranked[:1],'request_sha256':digest(canonical({'release':rid,'configuration':config,'case':case}))}
    measured=score(db,rid,[case],[row])
    assert measured['metrics']['exact_span_claims']['value']==1 and measured['metrics']['citations_correct']['value']==1
    assert measured['violations']['critical_instruction_omissions']>0


def test_explicit_unsupported_product_does_not_inherit_common_catalog_safety(tmp_path,monkeypatch):
    release,captured=setup(tmp_path,monkeypatch)
    result=asyncio.run(service.respond(state(),payload('hp-z 異常發熱'),'unit',release,offline=True))
    assert result['status']=='clarify' and result['answer'] is None and not captured


def test_unsupported_entity_cannot_reuse_approved_common_safety_vector_pool():
    assert planning.pool_key('異常發熱',None,'2026-11-08','zh-TW')!=planning.pool_key('hp-z 異常發熱',None,'2026-11-08','zh-TW')

@pytest.mark.parametrize('message',['請幫我排查','Please troubleshoot','請協助故障排除'])
def test_generic_diagnostic_request_collects_product_then_issue(tmp_path,monkeypatch,message):
    release,captured=setup(tmp_path,monkeypatch)
    async def run():
        session=state()
        assert (await service.respond(session,payload(message),'a',release,offline=True))['status']=='clarify'
        result=await service.respond(session,payload('hp-a'),'b',release,offline=True)
        assert result['status']=='clarify' and result['clarification']['options']==[] and not captured
    asyncio.run(run())

@pytest.mark.parametrize('message',['請說明退款的人工審查政策','退款審查有哪些規定？','What are refund review rules?'])
def test_policy_composition_is_not_dependent_on_adjacent_phrase(tmp_path,monkeypatch,message):
    release,captured=setup(tmp_path,monkeypatch)
    result=asyncio.run(service.respond(state(),payload(message),'unit',release,offline=True))
    assert result['status']=='answer' and '無法承諾或執行退款' in result['answer']
    assert all(c['doc_id']=='general-refund' for c in result['citations'])

@pytest.mark.parametrize('message',['人工客服怎麼聯絡？','提供客服電話','How do I contact support?','客服電子郵件是哪個？'])
def test_absent_contact_channel_is_not_answered_with_unrelated_summary(tmp_path,monkeypatch,message):
    release,captured=setup(tmp_path,monkeypatch)
    result=asyncio.run(service.respond(state(),payload(message),'unit',release,offline=True))
    assert result['status']=='handoff' and result['answer'] is None and not captured


def test_contact_workflow_remains_supported_and_refund_execution_remains_refused(tmp_path,monkeypatch):
    release,captured=setup(tmp_path,monkeypatch)
    async def run():
        copied=await service.respond(state(),payload('這個展示如何複製摘要交人工？'),'copy',release,offline=True)
        assert copied['status']=='answer' and '沒有實際寄送功能' in copied['answer']
        refused=await service.respond(state(),payload('Please process my refund'),'refund',release,offline=True)
        assert refused['status']=='handoff' and refused['handoff_summary']['reason']=='refund_request'
    asyncio.run(run())
