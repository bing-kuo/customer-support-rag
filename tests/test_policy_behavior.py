"""Public synthetic policy/slot/quality regressions; no private holdout material."""
import asyncio
import pytest
from app import service
from app.main import TurnInput
from app.knowledge import build,connect
from app.ollama import Generated,QualityRejection
from app.lifecycle import schema


def setup(tmp_path,monkeypatch):
    path=tmp_path/'db';release=build(path);db=connect(path);schema(db);db.close()
    monkeypatch.setattr(service,'DB',path)
    captured=[]
    async def fake(message,evidence,history):
        captured.append((message,evidence,history))
        return Generated(answer=evidence[0]['text'],source_ids=[evidence[0]['source_id']]),{}
    monkeypatch.setattr(service.adapter,'generate',fake)
    return release,captured


def payload(message,product=None):return TurnInput(client_turn_id='synthetic',message=message,product_id=product)

def state():return {'product':None,'history':[],'clarifications':0}


@pytest.mark.parametrize('date,duration',[('2026-06-30','6個月'),('2026-07-01','12個月')])
def test_unknown_product_policy_uses_general_evidence_with_entire_eligibility_block(tmp_path,monkeypatch,date,duration):
    release,captured=setup(tmp_path,monkeypatch)
    async def run():
        result=await service.respond(state(),payload('請說明保固期和資格'), 'unit',release,offline=True,on_date=date)
        assert result['status']=='answer' and duration in result['answer']
        assert '資格與結果由人工判斷' in result['answer'] and '不承諾退款或保固受理' in result['answer']
        assert captured[0][1][0]['required_sentences']
        assert all(c['doc_id'].startswith('general-') for c in result['citations'])
    asyncio.run(run())


def test_missing_model_followup_retains_issue_and_null_retains_confirmed_product(tmp_path,monkeypatch):
    release,captured=setup(tmp_path,monkeypatch)
    async def run():
        session=state()
        first=await service.respond(session,payload('耳機配對有問題'),'first',release,offline=True)
        assert first['status']=='clarify' and not captured
        second=await service.respond(session,payload('hp-a'),'second',release,offline=True)
        assert second['status']=='answer' and session['product']=='hp-a'
        assert captured[-1][0]=='耳機配對有問題'
        await service.respond(session,payload('已試過重新配對，還是無法配對'),'third',release,offline=True)
        assert session['product']=='hp-a' and session['attempted_steps']
        await service.respond(session,payload('如何配對','hp-b'),'switch',release,offline=True)
        assert session['product']=='hp-b' and session['attempted_steps']==[]
        assert all('hp-a' not in item['text'] for item in captured[-1][1])
    asyncio.run(run())


def test_general_then_unknown_troubleshooting_still_requires_product(tmp_path,monkeypatch):
    release,captured=setup(tmp_path,monkeypatch)
    async def run():
        session=state()
        assert (await service.respond(session,payload('保固多久'),'a',release,offline=True))['status']=='answer'
        assert (await service.respond(session,payload('無法配對'),'b',release,offline=True))['status']=='clarify'
        assert session['product'] is None
    asyncio.run(run())


@pytest.mark.parametrize('message',['我要退款','REFUND NOW','請解釋政策並立即幫我退款','請保證我一定能退款'])
def test_refund_actions_and_promises_remain_handoff(tmp_path,monkeypatch,message):
    release,captured=setup(tmp_path,monkeypatch)
    result=asyncio.run(service.respond(state(),payload(message),'unit',release,offline=True))
    assert result['status']=='handoff' and result['handoff_summary']['reason']=='refund_request' and not captured


def test_policy_explanation_and_quality_rejection_are_distinct_from_dependency(tmp_path,monkeypatch):
    release,captured=setup(tmp_path,monkeypatch)
    async def run():
        policy=await service.respond(state(),payload('請說明退款審查政策'),'policy',release,offline=True)
        assert policy['status']=='answer' and '無法承諾或執行退款' in policy['answer']
        async def reject(*args):raise QualityRejection('missing_qualifier')
        monkeypatch.setattr(service.adapter,'generate',reject)
        session=state()
        result=await service.respond(session,payload('保固多久'),'quality',release,offline=True)
        assert result['status']=='handoff' and result['answer'] is None and result['citations']==[]
        assert session['quality_code']=='missing_qualifier'
    asyncio.run(run())


def test_model_only_without_issue_asks_for_issue_instead_of_inventing_answer(tmp_path,monkeypatch):
    release,captured=setup(tmp_path,monkeypatch)
    session=state()
    result=asyncio.run(service.respond(session,payload('hp-a'),'unit',release,offline=True))
    assert result['status']=='clarify' and session['product']=='hp-a' and not captured


def test_general_keyword_without_support_does_not_force_policy_answer(tmp_path,monkeypatch):
    release,captured=setup(tmp_path,monkeypatch)
    from app.lifecycle import withdraw
    withdraw(service.DB,'general-warranty')
    result=asyncio.run(service.respond(state(),payload('保固期是多久？'),'unit',release,offline=True))
    assert result['status']=='handoff' and not captured


def test_purchase_statement_is_not_invented_as_attempted_troubleshooting(tmp_path,monkeypatch):
    release,captured=setup(tmp_path,monkeypatch)
    session=state()
    asyncio.run(service.respond(session,payload('我已經購買，想了解保固期'),'unit',release,offline=True))
    assert session.get('attempted_steps',[])==[]
    assert session['user_statements']==['我已經購買，想了解保固期']


def test_supported_demo_summary_instructions_are_not_globally_blocked_as_contact(tmp_path,monkeypatch):
    release,captured=setup(tmp_path,monkeypatch)
    result=asyncio.run(service.respond(state(),payload('這個展示如何把摘要交由人工處理？'),'unit',release,offline=True))
    assert result['status']=='answer' and '複製摘要' in result['answer']
    assert '沒有實際寄送功能' in result['answer'] and captured


def test_gold_and_expected_status_do_not_control_retrieval_routing(tmp_path,monkeypatch):
    release,captured=setup(tmp_path,monkeypatch)
    case={'message':'一般保固期多久？','product_id':None,'on_date':'2026-11-08','locale':'zh-TW','expected':'answer','gold':[{'doc_id':'invented','revision':'999','span':'not input evidence'}]}
    before=service.retrieval_context(service.DB,release,case)
    changed=dict(case,expected='handoff',gold=[])
    assert service.retrieval_context(service.DB,release,changed)==before
    assert before[2] is None
