"""Declared knowledge capabilities and source-bound response obligations.

This finite PoC ontology is not a semantic classifier. Unknown/ambiguous tasks
are explicit; an arbitrary FTS hit never establishes question answerability.
"""
import re
import json
import sqlite3
from dataclasses import dataclass
from app.knowledge import eligible,tokens,connect

# Public document scopes, with domain paraphrases; no private cases are inputs.
CAPABILITIES={
 'pairing':('task','配對',('配對','藍牙','連線','連接','bluetooth','pairing','pair','connect')),
 'charging':('task','充電',('充電','充不','電池','charging','charge','battery')),
 'compatibility':('task','相容性',('相容','支援裝置','compatibility','compatible')),
 'cleaning':('task','清潔',('清潔','擦拭','cleaning','clean')),
 'audio':('task','聲音',('聲音','聲道','音量','聽不到','audio','sound','volume')),
 'microphone':('task','麥克風',('麥克風','收音','microphone','mic')),
 'reset':('task','重設',('重設','重置','reset')),
 'warranty':('policy','保固',('保固','保修','warranty')),
 'refund':('policy','退款',('退款政策','退款規定','退款流程','退款條件','退款審查','refund policy','review process')),
 'contact':('policy','人工聯絡',('聯絡','聯繫','摘要','contact','summary')),
 'safety':('safety','安全',('發熱','過熱','燙','overheat','overheating','hot')),
}
PRODUCTS=('hp-a','hp-b')

@dataclass(frozen=True)
class Plan:
    kind:str
    topics:tuple
    product_required:bool
    operation:str="explain"


def contains(text,term):
    return bool(re.search(r'(?<![a-z])'+re.escape(term)+r'(?![a-z])',text)) if term.isascii() else term in text


def classify(message):
    text=message.casefold()
    # Explicitly negated heat symptoms do not create an urgent obligation.
    heat=re.sub(r'(沒有|並未|不會|不是|未|無|不)(異常)?(發熱|過熱|燙)', '',text)
    heat=re.sub(r'\b(not|no)\s+(overheating|overheat|hot)\b','',heat)
    if any(contains(heat,t) for t in CAPABILITIES['safety'][2]):return Plan('safety',('safety',),False)
    for _,_,terms in CAPABILITIES.values():
        for term in terms:
            text=re.sub(r'(不要|不需|不是在問|不是|不問|不談|別談|ignore|not about)\s*'+re.escape(term),'',text)
    refund_policy=any(contains(text,noun) for noun in ['退款','refund']) and any(contains(text,marker) for marker in ['政策','規定','規則','條件','流程','審查','policy','rules','conditions','process','review'])
    topics=tuple(k for k,(kind,_,terms) in CAPABILITIES.items() if k!='safety' and any(contains(text,t) for t in terms))
    if refund_policy and 'refund' not in topics:topics=topics+('refund',)
    tasks=tuple(t for t in topics if CAPABILITIES[t][0]=='task')
    policies=tuple(t for t in topics if CAPABILITIES[t][0]=='policy')
    if tasks:return Plan('ambiguous' if policies else 'task',topics,True)
    if not topics and any(contains(text,t) for t in ['排查','故障排除','troubleshoot','diagnose']):return Plan('task',(),True,'diagnose')
    if topics==('contact',):
        workflow=any(contains(text,t) for t in ['摘要','複製','summary','copy']) or (contains(text,'demo') and contains(text,'transfer'))
        return Plan('policy',topics,False,'summary_workflow' if workflow else 'contact_channel')
    return Plan('policy' if policies else 'unknown',topics,False)


def source_topics(path,release):
    scopes={'general-'+t:t for t in CAPABILITIES if CAPABILITIES[t][0]=='policy'}
    scopes.update({p+'-'+t:t for p in PRODUCTS for t in CAPABILITIES if CAPABILITIES[t][0]!='policy'})
    db=connect(path)
    try:
        manifest=json.loads(db.execute('SELECT manifest FROM releases WHERE id=?',(release,)).fetchone()[0])
        docs={(d['doc_id'],d['revision']):d['policy_scope'] for d in manifest['documents']}
        return {cid:scopes.get(docs[(doc,rev)]) for cid,doc,rev in db.execute('SELECT c.id,c.doc,p.revision FROM chunks c JOIN provenance p ON c.release=p.release AND c.id=p.id WHERE c.release=?',(release,))}
    finally:db.close()


def pool(path,release,query,product,on_date,locale='zh-TW'):
    plan=classify(query)
    if any(item not in PRODUCTS for item in re.findall(r'(?<![a-z0-9])hp-[a-z0-9]+',query.casefold())):return []
    topics=source_topics(path,release)
    if plan.kind in ('unknown','ambiguous') or plan.operation=='contact_channel':return []
    if plan.kind=='task' and product is None:return []
    scope=None if plan.kind=='policy' else product
    rows=[r for r in eligible(path,release,scope,on_date,locale) if topics.get(r[0]) in plan.topics]
    if plan.kind=='safety' and product is None:
        # Explicit common-support mapping: require the same approved text for
        # every supported product. Never relabel one product document general.
        groups={}
        for item in PRODUCTS:
            for row in eligible(path,release,item,on_date,locale):
                if topics.get(row[0])=='safety':groups.setdefault(row[3],{})[item]=row
        rows=[row for members in groups.values() if set(members)==set(PRODUCTS) for row in members.values()]
    return sorted({r[0]:r for r in rows}.values())


def search(path,release,query,product,on_date,locale='zh-TW'):
    rows=pool(path,release,query,product,on_date,locale)
    plan=classify(query)
    terms=tokens(query+' '+' '.join(CAPABILITIES[t][1] for t in plan.topics)).split()
    if not rows or not terms:return []
    db=sqlite3.connect(':memory:')
    try:
        db.execute('CREATE VIRTUAL TABLE candidates USING fts5(id UNINDEXED,terms)')
        db.executemany('INSERT INTO candidates VALUES(?,?)',[(r[0],tokens(r[2]+' '+r[3])) for r in rows])
        ids=db.execute('SELECT id FROM candidates WHERE candidates MATCH ? ORDER BY bm25(candidates),id LIMIT 10',(' OR '.join('"'+t+'"' for t in terms),)).fetchall()
        mapping={r[0]:r for r in rows}
        return [mapping[x[0]] for x in ids]
    finally:db.close()


def obligations(path,release,query,product,on_date,locale='zh-TW'):
    if classify(query).kind!='safety':return []
    rows=pool(path,release,query,product,on_date,locale)
    return [{'text':text,'require_all_sources':product is None,'source_ids':[r[0] for r in rows if r[3]==text]} for text in dict.fromkeys(r[3] for r in rows)]


def critical_failures(response,required):
    # Every status is checked. A handoff/clarification is not an exemption.
    return [item for item in required if response.get('status')!='answer' or item['text'] not in (response.get('answer') or '') or not (set(item['source_ids']) <= set(response.get('citations',[])) if item.get('require_all_sources') else set(item['source_ids']) & set(response.get('citations',[])))]


def pool_key(query,product,on_date,locale):
    plan=classify(query)
    supported_entities=all(item in PRODUCTS for item in re.findall(r'(?<![a-z0-9])hp-[a-z0-9]+',query.casefold()))
    return (plan.kind,plan.topics,plan.operation,product,on_date,locale,supported_entities)
