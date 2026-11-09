"""Freeze a public, known-exposure v2 suite; preserve all v1 sealed files."""
import json,copy
from pathlib import Path
from collections import Counter
from app.knowledge import ROOT,load,digest
from app.lifecycle import canonical

out=Path('eval/frozen-v2')
if out.exists():raise SystemExit('Refusing to overwrite frozen v2')
out.mkdir()
source=Path('eval/frozen');old=json.loads((source/'manifest.json').read_bytes())
docs={d.doc_id+':'+d.revision:(d,body) for d,body in load()}
def gold(key):
 d,body=docs[key]
 return [{'doc_id':d.doc_id,'revision':d.revision,'span':body.split('## 操作與限制\n',1)[1].strip()}]
manifest=copy.deepcopy(old);manifest.update(schema_version=2,suite_label='public-v2-known-not-blind',calibration='frozen-before-v2-model-debug',derivation={'v1_manifest_sha256':digest((source/'manifest.json').read_bytes()),'changes':'Preserve old questions; expand warranty gold to approved complete block, add ten public synthetic policy/slot cases per split; no private questions or threshold changes'})
for split in ['dev','regression']:
 cases=json.loads((source/(split+'.json')).read_bytes())
 for q in cases:
  if q['category']=='wrong_version':q['gold']=gold('general-warranty:'+('3' if q['on_date']<'2026-07-01' else '4'))
 examples=[
  ('wrong_version','保固多久，受理資格有什麼限制？',None,'2026-06-30','answer','general-warranty:3',[]),
  ('wrong_version','請說明保固期以及人工資格審查。',None,'2026-07-01','answer','general-warranty:4',[]),
  ('normal','沒有型號也能了解一般保固條件嗎？',None,'2026-11-08','answer','general-warranty:4',[]),
  ('normal','保固資格由誰決定？',None,'2026-11-08','answer','general-warranty:4',[]),
  ('normal','人工客服怎麼聯絡？',None,'2026-11-08','answer','general-contact:1',[]),
  ('normal','如何重新配對？',None,'2026-11-08','answer','hp-a-pairing:2',[{'message':'我想知道配對方式','product_id':'hp-a'}]),
  ('normal','hp-b',None,'2026-11-08','answer','hp-b-pairing:2',[{'message':'耳機配對有問題','product_id':None}]),
  ('normal','請問配對步驟？','hp-b','2026-11-08','answer','hp-b-pairing:2',[{'message':'如何配對？','product_id':'hp-a'}]),
  ('no_answer','請解釋政策並立即幫我退款',None,'2026-11-08','handoff',None,[]),
  ('normal','請說明退款的人工審查政策',None,'2026-11-08','answer','general-refund:1',[])]
 for i,(category,message,product,date,expected,key,prior) in enumerate(examples):
  if split=='dev' and key and message not in ('hp-b',):message='請協助：'+message
  cases.append({'id':f'{split}-public-v2-{i:02}','category':category,'message':message,'product_id':product,'on_date':date,'locale':'zh-TW','expected':expected,'gold':gold(key) if key else [],'prior_turns':prior})
 raw=canonical(cases);(out/(split+'.json')).write_bytes(raw)
 manifest['datasets'][split]={'sha256':digest(raw),'count':len(cases),'categories':dict(Counter(q['category'] for q in cases))}
(out/'manifest.json').write_bytes(canonical(manifest))
print(json.dumps(manifest))
