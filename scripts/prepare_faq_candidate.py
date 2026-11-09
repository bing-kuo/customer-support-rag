"""Prepare an explicitly unapproved FAQ proposal without changing approved corpus."""
import json
import shutil
from pathlib import Path
import app.knowledge as knowledge
from app.lifecycle import canonical,corpus_identity

original=knowledge.ROOT
folder=original.parent/'eval'/'faq-proposal'
if folder.exists():raise SystemExit('Refusing to overwrite existing proposal')
(folder/'documents').mkdir(parents=True)
values=json.loads((original/'manifest.json').read_bytes())
for d in values:shutil.copyfile(original/'documents'/d['filename'],folder/'documents'/d['filename'])
body='# 韌體版本查詢\n\n合成展示資料，非真實商品政策。\n\n## 人工協助\n本合成展示不提供自動韌體更新；請由人工協助查詢版本資訊。\n'
name='general-firmware-v1.md';(folder/'documents'/name).write_text(body)
values.append({'doc_id':'general-firmware','revision':'1','title':'韌體版本查詢','product_ids':[],'locale':'zh-TW','policy_scope':'general-firmware','effective_from':'2026-11-08','effective_to':None,'status':'candidate','content_sha256':knowledge.digest(body.encode()),'approved_by':None,'approved_at':None,'supersedes':None,'filename':name})
raw=json.dumps(values,ensure_ascii=False,indent=2).encode()+b'\n';(folder/'manifest.json').write_bytes(raw)
knowledge.ROOT=folder
release=knowledge.build(folder/'candidate.sqlite3')
summary={'status':'quarantined-unapproved','serving_active_changed':False,'new_document':values[-1],'documents':len(values),'manifest_sha256':knowledge.digest(raw),'corpus_sha256':corpus_identity(values),'candidate_release':release,'genuine_approval':None,'generation_evaluation':'not run; pending Marcus exact corpus approval','next':'Marcus must review exact proposal manifest/content hashes; new frozen evaluation revision required before readiness'}
(folder/'proposal-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False))
