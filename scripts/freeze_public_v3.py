"""Apply one independently adjudicated public gold correction; preserve v1/v2."""
import json,copy
from pathlib import Path
from collections import Counter
from app.knowledge import digest
from app.lifecycle import canonical
source=Path('eval/frozen-v2');out=Path('eval/frozen-v3')
if out.exists():raise SystemExit('Refusing to overwrite frozen v3')
out.mkdir()
manifest=copy.deepcopy(json.loads((source/'manifest.json').read_bytes()))
manifest.update(schema_version=3,suite_label='public-v3-known-not-blind',calibration='frozen-before-v3-model-run',derivation={'source_v2_manifest_sha256':digest((source/'manifest.json').read_bytes()),'adjudicator':'Calvin','adjudication_message_id':'msg_e8bbf58edf7c','reviewer_evidence_path':'knowledge/documents/general-contact-v1.md','reason':'The question asks for a contact channel. Approved document provides only manual summary-copy instructions and explicitly no sending; no contact channel is supported. Only this demonstrated mislabel is corrected; supported demo-summary questions remain answerable.'})
changes=[]
for split in ['dev','regression']:
 cases=json.loads((source/(split+'.json')).read_bytes())
 q=next(q for q in cases if q['id']==split+'-public-v2-04')
 before=copy.deepcopy(q)
 q.update(category='no_answer',expected='handoff',gold=[])
 changes.append({'split':split,'before':before,'after':copy.deepcopy(q),'source_raw_sha256':digest((source/(split+'.json')).read_bytes())})
 raw=canonical(cases);(out/(split+'.json')).write_bytes(raw)
 manifest['datasets'][split]={'sha256':digest(raw),'count':len(cases),'categories':dict(Counter(q['category'] for q in cases))}
(out/'gold-changes.json').write_bytes(canonical({'reviewer':'Calvin','message_id':'msg_e8bbf58edf7c','evidence_path':'knowledge/documents/general-contact-v1.md','corpus_unchanged':True,'thresholds_unchanged':True,'changes':changes}))
manifest['gold_changes_sha256']=digest((out/'gold-changes.json').read_bytes())
(out/'manifest.json').write_bytes(canonical(manifest))
assert manifest['thresholds']==json.loads((source/'manifest.json').read_bytes())['thresholds']
print(json.dumps(manifest))
