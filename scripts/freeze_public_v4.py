"""Add explicit critical-action requirements; preserve all prior raw bytes/gold."""
import copy,json
from pathlib import Path
from app.knowledge import digest
from app.lifecycle import canonical
source=Path('eval/frozen-v3');out=Path('eval/frozen-v4')
if out.exists():raise SystemExit('Refusing to replace frozen evidence')
out.mkdir();manifest=copy.deepcopy(json.loads((source/'manifest.json').read_bytes()));changes=[]
for split in ['dev','regression']:
    raw=(source/(split+'.json')).read_bytes();cases=json.loads(raw)
    for case in cases:
        actions=[dict(g) for g in case['gold'] if g['doc_id'].endswith('-safety')]
        if actions:
            case['required_actions']=actions;changes.append({'split':split,'id':case['id'],'added_required_actions':actions,'expected_unchanged':case['expected']})
    data=canonical(cases);(out/(split+'.json')).write_bytes(data);manifest['datasets'][split]['sha256']=digest(data)
proof={'source_manifest_sha256':digest((source/'manifest.json').read_bytes()),'reason':'Existing approved safety gold must be present in rendered response for every status; contact summary alone is insufficient. Status/questions/gold/counts/thresholds unchanged. Known public failure cases retained.','suite_label':'public-v4-known-not-blind','changes':changes}
(out/'required-action-changes.json').write_bytes(canonical(proof));manifest.update(schema_version=4,suite_label='public-v4-known-not-blind',calibration='frozen-before-v4-inference',derivation=proof,required_action_changes_sha256=digest((out/'required-action-changes.json').read_bytes()))
(out/'manifest.json').write_bytes(canonical(manifest));print(json.dumps({'manifest_sha256':digest((out/'manifest.json').read_bytes()),'changes':changes},ensure_ascii=False))
