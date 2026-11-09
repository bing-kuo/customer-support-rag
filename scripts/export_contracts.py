import json
import re
from pathlib import Path
from app.contracts import Turn, Source, Error
from app.main import app

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('contract', type=Path)
args = parser.parse_args()
text = args.contract.read_text()
pack = json.loads(re.search(r'```json\n(.*?)\n```', text, re.S)[1])
pack['source-superseded'] = dict(pack['source'], status='superseded')
pack['source-withdrawn'] = dict(pack['source'], status='withdrawn', text=None, title='Withdrawn source', section='')
pack['source-xss'] = dict(pack['source'], text='<img src=x onerror=alert(1)>')
statuses = {}
for name, code, status in [('source404','source_not_found',404),('session404','session_not_found',404),('422','validation_error',422),('429','capacity_exceeded',429),('503','no_active_release',503),('504','turn_timeout',504)]:
    pack[name] = {'error': {'code':code,'message':'Request cannot be completed.'},'request_id':'req-'+name}
    statuses[name]=status
for name, value in pack.items():
    model = Turn if name in ('answer','clarify','handoff','unavailable') else Source if name.startswith('source') and name!='source404' else Error
    model.model_validate(value)
Path('fixtures/api-v1.json').write_text(json.dumps(pack,ensure_ascii=False,indent=2)+'\n')
Path('fixtures/http-statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
Path('fixtures/openapi.json').write_text(json.dumps(app.openapi(),indent=2)+'\n')
