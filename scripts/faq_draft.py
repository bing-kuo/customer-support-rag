"""Create a local non-indexed draft from an explicitly synthetic conversation."""
import argparse
import json
from pathlib import Path
from uuid import uuid4
p=argparse.ArgumentParser();p.add_argument('--question',required=True);p.add_argument('--proposed-answer',required=True);a=p.parse_args()
if len(a.question)>2000 or len(a.proposed_answer)>4000:raise SystemExit('Draft exceeds budget')
folder=Path('eval/drafts');folder.mkdir(exist_ok=True)
ident=uuid4().hex
(folder/(ident+'.json')).write_text(json.dumps({'id':ident,'status':'draft','synthetic':True,'question':a.question,'proposed_answer':a.proposed_answer,'approved_by':None},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'draft_id':ident,'status':'draft','indexed':False}))
