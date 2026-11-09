"""Explicit local knowledge lifecycle. No browser management endpoints exist."""
import argparse
import asyncio
import json
from types import SimpleNamespace
from app.knowledge import ROOT,build,connect,load
from app.lifecycle import activate,pin,mark_ready

parser=argparse.ArgumentParser()
parser.add_argument('command',choices=['validate','build','evaluate','ready','activate','rollback','inspect'])
parser.add_argument('--candidate')
parser.add_argument('--exclusive-model-slot',action='store_true')
parser.add_argument('--suite',choices=['dev','regression'])
parser.add_argument('--approval-record')
parser.add_argument('--evaluation')
parser.add_argument('--release-approval')
parser.add_argument('--expected-active')
a=parser.parse_args();path=ROOT/'runtime.sqlite3'
try:
    if a.command=='validate':
        result={'status':'validated-candidate','documents':len(load()),'approved':False}
    elif a.command=='build':
        result={'release_id':build(path),'status':'quarantined'}
    elif a.command=='evaluate':
        if not a.suite or not a.approval_record:raise ValueError('Evaluation arguments required')
        from scripts.evaluate_candidate import run
        passed=asyncio.run(run(SimpleNamespace(split=a.suite,corpus_approval=a.approval_record,candidate=a.candidate,exclusive_model_slot=a.exclusive_model_slot)))
        raise SystemExit(0 if passed else 3)
    elif a.command=='ready':
        if not a.candidate or not a.approval_record or not a.evaluation:raise ValueError('Ready arguments required')
        mark_ready(path,a.candidate,a.approval_record,a.evaluation);result={'status':'ready','release_id':a.candidate}
    elif a.command in ('activate','rollback'):
        if not a.candidate or not a.release_approval:raise ValueError('Activation arguments required')
        activate(path,a.candidate,a.expected_active,a.release_approval);result={'status':'active','release_id':a.candidate}
    else:
        db=connect(path)
        result={'active':pin(path),'releases':[{'id':r[0],'manifest':json.loads(r[1])} for r in db.execute('SELECT * FROM releases')]}
        db.close()
    print(json.dumps(result,ensure_ascii=False))
except ValueError as exc:
    conflict='conflict' in str(exc).lower()
    print(json.dumps({'error':'activation_conflict' if conflict else 'validation_failed'}))
    raise SystemExit(4 if conflict else 2)
except (OSError,__import__('sqlite3').Error):
    print(json.dumps({'error':'storage_unavailable'}));raise SystemExit(5)
