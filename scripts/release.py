"""Trusted local release operations; approvals must be independently supplied."""
import argparse
import json
from app.knowledge import ROOT,build,seal
from app.lifecycle import mark_ready,activate,pin

p=argparse.ArgumentParser()
p.add_argument('command',choices=['build','ready','activate','rollback','inspect'])
p.add_argument('--release')
p.add_argument('--expected-active')
p.add_argument('--corpus-approval')
p.add_argument('--evaluation')
p.add_argument('--release-approval')
a=p.parse_args()
path=ROOT/'runtime.sqlite3'
try:
    if a.command=='build':
        result={'release_id':seal(path,build(path)),'status':'quarantined'}
    elif a.command=='ready':
        mark_ready(path,a.release,a.corpus_approval,a.evaluation)
        result={'status':'ready','release_id':a.release}
    elif a.command in ('activate','rollback'):
        activate(path,a.release,a.expected_active,a.release_approval)
        result={'status':'active','release_id':a.release}
    else:
        result={'active':pin(path)}
    print(json.dumps(result))
except Exception:
    print(json.dumps({'error':'release_operation_rejected'}))
    raise SystemExit(2)
