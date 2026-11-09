"""Actual localhost fail-closed smoke test; never activates a corpus."""
import json
import socket
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from app.knowledge import ROOT

with socket.socket() as probe:
    probe.bind(('127.0.0.1',0))
    port=probe.getsockname()[1]
process=subprocess.Popen([str(ROOT.parent/'.venv/bin/python'),'-m','uvicorn','app.main:app','--host','127.0.0.1','--port',str(port),'--workers','1'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=dict(os.environ,SUPPORT_RAG_PORT=str(port)))

def request(path,body=None,headers=None):
    req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json',**(headers or {})})
    try:
        with urllib.request.urlopen(req,timeout=5) as response:return response.status,response.read()
    except urllib.error.HTTPError as exc:return exc.code,exc.read()

try:
    for _ in range(100):
        try:request('/api/v1/health');break
        except (OSError,TimeoutError):time.sleep(.05)
    else:raise RuntimeError('Local server did not start')
    results=[]
    status,raw=request('/api/v1/sessions',{});session=json.loads(raw);assert status==201
    results.append({'case':'create_session','status':status})
    status,raw=request('/api/v1/sessions/'+session['session_id']+'/turns',{'client_turn_id':'smoke','message':'配對','product_id':'hp-a'})
    assert status==503 and json.loads(raw)['error']['code']=='no_active_release'
    results.append({'case':'unapproved_runtime_fail_closed','status':status})
    for path,body,headers,expected in [('/api/v1/sessions',{'role':'system'},None,422),('/api/v1/sessions',{}, {'Origin':'https://evil.invalid'},400),('/api/v1/sources/'+'0'*128,None,None,404),('/api/v1/missing',None,None,404),('/api/v1/sessions',{'extra':'x'*17000},None,413)]:
        status,raw=request(path,body,headers);value=json.loads(raw);assert status==expected and value['request_id'];results.append({'case':path,'status':status})
    status,raw=request('/');assert status==200 and b'<html' in raw.lower();results.append({'case':'same_origin_home','status':status})
    Path('eval/http-smoke.json').write_text(json.dumps({'scope':'actual localhost HTTP; active remains null; no generated answer claim','results':results},indent=2)+'\n')
    print(json.dumps(results))
finally:
    process.terminate()
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
