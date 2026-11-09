"""Single-worker localhost-only entrypoint with matching Origin allowlist."""
import argparse
import os
import uvicorn
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8000);a=p.parse_args()
if not 1024<=a.port<=65535:raise SystemExit('Invalid local port')
os.environ['SUPPORT_RAG_PORT']=str(a.port)
uvicorn.run('app.main:app',host='127.0.0.1',port=a.port,workers=1)
