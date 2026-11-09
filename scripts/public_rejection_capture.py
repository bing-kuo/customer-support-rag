"""Opt-in restricted synthetic-public rejection evidence; never ordinary logs."""
import json,os,re,hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'runtime'/'public-synthetic-eval-rejections'
SENSITIVE=re.compile(r'/Users/|/home/|/etc/|file://|BEGIN .*PRIVATE KEY|Authorization\s*:|OPENAI_API_KEY|\bsk-[A-Za-z0-9_-]{12,}')


class PublicSyntheticCapture:
    def __init__(self,manifest,root=None):
        if manifest.get('data_classification')!='public_synthetic':raise ValueError('Public synthetic manifest required')
        self.ids={q['id'] for q in manifest['cases']};self.manifest_hash=hashlib.sha256(json.dumps(manifest,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest();self.root=Path(root) if root is not None else ROOT;self.case=None
    def select_case(self,case):
        if case not in self.ids or not re.fullmatch('[A-Za-z0-9_-]{1,100}',case):raise ValueError('Unapproved public case')
        self.case=case
    def __call__(self,request_id,code,raw,items,aliases,key):
        if self.case not in self.ids or not re.fullmatch('[A-Za-z0-9_-]{1,128}',request_id):return {'saved':False,'reason':'unapproved_context'}
        from app.budget import CODES
        safe_code=code if code in CODES else 'unclassified_failure'
        pieces=[]
        if isinstance(raw,dict) and isinstance(raw.get('output'),list):
            for item in raw['output']:
                if isinstance(item,dict) and isinstance(item.get('content'),list):
                    for content in item['content']:
                        if isinstance(content,dict):
                            for field in ('text','refusal'):
                                if isinstance(content.get(field),str):pieces.append(content[field])
        text='\n'.join(pieces)
        evidence=[{'alias':i['source_id'],'source_id':aliases.get(i['source_id']),'text':i['text'],'required_sentences':i['required_sentences']} for i in items]
        encoded=json.dumps({'output':text,'evidence':evidence},ensure_ascii=False)
        suppress=len(text.encode())>2048 or len(encoded.encode())>4096 or bool(key and key in encoded) or bool(SENSITIVE.search(encoded))
        value={'manifest_sha256':self.manifest_hash,'case_id':self.case,'local_request_id':request_id,'rejection_code':safe_code,'model_output_kind':'source_selection','provider_output_validated':False,'raw_capture_suppressed':suppress}
        if not suppress:value.update(rejected_output_text=text,evidence=evidence)
        data=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
        if len(data)>4096:
            value.pop('rejected_output_text',None);value.pop('evidence',None);value['raw_capture_suppressed']=True;suppress=True
            data=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
        if self.root.is_symlink():return {'saved':False,'reason':'unsafe_capture_directory'}
        self.root.mkdir(mode=0o700,parents=True,exist_ok=True)
        if self.root.stat().st_mode & 0o077:return {'saved':False,'reason':'unsafe_capture_directory'}
        target=self.root/(request_id+'.json')
        try:fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        except FileExistsError:return {'saved':False,'reason':'existing_capture_preserved'}
        with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        return {'saved':True,'raw_suppressed':suppress,'restricted':True}
