"""Candidate-only knowledge storage. No activation capability is provided."""
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1] / 'knowledge'
FREEZE_MANIFEST=Path(__file__).resolve().parents[1]/'eval'/'frozen-v4'/'manifest.json'


def digest(value):
    return hashlib.sha256(value).hexdigest()


class Document(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    doc_id: str
    revision: str
    title: str
    product_ids: list[Literal['hp-a', 'hp-b']]
    locale: Literal['zh-TW']
    policy_scope: str
    effective_from: str
    effective_to: Optional[str]
    status: Literal['draft', 'candidate']
    content_sha256: str
    approved_by: Optional[str]
    approved_at: Optional[str]
    supersedes: Optional[str]
    filename: str


def load():
    from datetime import date
    docs = []
    seen = set()
    manifest_path=ROOT/'manifest.json'
    if manifest_path.is_symlink() or manifest_path.stat().st_size>1048576 or (ROOT/'documents').is_symlink():
        raise ValueError('Invalid corpus boundary')
    values=json.loads(manifest_path.read_text())
    if not isinstance(values,list) or len(values)>1000:
        raise ValueError('Document limit exceeded')
    for value in values:
        d = Document.model_validate(value)
        if not re.fullmatch(r'[a-z0-9-]+\.md', d.filename):
            raise ValueError('Invalid document filename')
        path = ROOT / 'documents' / d.filename
        if path.is_symlink() or path.stat().st_size>65536:
            raise ValueError('Symlinks are forbidden')
        body = path.read_text()
        if digest(body.encode()) != d.content_sha256:
            raise ValueError('Content hash mismatch')
        key = (d.doc_id, d.revision)
        if key in seen:
            raise ValueError('Duplicate document revision')
        seen.add(key)
        start = date.fromisoformat(d.effective_from)
        if start.isoformat() != d.effective_from or (d.effective_to and date.fromisoformat(d.effective_to).isoformat() != d.effective_to):
            raise ValueError('Noncanonical date')
        if d.product_ids != sorted(set(d.product_ids)):
            raise ValueError('Noncanonical product set')
        if d.effective_to and date.fromisoformat(d.effective_to) <= start:
            raise ValueError('Invalid effective interval')
        docs.append((d, body))
    for i, (a, _) in enumerate(docs):
        for b, _ in docs[i + 1:]:
            if a.locale == b.locale and a.policy_scope == b.policy_scope and set(a.product_ids or ['hp-a','hp-b']) & set(b.product_ids or ['hp-a','hp-b']):
                if a.effective_from < (b.effective_to or '9999') and b.effective_from < (a.effective_to or '9999'):
                    raise ValueError('Overlapping policy interval')
    return docs


def tokens(text):
    words = re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', text.lower())
    chinese = ''.join(re.findall(r'[\u4e00-\u9fff]', text))
    return ' '.join(words + [chinese[i:i+2] for i in range(len(chinese)-1)])


def connect(path):
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE IF NOT EXISTS releases(id TEXT PRIMARY KEY, manifest TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS chunks(release TEXT, id TEXT, doc TEXT, product TEXT, locale TEXT, start TEXT, end TEXT, section TEXT, text TEXT, PRIMARY KEY(release,id))')
    db.execute('CREATE TABLE IF NOT EXISTS document_snapshots(release TEXT,doc TEXT,revision TEXT,body TEXT,PRIMARY KEY(release,doc,revision))')
    db.execute('CREATE TABLE IF NOT EXISTS provenance(release TEXT,id TEXT,revision TEXT,offset INTEGER,length INTEGER,text_hash TEXT,body_hash TEXT,PRIMARY KEY(release,id))')
    db.execute('CREATE VIRTUAL TABLE IF NOT EXISTS lexical USING fts5(release UNINDEXED, chunk UNINDEXED, terms)')
    return db


def configuration():
    from app.ollama import MODELS, PolicySelection, RESIDENCY
    from app.cloud import SELECTION_SYSTEM,SourceSelection
    base=Path(__file__).resolve().parents[1]
    public_sources=sorted((base/'tests').rglob('*.py'))+[base/name for name in ['scripts/evaluate_candidate.py','scripts/verify_local.py','scripts/scan_public_sources.py','scripts/cloud_setup.py','scripts/cloud_smoke.py','scripts/public_dev10.py','scripts/public_rejection_capture.py','scripts/source_selection_retest.py','requirements.lock','.gitleaks.toml']]
    provider_sources={str(p.relative_to(base)):digest(p.read_bytes()) for p in public_sources}
    return {'verification_source_sha256':provider_sources,'model_residency':{'embedding_keep_alive':'60s','embedding_owned_unload':True,'local_generation':False},'parser':'markdown-offset-v2','chunker':'subsection-blocks-characters-240-v4','retrieval':'capability-scoped-fts5-v4','embedding':{'enabled':False,'model_digest':MODELS['qwen3-embedding:0.6b'],'dimension':1024,'truncate':False,'query_template':'Instruct: Retrieve supporting customer support evidence\nQuery: '},'generation':dict(__import__('app.cloud_config',fromlist=['POLICY']).POLICY,selection_renderer='approved-current-evidence-whole-block-v1',system_sha256=digest(SELECTION_SYSTEM.encode()),schema_sha256=digest(json.dumps(SourceSelection.model_json_schema(),sort_keys=True).encode()),policy_selection_schema_sha256=digest(json.dumps(PolicySelection.model_json_schema(),sort_keys=True).encode())),'evaluation_revision':'public-v4-known','code_sha256':{name:digest((Path(__file__).resolve().parent/name).read_bytes()) for name in ['knowledge.py','cloud_config.py','budget.py','cloud.py','reconciliation.py','providers.py','planning.py','lifecycle.py','service.py','ollama.py','evaluation.py','contracts.py','main.py']},'frozen_split_manifest_sha256':digest(FREEZE_MANIFEST.read_bytes())}


def build(path):
    from app.lifecycle import canonical
    docs=sorted(load(),key=lambda pair:(pair[0].doc_id,pair[0].revision))
    manifest=dict(configuration(),status='candidate-unapproved',source_manifest_sha256=digest((ROOT/'manifest.json').read_bytes()),documents=[d.model_dump() for d,_ in docs if d.status!='draft'])
    chunks=[]
    provenance=[]
    for d,body in docs:
        if d.status=='draft':
            continue
        headings=list(re.finditer(r'(?m)^(#{1,6}) ([^\n]+)\n?',body))
        for index,heading in enumerate(headings):
            if len(heading[1])<2:
                continue
            section=d.title+' / '+heading[2].strip()
            end=headings[index+1].start() if index+1<len(headings) else len(body)
            raw=body[heading.end():end]
            leading=len(raw)-len(raw.lstrip())
            content=raw.strip()
            for offset in range(0,len(content),240):
                text=content[offset:offset+240]
                if not text.strip():continue
                absolute=heading.end()+leading+offset
                text_hash=digest(text.encode())
                cid=digest(canonical([d.doc_id,d.revision,section,absolute,text_hash]))
                chunks.append(('sealed',cid,d.doc_id,json.dumps(d.product_ids),d.locale,d.effective_from,d.effective_to,section,text))
                provenance.append(('sealed',cid,d.revision,absolute,len(text),text_hash,d.content_sha256))
                if len(chunks)>1000:
                    raise ValueError('Chunk limit exceeded')
    chunks.sort(key=lambda r:r[1]);provenance.sort(key=lambda r:r[1])
    manifest['chunks_sha256']=digest(canonical(chunks))
    manifest['provenance_sha256']=digest(canonical(provenance))
    manifest['configuration_sha256']=digest(canonical(configuration()))
    encoded=json.dumps(manifest,sort_keys=True,ensure_ascii=False)
    rid=digest(encoded.encode())
    db=connect(path)
    try:
        with db:
            if db.execute('SELECT 1 FROM releases WHERE id=?',(rid,)).fetchone():
                from app.lifecycle import verify
                verify(db,rid)
                return rid
            db.execute('INSERT INTO releases VALUES(?,?)',(rid,encoded))
            for d,body in docs:
                if d.status!='draft':
                    db.execute('INSERT INTO document_snapshots VALUES(?,?,?,?)',(rid,d.doc_id,d.revision,body))
            for row in chunks:
                db.execute('INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)',tuple([rid]+list(row[1:])))
                db.execute('INSERT INTO lexical VALUES(?,?,?)',(rid,row[1],tokens(row[8])))
            for row in provenance:
                db.execute('INSERT INTO provenance VALUES(?,?,?,?,?,?,?)',tuple([rid]+list(row[1:])))
        return rid
    finally:
        db.close()


def eligible(path,release,product,on_date,locale='zh-TW'):
    from datetime import date
    if product not in (None,'hp-a','hp-b'):
        raise ValueError('Unknown product')
    if date.fromisoformat(on_date).isoformat()!=on_date:
        raise ValueError('Noncanonical date')
    db=connect(path)
    try:
        has_denylist=db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='withdrawn'").fetchone()
        deny=" AND doc NOT IN (SELECT doc FROM withdrawn)" if has_denylist else ''
        rows=db.execute("SELECT id,doc,section,text FROM chunks WHERE release=? AND locale=? AND start<=? AND (end IS NULL OR end>?) AND (product='[]' OR EXISTS (SELECT 1 FROM json_each(product) WHERE value=?))"+deny+" ORDER BY id",(release,locale,on_date,on_date,product)).fetchall()
        if len(rows)>1000:
            raise ValueError('Chunk limit exceeded')
        return rows
    finally:
        db.close()


def search_candidate(path,release,query,product,on_date,locale='zh-TW'):
    """Offline quarantine access; online service first pins an approved release.

    An ephemeral FTS index contains only this eligible release. Other candidates
    cannot alter BM25 statistics or push eligible chunks out of the top ten.
    """
    rows=eligible(path,release,product,on_date,locale)
    terms=tokens(query).split()
    if not rows or not terms:
        return []
    db=sqlite3.connect(':memory:')
    try:
        db.execute('CREATE VIRTUAL TABLE isolated USING fts5(id UNINDEXED,terms)')
        db.executemany('INSERT INTO isolated VALUES(?,?)',[(r[0],tokens(r[2]+' '+r[3])) for r in rows])
        match=' OR '.join('"'+t+'"' for t in terms)
        ids=db.execute('SELECT id FROM isolated WHERE isolated MATCH ? ORDER BY bm25(isolated),id LIMIT 10',(match,)).fetchall()
        mapping={r[0]:r for r in rows}
        return [mapping[r[0]] for r in ids]
    finally:
        db.close()


def seal(path,candidate):
    from app.lifecycle import verify
    db=connect(path)
    try:
        verify(db,candidate)
        return candidate
    finally:
        db.close()
