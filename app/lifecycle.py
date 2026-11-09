"""Local trusted-record gates; record provenance must be independently reviewed."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from app.knowledge import connect, digest


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def corpus_identity(documents):
    return digest(canonical(sorted(documents, key=lambda d: (d['doc_id'], d['revision']))))


def schema(db):
    db.executescript('''CREATE TABLE IF NOT EXISTS ready(release TEXT PRIMARY KEY, corpus_approval TEXT, evaluation TEXT);
    CREATE TABLE IF NOT EXISTS pointer(singleton INTEGER PRIMARY KEY CHECK(singleton=1), release TEXT);
    INSERT OR IGNORE INTO pointer VALUES(1,NULL);
    CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY, previous TEXT, target TEXT, approval TEXT);
    CREATE TABLE IF NOT EXISTS evaluations(release TEXT PRIMARY KEY,report TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS withdrawn(doc TEXT PRIMARY KEY);
    ''')


def record(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Invalid approval record')
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get('reviewer') != 'Marcus' or not value.get('approved_at'):
        raise ValueError('Missing independent approval')
    datetime.fromisoformat(value['approved_at'])
    return value, digest(raw)


def verify(db, release):
    row = db.execute('SELECT manifest FROM releases WHERE id=?', (release,)).fetchone()
    if not row or digest(row[0].encode()) != release:
        raise ValueError('Invalid release manifest')
    manifest = json.loads(row[0])
    expected = manifest.get('chunks_sha256')
    chunks = db.execute('SELECT * FROM chunks WHERE release=? ORDER BY id', (release,)).fetchall()
    if not chunks or expected != digest(canonical([tuple(['sealed']+list(r[1:])) for r in chunks])):
        raise ValueError('Invalid release chunks')
    provenance=db.execute('SELECT * FROM provenance WHERE release=? ORDER BY id',(release,)).fetchall()
    if manifest.get('provenance_sha256') != digest(canonical([tuple(['sealed']+list(r[1:])) for r in provenance])) or len(provenance)!=len(chunks):
        raise ValueError('Invalid release provenance')
    by_id={row[1]:row for row in provenance}
    identities={(d['doc_id'],d['revision']):d for d in manifest['documents']}
    bodies={(doc,revision):body for doc,revision,body in db.execute('SELECT doc,revision,body FROM document_snapshots WHERE release=?',(release,))}
    if set(bodies)!=set(identities) or any(digest(body.encode())!=identities[key]['content_sha256'] for key,body in bodies.items()):
        raise ValueError('Invalid document snapshot')
    for row in chunks:
        origin=by_id[row[1]]
        doc=identities.get((row[2],origin[2]))
        if not doc or origin[3]<0 or origin[4]!=len(row[8]) or origin[5]!=digest(row[8].encode()) or origin[6]!=doc['content_sha256']:
            raise ValueError('Invalid chunk origin')
        if bodies[(row[2],origin[2])][origin[3]:origin[3]+origin[4]]!=row[8]:
            raise ValueError('Chunk offset does not match snapshot')
        expected_id=digest(canonical([row[2],origin[2],row[7],origin[3],origin[5]]))
        if expected_id!=row[1]:
            raise ValueError('Invalid chunk identity')
    return manifest


def mark_ready(path, release, approval_path, evaluation_path):
    approval, approval_hash = record(approval_path)
    evaluation_raw = Path(evaluation_path).read_bytes()
    evaluation = json.loads(evaluation_raw)
    db = connect(path)
    schema(db)
    try:
        manifest = verify(db, release)
        docs = manifest['documents']
        identities = {d['doc_id']+':'+d['revision']: d['content_sha256'] for d in docs}
        if approval.get('document_hashes') != identities or approval.get('manifest_sha256') != manifest['source_manifest_sha256']:
            raise ValueError('Corpus approval mismatch')
        # Canonical metadata binding is required for newly issued approvals.
        if approval.get('corpus_sha256') != corpus_identity(docs):
            raise ValueError('Canonical corpus approval required')
        if evaluation.get('corpus_approval_sha256')!=approval_hash:
            raise ValueError('Evaluation corpus approval mismatch')
        from app.evaluation import validate_report
        validate_report(db,release,evaluation)
        with db:
            db.execute('INSERT OR REPLACE INTO ready VALUES(?,?,?)', (release, approval_hash, digest(evaluation_raw)))
            db.execute('INSERT OR REPLACE INTO evaluations VALUES(?,?)',(release,evaluation_raw.decode()))
    finally:
        db.close()


def activate(path, target, expected, release_approval_path):
    approval, approval_hash = record(release_approval_path)
    db = connect(path)
    schema(db)
    try:
        db.execute('BEGIN IMMEDIATE')
        manifest = verify(db, target)
        ready = db.execute('SELECT corpus_approval,evaluation FROM ready WHERE release=?', (target,)).fetchone()
        if not ready or approval.get('release_id') != target or approval.get('manifest_sha256') != target or approval.get('corpus_approval_sha256') != ready[0] or approval.get('evaluation_sha256') != ready[1]:
            raise ValueError('Release approval mismatch')
        stored=db.execute('SELECT report FROM evaluations WHERE release=?',(target,)).fetchone()
        from app.evaluation import validate_report
        if not stored or digest(stored[0].encode())!=ready[1]:
            raise ValueError('Stored evaluation mismatch')
        validate_report(db,target,json.loads(stored[0]))
        from app.evaluation import verify_manual_semantics
        verify_manual_semantics(approval,json.loads(stored[0]),target,manifest['configuration_sha256'],ready[1])
        if approval.get('holdout_sha256')!=json.loads(stored[0]).get('holdout_sha256'):
            raise ValueError('Holdout approval mismatch')
        if any(db.execute('SELECT 1 FROM withdrawn WHERE doc=?', (d['doc_id'],)).fetchone() for d in manifest['documents']):
            raise ValueError('Withdrawn document cannot be restored')
        old = db.execute('SELECT release FROM pointer').fetchone()[0]
        if old != expected:
            raise ValueError('Activation conflict')
        db.execute('UPDATE pointer SET release=? WHERE singleton=1', (target,))
        db.execute('INSERT INTO events(previous,target,approval) VALUES(?,?,?)', (old,target,approval_hash))
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def pin(path):
    db = connect(path)
    schema(db)
    try:
        release = db.execute('SELECT release FROM pointer').fetchone()[0]
        if release:
            manifest=verify(db, release)
            from app.knowledge import configuration
            if manifest['configuration_sha256']!=digest(canonical(configuration())):
                raise ValueError('Active configuration mismatch')
            ready=db.execute('SELECT evaluation FROM ready WHERE release=?',(release,)).fetchone()
            evidence=db.execute('SELECT report FROM evaluations WHERE release=?',(release,)).fetchone()
            event=db.execute('SELECT 1 FROM events WHERE target=?',(release,)).fetchone()
            if not ready or not evidence or not event or digest(evidence[0].encode())!=ready[0]:
                raise ValueError('Active release gate corruption')
        return release
    finally:
        db.close()


def withdraw(path, doc):
    db=connect(path)
    schema(db)
    with db:
        db.execute('INSERT OR IGNORE INTO withdrawn VALUES(?)',(doc,))
    db.close()
