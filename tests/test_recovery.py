"""Fault injection uses temporary databases and synthetic approval fixtures only."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import pytest
from app.knowledge import build,connect,ROOT,search_candidate
from app.lifecycle import mark_ready,activate,pin,verify
from tests.test_lifecycle import fixture_records


def ready(tmp_path):
    path=tmp_path/'db';rid=build(path);db=connect(path)
    cp,ep,rp=fixture_records(tmp_path,rid,db);mark_ready(path,rid,cp,ep)
    db.close()
    return path,rid,rp


def test_activation_event_failure_atomic_and_reopen(tmp_path):
    path,rid,rp=ready(tmp_path);db=connect(path)
    db.execute("CREATE TRIGGER fail_event BEFORE INSERT ON events BEGIN SELECT RAISE(ABORT,'simulated disk write failure'); END")
    db.commit();db.close()
    with pytest.raises(sqlite3.Error):activate(path,rid,None,rp)
    assert pin(path) is None
    db=connect(path);assert db.execute('SELECT count(*) FROM events').fetchone()[0]==0
    db.execute('DROP TRIGGER fail_event');db.commit();db.close()
    activate(path,rid,None,rp)
    assert pin(path)==rid


def test_two_activation_transactions_one_winner(tmp_path):
    path,rid,rp=ready(tmp_path)
    def attempt(_):
        try:activate(path,rid,None,rp);return 'won'
        except ValueError:return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(attempt,range(2)))==['conflict','won']
    assert pin(path)==rid
    db=connect(path);assert db.execute('SELECT count(*) FROM events').fetchone()[0]==1


def test_provenance_matches_actual_body_and_tamper_fails(tmp_path):
    path,rid,rp=ready(tmp_path);db=connect(path);manifest=verify(db,rid)
    docs={(d['doc_id'],d['revision']):d for d in manifest['documents']}
    for doc,rev,offset,length,text in db.execute('SELECT c.doc,p.revision,p.offset,p.length,c.text FROM chunks c JOIN provenance p ON p.release=c.release AND p.id=c.id WHERE c.release=?',(rid,)):
        body=(ROOT/'documents'/docs[(doc,rev)]['filename']).read_text()
        assert body[offset:offset+length]==text
    db.execute('UPDATE provenance SET offset=offset+1 WHERE release=?',(rid,));db.commit()
    with pytest.raises(ValueError,match='provenance'):verify(db,rid)


def test_fts_other_release_and_ineligible_rows_cannot_shift_ranking(tmp_path):
    path=tmp_path/'db';rid=build(path)
    before=search_candidate(path,rid,'配對','hp-a','2026-11-08');assert before
    db=connect(path)
    for i in range(120):
        db.execute('INSERT INTO lexical VALUES(?,?,?)',('unrelated',str(i),'配對 配對 配對'))
        db.execute('INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)',(rid,'wrong-'+str(i),'other-product','["hp-b"]','zh-TW','2026-01-01',None,'fake','配對 配對 配對'))
    db.commit();db.close()
    assert search_candidate(path,rid,'配對','hp-a','2026-11-08')==before


def test_uncommitted_activation_crash_recovery(tmp_path):
    path,rid,rp=ready(tmp_path)
    db=connect(path);db.execute('BEGIN IMMEDIATE');db.execute('UPDATE pointer SET release=?',(rid,));db.close()
    assert pin(path) is None


def test_successful_rollback_is_atomic_and_audited(tmp_path):
    from app.lifecycle import canonical
    from app.knowledge import digest
    path,first,first_approval=ready(tmp_path)
    activate(path,first,None,first_approval)
    db=connect(path)
    manifest=json.loads(db.execute('SELECT manifest FROM releases WHERE id=?',(first,)).fetchone()[0]);manifest['unit_fixture_release']='second'
    raw=json.dumps(manifest,sort_keys=True,ensure_ascii=False);second=digest(raw.encode())
    with db:
        db.execute('INSERT INTO releases VALUES(?,?)',(second,raw))
        for table in ['chunks','provenance','document_snapshots']:
            rows=db.execute('SELECT * FROM '+table+' WHERE release=?',(first,)).fetchall()
            for row in rows:
                db.execute('INSERT INTO '+table+' VALUES('+','.join('?' for _ in row)+')',tuple([second]+list(row[1:])))
    other=tmp_path/'second';other.mkdir();cp,ep,rp=fixture_records(other,second,db)
    mark_ready(path,second,cp,ep);activate(path,second,first,rp)
    assert pin(path)==second
    activate(path,first,second,first_approval)
    assert pin(path)==first
    assert [row[0] for row in db.execute('SELECT target FROM events ORDER BY sequence')]==[first,second,first]


def test_candidate_disk_write_abort_keeps_previous_snapshot(tmp_path,monkeypatch):
    import app.knowledge as knowledge
    path=tmp_path/'db';first=build(path);db=connect(path)
    db.execute("CREATE TRIGGER fail_chunk BEFORE INSERT ON chunks BEGIN SELECT RAISE(ABORT,'simulated full disk'); END")
    db.commit()
    actual=knowledge.load()
    changed=[(d.model_copy(update={'title':d.title+' fixture'}),body) for d,body in actual]
    monkeypatch.setattr(knowledge,'load',lambda:changed)
    with pytest.raises(sqlite3.Error):build(path)
    assert db.execute('SELECT id FROM releases').fetchall()==[(first,)]
    assert db.execute('SELECT count(*) FROM document_snapshots WHERE release!=?',(first,)).fetchone()[0]==0


def test_process_exit_mid_transaction_recovers_old_pointer(tmp_path):
    import subprocess,sys
    path,rid,rp=ready(tmp_path)
    script='import sqlite3,sys,os; db=sqlite3.connect(sys.argv[1]); db.execute("BEGIN IMMEDIATE"); db.execute("UPDATE pointer SET release=?",(sys.argv[2],)); os._exit(17)'
    result=subprocess.run([sys.executable,'-c',script,str(path),rid],check=False)
    assert result.returncode==17 and pin(path) is None


def test_dangling_active_pointer_is_not_an_approval(tmp_path):
    path=tmp_path/'db';rid=build(path)
    assert pin(path) is None
    db=connect(path);db.execute('UPDATE pointer SET release=?',(rid,));db.commit()
    with pytest.raises(ValueError,match='gate corruption'):pin(path)


def test_withdrawn_rows_are_filtered_before_lexical_top_ten(tmp_path):
    from app.knowledge import search_candidate
    from app.lifecycle import schema,withdraw
    path=tmp_path/'db';rid=build(path);db=connect(path);schema(db)
    before=search_candidate(path,rid,'配對','hp-a','2026-11-08');assert before
    withdraw(path,'hp-a-pairing')
    assert not search_candidate(path,rid,'配對','hp-a','2026-11-08')
    assert search_candidate(path,rid,'配對','hp-b','2026-11-08')
