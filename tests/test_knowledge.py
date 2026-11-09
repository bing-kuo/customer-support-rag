import json
import pytest
from app.knowledge import build, connect, load, search_candidate, ROOT
from app.contracts import Turn, Source


def test_candidate_build_dedup_and_product_filter(tmp_path):
    path = tmp_path/'test.sqlite3'
    rid = build(path)
    assert build(path) == rid
    db = connect(path)
    assert db.execute('SELECT count(*) FROM releases').fetchone()[0] == 1
    assert len(load()) == 20
    assert all('hp-b' not in row[1] for row in search_candidate(path,rid,'充電','hp-a','2026-11-08'))
    assert all(row[1].startswith('general') for row in search_candidate(path,rid,'退款',None,'2026-11-08'))
    assert db.execute('SELECT manifest FROM releases').fetchone()[0].find('candidate-unapproved') >= 0
    db.close()


def test_hash_tamper_fails_without_new_release(tmp_path, monkeypatch):
    import app.knowledge as k
    root = tmp_path/'knowledge'
    import shutil
    shutil.copytree(ROOT/'documents',root/'documents')
    shutil.copy(ROOT/'manifest.json',root/'manifest.json')
    monkeypatch.setattr(k,'ROOT',root)
    path=tmp_path/'db'
    rid=build(path)
    next((root/'documents').glob('*.md')).write_text('modified')
    with pytest.raises(ValueError,match='hash'):
        build(path)
    db=connect(path)
    assert db.execute('SELECT id FROM releases').fetchall()==[(rid,)]


def test_fixture_contracts_and_inconsistent_model_output():
    pack=json.loads((ROOT.parent/'fixtures/api-v1.json').read_text())
    for name in ['answer','clarify','handoff','unavailable']:
        Turn.model_validate(pack[name])
    with pytest.raises(ValueError):
        Turn.model_validate(dict(pack['answer'],citations=[]))
    with pytest.raises(ValueError):
        Source.model_validate(dict(pack['source-withdrawn'],text='secret'))


@pytest.mark.parametrize('products',[['hp-a','hp-b'],[],['hp-a']])
def test_overlapping_policy_product_intersection(tmp_path,monkeypatch,products):
    import shutil
    import app.knowledge as k
    root=tmp_path/'knowledge'
    shutil.copytree(ROOT/'documents',root/'documents')
    values=json.loads((ROOT/'manifest.json').read_text())
    other=dict(values[0],doc_id='conflicting',product_ids=products)
    values.append(other)
    (root/'manifest.json').write_text(json.dumps(values))
    monkeypatch.setattr(k,'ROOT',root)
    with pytest.raises(ValueError,match='Overlapping'):
        k.load()


def test_draft_revision_is_not_in_snapshot_or_index(tmp_path,monkeypatch):
    import app.knowledge as k
    docs=k.load();draft=docs[0][0].model_copy(update={'status':'draft'})
    monkeypatch.setattr(k,'load',lambda:[(draft,docs[0][1])]+docs[1:])
    path=tmp_path/'draft-db';rid=k.build(path);db=k.connect(path)
    assert db.execute('SELECT count(*) FROM chunks WHERE release=? AND doc=?',(rid,draft.doc_id)).fetchone()[0]==0
    assert db.execute('SELECT count(*) FROM document_snapshots WHERE release=? AND doc=?',(rid,draft.doc_id)).fetchone()[0]==0
    manifest=json.loads(db.execute('SELECT manifest FROM releases WHERE id=?',(rid,)).fetchone()[0])
    assert all(d['doc_id']!=draft.doc_id for d in manifest['documents'])
