"""Offline tests for SQLite embedding storage."""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from customer_support_rag.knowledge import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EmbeddedChunk,
    EmbeddingStore,
    KnowledgeDocument,
    KnowledgeMetadata,
    OllamaEmbeddingAdapter,
    StorageConflictError,
    StorageError,
    chunk_document,
    document_embedding_key,
    read_document,
)
from customer_support_rag.knowledge.storage import (
    CHUNKING_PROFILE,
    FORMAT_VERSION,
    INPUT_PROFILE,
)

REPO = Path(__file__).parents[1]
KNOWLEDGE = REPO / "knowledge"
MODEL_DIGEST = "ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d"
OTHER_DIGEST = "b" * 64
HP_A_SHA256 = "223243a3ba9c40840746b1b2ee66e40b970a8fefd1324bb586930a381f9f4e59"
HP_A_ARTIFACT = "9a2283a73584261d31460d1b46855145d384c756834c8984c1539e96647cb4ad"
HP_A_REVISED_BODY_ARTIFACT = (
    "e87a5d1a3cf5ec1a49129b18d5dd16b0deece91393d52ffdaf2b88ca9d6d4338"
)
HP_A_REVISED_TITLE_ARTIFACT = (
    "3c8e61013408d7fdbf2ee6154545f87eed646c44a1777f50d78225f3fd6c04d5"
)
HP_B_ARTIFACT = "18812b328a8e3614c4c83a849c2bfcd108e695453c8eaf9d57cf87200cefafae"
HP_A_OTHER_DIGEST_ARTIFACT = (
    "8de241d30a4b524e9556d8ce6bfcb6af7d3ed2db60d44140c315983b6a01c17c"
)
MARKER = "UNIQUE_BODY_MARKER_9f3c"


def mock_vector(scale: float = 1.0) -> tuple[float, ...]:
    """Return a full 1024-dimension synthetic vector.

    Every component is set. This is not a live vector and it is not the
    recorded live prefix padded with zeros.
    """
    return tuple(
        (scale + index) / EMBEDDING_DIMENSIONS for index in range(EMBEDDING_DIMENSIONS)
    )


def metadata(
    document_id: str = "sample",
    product_id: str = "hp-a",
    title: str = "Sample",
    version: str = "1",
) -> KnowledgeMetadata:
    return KnowledgeMetadata(
        document_id=document_id,
        product_id=product_id,
        title=title,
        version=version,
        synthetic=True,
    )


def document(body: str, **kwargs: str) -> KnowledgeDocument:
    return KnowledgeDocument(metadata=metadata(**kwargs), body=body)


def embed_eligible(chunks: list, scale: float = 1.0) -> list[EmbeddedChunk]:
    return [
        EmbeddedChunk(chunk=chunk, vector=mock_vector(scale))
        for chunk in chunks
        if chunk.text.strip()
    ]


def rows(path: Path) -> list[tuple[str, str]]:
    connection = sqlite3.connect(path)
    try:
        return list(
            connection.execute(
                "SELECT artifact_key, payload_json FROM embedding_documents "
                "ORDER BY artifact_key"
            )
        )
    finally:
        connection.close()


def test_hp_a_roundtrip_survives_a_new_process(tmp_path: Path) -> None:
    source = read_document(KNOWLEDGE, "hp-a/headphone-care.md")
    chunks = chunk_document(source)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"model": EMBEDDING_MODEL, "embeddings": [list(mock_vector())]},
        )

    with OllamaEmbeddingAdapter(httpx.MockTransport(handler)) as adapter:
        embedded = adapter.embed_chunks(chunks)

    assert len(requests) == 1
    assert json.loads(requests[0].content)["input"] == [chunks[1].text]
    assert len(chunks) == 2
    assert chunks[0].text == "\n"
    assert chunks[1].index == 1
    assert chunks[1].body_sha256 == HP_A_SHA256
    assert "日常保養" in chunks[1].text
    assert embedded[0].vector[0] == 1 / EMBEDDING_DIMENSIONS

    path = tmp_path / "embeddings.sqlite"
    with EmbeddingStore(path) as store:
        key = store.put_document(source, embedded, model_digest=MODEL_DIGEST)
        assert key == HP_A_ARTIFACT
        assert key == document_embedding_key(source, model_digest=MODEL_DIGEST)
        assert store.get_document("0" * 64) is None

    stored = rows(path)
    assert len(stored) == 1
    payload_json = stored[0][1]
    assert payload_json == json.dumps(
        json.loads(payload_json),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert "日常保養" in payload_json
    assert "\\u" not in payload_json
    parsed = json.loads(payload_json)
    assert parsed["source"]["body"] == source.body
    assert parsed["chunks"][0]["vector"] is None
    assert len(parsed["chunks"][1]["vector"]) == EMBEDDING_DIMENSIONS
    assert parsed["chunks"][1]["vector"][1] == (1 + 1) / EMBEDDING_DIMENSIONS

    script = tmp_path / "read_back.py"
    script.write_text(
        """
import json
import socket
import sys

from customer_support_rag.knowledge.storage import EmbeddingStore

def blocked(*args, **kwargs):
    raise RuntimeError("network blocked")

socket.socket = blocked
socket.create_connection = blocked

database, key = sys.argv[1], sys.argv[2]
with EmbeddingStore(database) as store:
    missing = store.get_document("0" * 64)
    found = store.get_document(key)
if missing is not None or found is None:
    raise SystemExit("unexpected read result")
json.dump(
    {
        "body": found.document.body,
        "title": found.document.metadata.title,
        "chunk_count": len(found.chunks),
        "embedded_indexes": [item.chunk.index for item in found.embedded_chunks],
        "texts": [chunk.text for chunk in found.chunks],
        "headings": [list(chunk.heading_path) for chunk in found.chunks],
        "vector": list(found.embedded_chunks[0].vector),
    },
    sys.stdout,
    ensure_ascii=False,
)
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), env.get("PYTHONPATH", "")])
    completed = subprocess.run(
        [sys.executable, str(script), str(path), key],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    loaded = json.loads(completed.stdout)
    assert loaded["body"] == source.body
    assert loaded["title"] == "HP-A 耳機日常保養"
    assert loaded["chunk_count"] == 2
    assert loaded["embedded_indexes"] == [1]
    assert loaded["texts"] == [chunk.text for chunk in chunks]
    assert loaded["headings"] == [[], ["日常保養"]]
    assert tuple(loaded["vector"]) == embedded[0].vector


def test_new_database_has_one_versioned_table(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.sqlite"
    with EmbeddingStore(path) as store:
        timeout = store._connection.execute("PRAGMA busy_timeout").fetchone()
    assert timeout is not None
    assert timeout[0] == 5000
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        tables = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        assert tables == [("embedding_documents",)]
    finally:
        connection.close()


def test_identical_put_keeps_a_single_row(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.sqlite"
    source = document("alpha\n")
    chunks = chunk_document(source)
    embedded = embed_eligible(chunks)
    with EmbeddingStore(path) as store:
        first = store.put_document(source, embedded, model_digest=MODEL_DIGEST)
        second = store.put_document(source, embedded, model_digest=MODEL_DIGEST)
        assert first == second
        assert len(store.get_document(first).embedded_chunks) == 1  # type: ignore[union-attr]
    assert len(rows(path)) == 1


def test_different_vector_keeps_the_stored_row(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.sqlite"
    source = document("alpha\n")
    chunks = chunk_document(source)
    with EmbeddingStore(path) as store:
        key = store.put_document(
            source, embed_eligible(chunks, 1.0), model_digest=MODEL_DIGEST
        )
        before = rows(path)
        with pytest.raises(StorageConflictError, match="conflicts") as caught:
            store.put_document(
                source, embed_eligible(chunks, 2.0), model_digest=MODEL_DIGEST
            )
        assert source.body not in str(caught.value)
        assert rows(path) == before
        loaded = store.get_document(key)
    assert loaded is not None
    assert loaded.embedded_chunks[0].vector == mock_vector(1.0)


def test_identity_changes_keep_the_previous_document(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.sqlite"
    original = read_document(KNOWLEDGE, "hp-a/headphone-care.md")
    revised_body = original.model_copy(update={"body": original.body + "補充。\n"})
    revised_title = original.model_copy(
        update={
            "metadata": original.metadata.model_copy(update={"title": "HP-A 保養改版"})
        }
    )
    other = read_document(KNOWLEDGE, "hp-b/connection-troubleshooting.md")
    with EmbeddingStore(path) as store:
        original_key = store.put_document(
            original,
            embed_eligible(chunk_document(original)),
            model_digest=MODEL_DIGEST,
        )
        body_key = store.put_document(
            revised_body,
            embed_eligible(chunk_document(revised_body), 2.0),
            model_digest=MODEL_DIGEST,
        )
        title_key = store.put_document(
            revised_title,
            embed_eligible(chunk_document(revised_title), 3.0),
            model_digest=MODEL_DIGEST,
        )
        other_key = store.put_document(
            other, embed_eligible(chunk_document(other), 4.0), model_digest=MODEL_DIGEST
        )
        digest_key = store.put_document(
            original,
            embed_eligible(chunk_document(original), 5.0),
            model_digest=OTHER_DIGEST,
        )
        assert original_key == HP_A_ARTIFACT
        assert body_key == HP_A_REVISED_BODY_ARTIFACT
        assert title_key == HP_A_REVISED_TITLE_ARTIFACT
        assert other_key == HP_B_ARTIFACT
        assert digest_key == HP_A_OTHER_DIGEST_ARTIFACT
        assert len(rows(path)) == 5
        reread = store.get_document(original_key)
    assert reread is not None
    assert reread.document.body == original.body
    assert reread.document.metadata.title == original.metadata.title
    assert reread.embedded_chunks[0].vector == mock_vector(1.0)
    assert document_embedding_key(original, model_digest=OTHER_DIGEST) == digest_key


@pytest.mark.parametrize(
    "mutate",
    ["count", "index", "digest", "offsets", "order"],
)
def test_source_mismatch_writes_nothing(tmp_path: Path, mutate: str) -> None:
    path = tmp_path / "embeddings.sqlite"
    source = document("# 一\nalpha\n# 二\nbeta\n", document_id="two-parts")
    chunks = chunk_document(source)
    embedded = embed_eligible(chunks)
    assert len(embedded) >= 2
    if mutate == "count":
        rejected = embedded[:-1]
    elif mutate == "index":
        moved = embedded[0].chunk.model_copy(update={"index": 9})
        rejected = [embedded[0].model_copy(update={"chunk": moved}), *embedded[1:]]
    elif mutate == "digest":
        changed = embedded[0].chunk.model_copy(update={"body_sha256": "a" * 64})
        rejected = [embedded[0].model_copy(update={"chunk": changed}), *embedded[1:]]
    elif mutate == "offsets":
        shifted = embedded[0].chunk.model_copy(update={"char_start": 1})
        rejected = [embedded[0].model_copy(update={"chunk": shifted}), *embedded[1:]]
    else:
        rejected = list(reversed(embedded))

    seed = document("seed\n", document_id="seed")
    with EmbeddingStore(path) as store:
        store.put_document(
            seed, embed_eligible(chunk_document(seed)), model_digest=MODEL_DIGEST
        )
        before = rows(path)
        with pytest.raises(ValueError, match="eligible chunks"):
            store.put_document(source, rejected, model_digest=MODEL_DIGEST)
        assert rows(path) == before


@pytest.mark.parametrize("label", ["dimension", "nan", "bool", "overflow", "zero"])
def test_illegal_vector_writes_nothing(tmp_path: Path, label: str) -> None:
    path = tmp_path / "embeddings.sqlite"
    source = document("alpha\n", document_id=label)
    chunks = chunk_document(source)
    embedded = embed_eligible(chunks)
    filler = (0.5,) * (EMBEDDING_DIMENSIONS - 1)
    if label == "dimension":
        vector: object = (0.5,) * (EMBEDDING_DIMENSIONS - 1)
    elif label == "zero":
        vector = (0.0,) * EMBEDDING_DIMENSIONS
    elif label == "nan":
        vector = (float("nan"),) + filler
    elif label == "bool":
        vector = (True,) + filler
    else:
        vector = (10**10000,) + filler
    rejected = [EmbeddedChunk.model_construct(chunk=embedded[0].chunk, vector=vector)]
    with EmbeddingStore(path) as store:
        store.put_document(
            document("seed\n", document_id="seed"),
            embed_eligible(chunk_document(document("seed\n", document_id="seed"))),
            model_digest=MODEL_DIGEST,
        )
        before = rows(path)
        with pytest.raises(ValueError, match="1024 finite non-zero"):
            store.put_document(source, rejected, model_digest=MODEL_DIGEST)
        assert rows(path) == before


def test_blank_document_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.sqlite"
    with EmbeddingStore(path) as store:
        for body in ("", " \n\t"):
            with pytest.raises(ValueError, match="non-blank"):
                store.put_document(document(body), [], model_digest=MODEL_DIGEST)
    assert rows(path) == []


def test_insert_abort_keeps_the_existing_row(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.sqlite"
    source = document("alpha\n")
    other = document("beta\n", document_id="other")
    with EmbeddingStore(path) as store:
        key = store.put_document(
            source, embed_eligible(chunk_document(source)), model_digest=MODEL_DIGEST
        )
        with sqlite3.connect(path) as connection:
            connection.execute(
                """
                CREATE TRIGGER abort_embedding_insert
                BEFORE INSERT ON embedding_documents
                BEGIN
                    SELECT RAISE(ABORT, 'abort insert');
                END
                """
            )
        before = rows(path)
        with pytest.raises(StorageError, match="write failed") as caught:
            store.put_document(
                other,
                embed_eligible(chunk_document(other), 2.0),
                model_digest=MODEL_DIGEST,
            )
        assert not isinstance(caught.value, StorageConflictError)
        assert MARKER not in str(caught.value)
        assert rows(path) == before
        loaded = store.get_document(key)
        assert loaded is not None
        assert loaded.embedded_chunks[0].vector == mock_vector()
    # The abort trigger is test-only schema. Remove it before a new store opens.
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER abort_embedding_insert")
    with EmbeddingStore(path) as reopened:
        reread = reopened.get_document(key)
    assert reread is not None
    assert reread.document.body == source.body


@pytest.mark.parametrize("version", [0, 2])
def test_unknown_schema_is_left_unchanged(tmp_path: Path, version: int) -> None:
    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE legacy (id INTEGER)")
        connection.execute(f"PRAGMA user_version = {version}")
    with pytest.raises(StorageError, match="Unsupported database schema"):
        EmbeddingStore(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (version,)
        names = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    assert names == [("legacy",)]


@pytest.mark.parametrize(
    "payload",
    ["{", json.dumps({"format_version": 1})],
)
def test_corrupt_payload_is_not_repaired(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "embeddings.sqlite"
    source = document("alpha\n")
    with EmbeddingStore(path) as store:
        key = store.put_document(
            source, embed_eligible(chunk_document(source)), model_digest=MODEL_DIGEST
        )
        with sqlite3.connect(path) as connection:
            connection.execute(
                "UPDATE embedding_documents SET payload_json = ? "
                "WHERE artifact_key = ?",
                (payload, key),
            )
        with pytest.raises(StorageError, match="Invalid stored embedding payload"):
            store.get_document(key)
        assert rows(path) == [(key, payload)]
        with pytest.raises(StorageError, match="Invalid stored embedding payload"):
            store.get_document(key)
        assert rows(path) == [(key, payload)]


def test_payload_key_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.sqlite"
    source = document("alpha\n")
    with EmbeddingStore(path) as store:
        key = store.put_document(
            source, embed_eligible(chunk_document(source)), model_digest=MODEL_DIGEST
        )
        payload = rows(path)[0][1]
        other_key = "ab" * 32
        with sqlite3.connect(path) as connection:
            connection.execute(
                "INSERT INTO embedding_documents (artifact_key, payload_json) "
                "VALUES (?, ?)",
                (other_key, payload),
            )
        with pytest.raises(StorageError, match="Invalid stored embedding payload"):
            store.get_document(other_key)
        loaded = store.get_document(key)
    assert loaded is not None
    assert loaded.document.body == source.body
    assert rows(path) == sorted([(key, payload), (other_key, payload)])


def test_hostile_text_stays_data(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.sqlite"
    body = (
        f"{MARKER} '; DROP TABLE embedding_documents; --\nhttps://evil.example/steal\n"
    )
    source = document(body, title="合成 SQL")
    with EmbeddingStore(path) as store:
        key = store.put_document(
            source, embed_eligible(chunk_document(source)), model_digest=MODEL_DIGEST
        )
        loaded = store.get_document(key)
        with pytest.raises(StorageConflictError) as caught:
            store.put_document(
                source,
                embed_eligible(chunk_document(source), 2.0),
                model_digest=MODEL_DIGEST,
            )
    assert loaded is not None
    assert loaded.document.body == body
    assert MARKER not in str(caught.value)
    assert "evil.example" not in str(caught.value)
    with sqlite3.connect(path) as connection:
        names = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    assert names == [("embedding_documents",)]


@pytest.mark.parametrize("max_chars", [True, False, 1, 1201])
def test_max_chars_must_be_inside_the_storage_limit(max_chars: object) -> None:
    source = document("alpha\n")
    with pytest.raises(ValueError, match="max_chars"):
        document_embedding_key(source, model_digest=MODEL_DIGEST, max_chars=max_chars)


def test_uppercase_digest_is_rejected() -> None:
    source = document("alpha\n")
    with pytest.raises(ValueError, match="model digest"):
        document_embedding_key(source, model_digest=MODEL_DIGEST.upper())


def test_missing_parent_directory_creates_nothing(tmp_path: Path) -> None:
    path = tmp_path / "missing" / "embeddings.sqlite"
    with pytest.raises(ValueError, match="parent directory"):
        EmbeddingStore(path)
    assert not path.parent.exists()


def test_readonly_open_does_not_create_or_rewrite(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(StorageError, match="could not open"):
        EmbeddingStore.open_readonly(missing)
    assert not missing.exists()

    path = tmp_path / "embeddings.sqlite"
    source = read_document(KNOWLEDGE, "hp-a/headphone-care.md")
    with EmbeddingStore(path) as store:
        key = store.put_document(
            source,
            embed_eligible(chunk_document(source)),
            model_digest=MODEL_DIGEST,
        )
    before = path.read_bytes()
    names = sorted(item.name for item in tmp_path.iterdir())
    with EmbeddingStore.open_readonly(path) as store:
        loaded = store.get_document(key)
        with pytest.raises(StorageError):
            store.put_document(
                source,
                embed_eligible(chunk_document(source), 2.0),
                model_digest=MODEL_DIGEST,
            )
    assert loaded is not None
    assert loaded.artifact_key == HP_A_ARTIFACT
    assert path.read_bytes() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == names


def test_closed_store_cannot_read(tmp_path: Path) -> None:
    store = EmbeddingStore(tmp_path / "embeddings.sqlite")
    store.close()
    with pytest.raises(StorageError, match="read failed"):
        store.get_document("0" * 64)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _identity_key(source: KnowledgeDocument, body_hash: str) -> str:
    identity = {
        "format_version": FORMAT_VERSION,
        "metadata": {
            "document_id": source.metadata.document_id,
            "product_id": source.metadata.product_id,
            "title": source.metadata.title,
            "version": source.metadata.version,
            "synthetic": source.metadata.synthetic,
        },
        "body_sha256": body_hash,
        "embedding": {
            "model": EMBEDDING_MODEL,
            "model_digest": MODEL_DIGEST,
            "dimensions": EMBEDDING_DIMENSIONS,
            "input_profile": INPUT_PROFILE,
        },
        "chunking": {"profile": CHUNKING_PROFILE, "max_chars": 1200},
    }
    return hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()


def _replace_row(
    path: Path, key: str, payload: str, new_key: str | None = None
) -> None:
    with sqlite3.connect(path) as connection:
        if new_key is None:
            connection.execute(
                """
                UPDATE embedding_documents
                SET payload_json = ?
                WHERE artifact_key = ?
                """,
                (payload, key),
            )
            return
        connection.execute(
            """
            UPDATE embedding_documents
            SET artifact_key = ?, payload_json = ?
            WHERE artifact_key = ?
            """,
            (new_key, payload, key),
        )


@pytest.mark.parametrize("case", ["deep_json", "huge_integer", "lone_surrogate"])
def test_decode_boundary_is_storage_error(tmp_path: Path, case: str) -> None:
    path = tmp_path / "embeddings.sqlite"
    source = document("alpha\n")
    with EmbeddingStore(path) as store:
        key = store.put_document(
            source, embed_eligible(chunk_document(source)), model_digest=MODEL_DIGEST
        )
        payload = rows(path)[0][1]
        if case == "deep_json":
            damaged = "[" * 12000 + "0" + "]" * 12000
        elif case == "huge_integer":
            damaged = '{"x":' + ("9" * 5000) + "}"
        else:
            record = json.loads(payload)
            record["source"]["body"] = "\ud800"
            damaged = json.dumps(record)
        _replace_row(path, key, damaged)
        with pytest.raises(
            StorageError, match="Invalid stored embedding payload"
        ) as caught:
            loaded = store.get_document(key)
            assert loaded is None
        assert caught.type is StorageError
        assert caught.value.__cause__ is None
        assert rows(path) == [(key, damaged)]


@pytest.mark.parametrize(
    ("field", "value"),
    [("synthetic", False), ("product_id", "unsupported")],
)
def test_forged_metadata_is_rejected_before_write(
    tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "embeddings.sqlite"
    source = document("alpha\n")
    forged = source.model_copy(
        update={"metadata": source.metadata.model_copy(update={field: value})}
    )
    with pytest.raises(ValueError, match="metadata"):
        document_embedding_key(forged, model_digest=MODEL_DIGEST)
    embedded = embed_eligible(chunk_document(forged))
    with EmbeddingStore(path) as store:
        key = store.put_document(
            source, embed_eligible(chunk_document(source)), model_digest=MODEL_DIGEST
        )
        before = rows(path)
        with pytest.raises(ValueError, match="metadata"):
            store.put_document(forged, embedded, model_digest=MODEL_DIGEST)
        assert rows(path) == before
        loaded = store.get_document(key)
    assert loaded is not None
    assert loaded.document.metadata.synthetic is True
    assert loaded.document.metadata.product_id == "hp-a"
    assert loaded.document.body == source.body


@pytest.mark.parametrize(
    "case", ["boolean_format_version", "float_dimensions", "blank_body"]
)
def test_loose_or_blank_payload_is_storage_error(tmp_path: Path, case: str) -> None:
    path = tmp_path / "embeddings.sqlite"
    source = document("alpha\n")
    legal_hash = hashlib.sha256(b"alpha\n").hexdigest()
    assert _identity_key(source, legal_hash) == document_embedding_key(
        source, model_digest=MODEL_DIGEST
    )
    with EmbeddingStore(path) as store:
        key = store.put_document(
            source, embed_eligible(chunk_document(source)), model_digest=MODEL_DIGEST
        )
        record = json.loads(rows(path)[0][1])
        read_key = key
        if case == "boolean_format_version":
            record["format_version"] = True
        elif case == "float_dimensions":
            record["embedding"]["dimensions"] = 1024.0
        else:
            blank = document(" \n")
            with pytest.raises(ValueError, match="non-blank"):
                document_embedding_key(blank, model_digest=MODEL_DIGEST)
            body_hash = hashlib.sha256(b" \n").hexdigest()
            read_key = _identity_key(blank, body_hash)
            record["source"]["body"] = " \n"
            record["source"]["body_sha256"] = body_hash
            chunks = chunk_document(blank)
            assert chunks
            assert all(not chunk.text.strip() for chunk in chunks)
            record["chunks"] = [
                {
                    "index": chunk.index,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "heading_path": list(chunk.heading_path),
                    "vector": None,
                }
                for chunk in chunks
            ]
        damaged = _canonical(record)
        _replace_row(path, key, damaged, None if read_key == key else read_key)
        with pytest.raises(StorageError, match="Invalid stored embedding payload"):
            store.get_document(read_key)
        assert rows(path) == [(read_key, damaged)]
        reread = json.loads(damaged)
        if case == "boolean_format_version":
            assert reread["format_version"] is True
        elif case == "float_dimensions":
            assert type(reread["embedding"]["dimensions"]) is float
        else:
            assert reread["source"]["body"] == " \n"
