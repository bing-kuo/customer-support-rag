# 知識文件

本目錄存放少量耳機客服知識文件範例。所有內容均為合成資料，不代表真實商品或客戶資訊。

Markdown 文件以首行及獨立一行 `+++` 包住 TOML front matter，正文為原始 Markdown。必要欄位為 `document_id`、`product_id`（`hp-a` 或 `hp-b`）、`title`、`version` 及布林 `synthetic = true`。目前只維護各文件的現行版本。

讀取成功只表示 front matter 和正文符合格式，不代表文件已經人工審查、核准或啟用於線上 RAG。此切片不執行正文、不 render HTML，也不讀取 URL。

## 本機切段

`chunk_document` 接受已解析的 `KnowledgeDocument`，依 ATX 標題與段落邊界切分正文。字元上限包含空白和換行，計算 Python 字串字元，不等同模型 token 預算。

```python
from customer_support_rag.knowledge import chunk_document, read_document

document = read_document("knowledge", "hp-a/headphone-care.md")
chunks = chunk_document(document, max_chars=1200)
```

每個不可變 chunk 保留來源 metadata、正文 SHA-256、零起算順序、正文半開字元 offsets、原文片段及標題路徑。片段不重疊，依序串接會還原完整正文；空白與 LF、CRLF、CR 換行均原樣保留。切段只辨識 0–3 個前置空格的 ATX 標題及 fenced code；不解析 Setext、HTML 或完整 CommonMark，長行可能在任意字元處切開。正文中的指令、HTML 與 URL 都只是原樣文字，不會執行或讀取。

## 本機 embedding

`OllamaEmbeddingAdapter.embed_chunks` 接收既有 `KnowledgeChunk`，只把每個非空白 `chunk.text` 送到固定位址 `POST http://127.0.0.1:11434/api/embed`。模型常數是 `qwen3-embedding:0.6b`，每個向量必須恰為 1024 維、有限且平方和大於 0。回傳的 frozen `EmbeddedChunk` 持有原 chunk 與 float tuple，只存在記憶體，不寫資料庫，也不啟用檢索。

送出的字串就是 `chunk.text`。不會 strip、不會正規化換行，也不會加上 metadata title、product、heading path 或 query 指令。正文裡原本的標題會照樣送出。`not chunk.text.strip()` 的片段略過，原 chunks 不修改；回傳順序是其餘輸入的順序，來源 index 不重編。相同文字的不同來源各送一次，不依文字或 hash 去重。

單次最多 8 個原始 chunks，每段最多 1200 個 Python 字元。超過任一個上限時整批 `ValueError`，不會自動分批或截斷。字元數不是 token 數。文件 embedding 仍只送 `chunk.text`。問句使用另一個 query profile，見下方檢索一節；它不會改寫已保存的文件向量。

Payload 只有固定 `model`、`input`、`truncate: false`、`keep_alive: 0`。Adapter 建立並關閉自己的 HTTPX client：`trust_env` 為 false、沒有 auth、params、cookies 或 event hooks，也不跟隨 redirect。正式路徑使用 `HTTPTransport()`，預設連線重試次數是 0。測試只能傳入精確的 `httpx.MockTransport`，不能傳入 caller client。下面是預期的 live 呼叫形狀；本變更的測試使用 mock transport，沒有執行這段，也沒有連到 Ollama：

```python
from customer_support_rag.knowledge import (
    OllamaEmbeddingAdapter,
    chunk_document,
    read_document,
)

document = read_document("knowledge", "hp-a/headphone-care.md")
chunks = chunk_document(document)
with OllamaEmbeddingAdapter() as adapter:
    embedded = adapter.embed_chunks(chunks)
```

每次請求指定 `follow_redirects=False`，以及 connect 5 秒、read 60 秒、write 5 秒、pool 5 秒。這是各階段 timeout，不是嚴格的整體 deadline。目的地固定為上面的 URL，不重試、不換模型、不回傳部分結果。3xx、非 2xx、連線錯誤、timeout 與不合契約的回應都拋 `EmbeddingError`。timeout 不表示 Ollama 已經停止。錯誤訊息只標 timeout、request、HTTP status 或 invalid response，不包含正文。

設計依據是 [Ollama embed API](https://docs.ollama.com/api/embed) 的文字陣列、`truncate=false` 與 `keep_alive`，以及 [Qwen3-Embedding-0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) 列出的 1024 維。官方說明不是本機 tag、digest 或維度的 live 實測。

### HP-A 範例（mock，非 live）

來源文件是 `knowledge/hp-a/headphone-care.md`，內容與 `02053e16eb7d8234a329ad76cc6ed4f5ca5e73ba` 相同。metadata 為 `document_id=hp-a-care`、`product_id=hp-a`、`version=1`、`title=HP-A 耳機日常保養`、`synthetic=true`。parser 去掉 front matter 後，正文 SHA-256 是 `223243a3ba9c40840746b1b2ee66e40b970a8fefd1324bb586930a381f9f4e59`，含開頭一個 LF，共 71 個 Python 字元。這是正文 hash，不是整份檔案 hash。

`chunk_document(document, max_chars=1200)` 的對應如下。`tests/test_embedding.py` 用真實 parser、chunker 與 mock adapter 核對這份文件。

| 原始 index | offsets | 字元數 | heading path | 此次處理 |
| --- | --- | --- | --- | --- |
| 0 | [0, 1) | 1 | 空 | 文字是 `\n`，略過；原 chunk 保留 |
| 1 | [1, 71) | 70 | 日常保養 | 精確送出下面 repr 的字串 |

實際送出的 Python 字串 repr 如下。結尾的 `\n` 是正文的 LF。`# 日常保養` 來自正文，不是 metadata title，也不是模型產生的文字。略過的 chunk 0 是 `"\n"`。

```text
'# 日常保養\n\n使用後以乾燥柔軟的布擦拭耳罩。請勿將耳機浸入水中，也不要使用溶劑清潔。\n\n如需檢查保固條件，請聯絡購買通路並備妥購買證明。\n'
```

只有這一個 eligible input，所以只會有一筆 1024 維結果，來源 index 仍是 1。Mock 測試只檢查組裝、數量與來源，不使用或展示 Ollama 的真實向量。空白 chunk 被略過時，結果數可以少於原 chunks。等長回應若被 provider 重排，shape 檢查看不出來；live 核對尚未執行。

## 持久保存

`EmbeddingStore` 把一份文件的來源、chunk 位置與向量放進呼叫者指定的 SQLite 檔。它不呼叫 adapter、不掃資料夾、也不重算向量。路徑的 parent 必須已經存在；建議用 `.data/`，這個目錄已在 `.gitignore`，不要提交 DB 或 journal。

新的空檔才會建立 schema。`PRAGMA user_version` 必須是 1，而且只能有這一張表：

```sql
CREATE TABLE embedding_documents (
    artifact_key TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL
);
```

`payload_json` 是一段 canonical JSON：`ensure_ascii=False`、`sort_keys=True`、`separators=(",", ":")`、`allow_nan=False`。下面的縮排範例只給人讀，向量用說明文字代替，不是資料庫裡的內容，也不能拿來算 key。

```json
{
  "format_version": 1,
  "source": {
    "metadata": {
      "document_id": "hp-a-care",
      "product_id": "hp-a",
      "title": "HP-A 耳機日常保養",
      "version": "1",
      "synthetic": true
    },
    "body": "<完整原正文，含開頭 LF>",
    "body_sha256": "223243a3ba9c40840746b1b2ee66e40b970a8fefd1324bb586930a381f9f4e59"
  },
  "embedding": {
    "model": "qwen3-embedding:0.6b",
    "model_digest": "<64 位小寫 hex>",
    "dimensions": 1024,
    "input_profile": "chunk-text-v1"
  },
  "chunking": {"profile": "heading-char-v1", "max_chars": 1200},
  "chunks": [
    {
      "index": 0,
      "char_start": 0,
      "char_end": 1,
      "heading_path": [],
      "vector": null
    },
    {
      "index": 1,
      "char_start": 1,
      "char_end": 71,
      "heading_path": ["日常保養"],
      "vector": "<資料庫裡是完整 1024 個 JSON number，這裡省略>"
    }
  ]
}
```

沒有 approval、active、ready 或 published 欄位。空白 chunk 的 `vector` 是 JSON null；其餘是完整 1024 個有限且平方和大於 0 的數字。chunk 原文用 `source.body[char_start:char_end]` 還原，不另外存一份正文。

`artifact_key` 是 identity 的 canonical JSON UTF-8 SHA-256。identity 含 `format_version`、五欄 metadata、`body_sha256`、整個 embedding 設定與整個 chunking 設定，不含向量、也不含正文本身。這是內容識別，不是核准或簽章。精確 key 在 `tests/test_embedding_storage.py`。

呼叫者在推論前用同一份文件和已知 digest 計算 key。digest 由呼叫端提供，store 不查 Ollama。下面的 digest 是先前 live 觀察到的 tag digest；本範例沒有重新查詢或執行模型。向量是 `(1 + index) / 1024` 的 mock，不是 live 前綴補零。

```python
from customer_support_rag.knowledge import (
    EmbeddedChunk,
    EmbeddingStore,
    chunk_document,
    document_embedding_key,
    read_document,
)

document = read_document("knowledge", "hp-a/headphone-care.md")
digest = "ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d"
chunks = chunk_document(document)
embeddings = [
    EmbeddedChunk(
        chunk=chunk,
        vector=tuple((1 + index) / 1024 for index in range(1024)),
    )
    for index, chunk in enumerate(chunks)
    if chunk.text.strip()
]
key = document_embedding_key(document, model_digest=digest)
with EmbeddingStore(".data/embeddings.sqlite") as store:
    stored_key = store.put_document(document, embeddings, model_digest=digest)

with EmbeddingStore(".data/embeddings.sqlite") as store:
    loaded = store.get_document(stored_key)
```

正文必須有非空白內容，metadata 必須能再通過合成資料與 `hp-a` / `hp-b` 檢查；`model_copy` 繞過驗證時，key 與 put 會在 transaction 前拒絕。驗證在記憶體做完才進入 `BEGIN IMMEDIATE`。SQLite lock timeout 是 5 秒，失敗不重試。commit 成功才回傳 key。寫入失敗會 rollback，舊列保持不變。同 key 且 canonical payload 完全相同時不新增列；同 key 但向量不同時拋 `StorageConflictError`，舊值保留。沒有自動 latest、刪除或 active pointer。`get_document` 找不到 key 才回 `None`；列存在但內容不合契約時拋 `StorageError`，不把它當成 miss，也不修復或重算。讀回時 `format_version` 與 `dimensions` 必須是精確整數。get 成功只表示這筆技術保存可讀，不表示文件已核准，也不能當作檢索或問答結果。

## 問句向量與 exact cosine

`query_embedding_input` 組出固定 profile `qwen3-retrieval-query-v1`。中間是一個 LF，`Query:` 後面不加空格，結尾不加額外 LF。問句必須是非空白 str、最多 1,000 個 Python 字元；驗證後保留原字串，不 strip、不加 metadata，也不做 Unicode 正規化。完整輸入再檢查不超過 1,200 字元，超過就拒絕、不截斷。字元數不是 token 數。

```text
Instruct: Given a customer support question, retrieve relevant passages that answer the question
Query:HP-A 耳機可以用水洗嗎？
```

`OllamaEmbeddingAdapter.embed_query` 用同一個 owned client，在唯一一次 `POST /api/embed` 的前後各 `GET /api/tags`。兩次都要有唯一一筆 name 恰為 `qwen3-embedding:0.6b` 的項目，而且 digest 是 64 位小寫 hex、等於呼叫者給的 `expected_model_digest`。缺模型、重複、格式錯誤、與預期不符，或任一次 GET／POST 失敗，都拋 `EmbeddingError`，不回傳向量。前置失敗不會 POST。timeout、不跟隨 redirect、不重試，與文件 embedding 相同。問句與向量不寫進文件 DB。

`ExactCosineIndex.from_store` 只 `get` 呼叫者列出的 artifact keys，不掃資料庫、不呼叫模型。任一 key 缺失是 `RetrievalError`；存檔損毀仍是 `StorageError`。文件的 model、完整 digest、1024 維、`chunk-text-v1` 與 `heading-char-v1` 必須和這次 query 相同，否則整批失敗。空白 chunk 不進矩陣，原 index 與 offsets 保留。超過 1,000 個 eligible chunks 整批拒絕。

`search` 只在記憶體算分。`product_id` 必須是 `hp-a` 或 `hp-b`，先依 `chunk.metadata.product_id` 過濾，再算 cosine 與 top-k。沒有該商品的候選就回 `[]`；這只表示所選 corpus 沒有該商品，不是語意拒答。`top_k` 是 1 到 10 的整數，預設 5。向量用 NumPy `float64` 做副本上的 L2 normalize，不改已存 tuple。排序鍵是 raw cosine 由高到低，其次 artifact key 與原 chunk index 由低到高。分數不四捨五入、不乘 100、不當機率，也沒有 similarity threshold；低分或負分仍會回傳。

這些 keys 是技術實驗的選擇，不是 approval 或已啟用的知識。介面不接 API、chat 或瀏覽器。

### Mock 排名示例（非 live、非品質分數）

下面的 `1.0` 是人造 basis 向量的數學結果：query 是 e1（第 1 維為 1，其餘為 0），HP-A 的 eligible chunk 是 `2 * e1`。HP-B 即使設成 `9 * e1`、cosine 同樣是 1，`product_id="hp-a"` 也不會讓它進入結果。測試 `tests/test_query_retrieval.py` 用真實 reader、chunker、SQLite 與 mock transport 核對這段；本 README 沒有呼叫 Ollama。

來源是 `knowledge/hp-a/headphone-care.md`。明確 keys：

| 文件 | artifact key |
| --- | --- |
| HP-A `hp-a-care` | `9a2283a73584261d31460d1b46855145d384c756834c8984c1539e96647cb4ad` |
| HP-B `hp-b-connection` | `18812b328a8e3614c4c83a849c2bfcd108e695453c8eaf9d57cf87200cefafae` |

`product_id="hp-a"` 的第一筆是 HP-A key、`document_id=hp-a-care`、`version=1`、chunk index 1、offsets `[1, 71)`、heading `日常保養`。原文就是上方 embedding 小節那一筆 70 字元的 chunk，raw score 是 `1.0`。只有一個 HP-A 候選時，第一名不代表品質已經驗證。

```python
from customer_support_rag.knowledge import (
    ExactCosineIndex,
    OllamaEmbeddingAdapter,
    query_embedding_input,
)

question = "HP-A 耳機可以用水洗嗎？"
text = query_embedding_input(question)
digest = "ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d"
keys = (
    "9a2283a73584261d31460d1b46855145d384c756834c8984c1539e96647cb4ad",
    "18812b328a8e3614c4c83a849c2bfcd108e695453c8eaf9d57cf87200cefafae",
)
with OllamaEmbeddingAdapter() as adapter:
    query = adapter.embed_query(question, expected_model_digest=digest)
index = ExactCosineIndex.from_store(
    store, artifact_keys=keys, expected_model_digest=digest
)
hits = index.search(query, product_id="hp-a", top_k=5)
```

`store` 必須是呼叫者已經打開、而且只含自己指定 keys 的 `EmbeddingStore`。上面的 live 呼叫形狀沒有在這個變更裡執行。
