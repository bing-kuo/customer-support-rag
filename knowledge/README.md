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

單次最多 8 個原始 chunks，每段最多 1200 個 Python 字元。超過任一個上限時整批 `ValueError`，不會自動分批或截斷。字元數不是 token 數。這次沒有 query embedding；文件向量與未來問句若要使用 Qwen instruct 格式，必須另外定義，不能視為已經對齊。

Payload 只有固定 `model`、`input`、`truncate: false`、`keep_alive: 0`。呼叫端建立並關閉 client。下面是預期的 live 呼叫形狀；本變更的測試使用 `httpx.MockTransport`，沒有執行這段，也沒有連到 Ollama：

```python
import httpx

from customer_support_rag.knowledge import (
    OllamaEmbeddingAdapter,
    chunk_document,
    read_document,
)

document = read_document("knowledge", "hp-a/headphone-care.md")
chunks = chunk_document(document)
with httpx.Client(trust_env=False, follow_redirects=False) as client:
    embedded = OllamaEmbeddingAdapter(client).embed_chunks(chunks)
```

Adapter 每次請求另外指定 `follow_redirects=False`，以及 connect 5 秒、read 60 秒、write 5 秒、pool 5 秒。這是各階段 timeout，不是嚴格的整體 deadline。client 不可使用會重試的 transport、proxy、event hook 或其他目的地；正式路徑用預設且不重試的 HTTP transport，測試才注入 mock。3xx、非 2xx、連線錯誤、timeout 與不合契約的回應都拋 `EmbeddingError`，不重試、不換模型、不回傳部分結果。timeout 不表示 Ollama 已經停止。錯誤訊息只標 timeout、request、HTTP status 或 invalid response，不包含正文。

設計依據是 [Ollama embed API](https://docs.ollama.com/api/embed) 的文字陣列、`truncate=false` 與 `keep_alive`，以及 [Qwen3-Embedding-0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) 列出的 1024 維。官方說明不是本機 tag、digest 或維度的 live 實測。開發用的 Grok 不是這個產品的 embedding 模型。

### HP-A 範例（mock，非 live）

來源文件是 `knowledge/hp-a/headphone-care.md`，內容與 `02053e16eb7d8234a329ad76cc6ed4f5ca5e73ba` 相同。metadata 為 `document_id=hp-a-care`、`product_id=hp-a`、`version=1`、`title=HP-A 耳機日常保養`、`synthetic=true`。parser 去掉 front matter 後，正文 SHA-256 是 `223243a3ba9c40840746b1b2ee66e40b970a8fefd1324bb586930a381f9f4e59`，含開頭一個 LF，共 71 個 Python 字元。這是正文 hash，不是整份檔案 hash。

`chunk_document(document, max_chars=1200)` 的對應如下。`tests/test_embedding.py` 用真實 parser、chunker 與 mock adapter 核對這份文件。

| 原始 index | offsets | 字元數 | heading path | 此次處理 |
| --- | --- | --- | --- | --- |
| 0 | [0, 1) | 1 | 空 | 文字是 `\n`，略過；原 chunk 保留 |
| 1 | [1, 71) | 70 | 日常保養 | 精確送出下面的字串 |

實際送出的 Python 字串如下，結尾有一個 LF。`# 日常保養` 來自正文，不是 metadata title，也不是模型產生的文字：

```text
# 日常保養

使用後以乾燥柔軟的布擦拭耳罩。請勿將耳機浸入水中，也不要使用溶劑清潔。

如需檢查保固條件，請聯絡購買通路並備妥購買證明。
```

略過的 chunk 0 是 `"\n"`。送出字串的 repr 是：

```text
'# 日常保養\n\n使用後以乾燥柔軟的布擦拭耳罩。請勿將耳機浸入水中，也不要使用溶劑清潔。\n\n如需檢查保固條件，請聯絡購買通路並備妥購買證明。\n'
```

只有這一個 eligible input，所以只會有一筆 1024 維結果，來源 index 仍是 1。Mock 測試只檢查組裝、數量與來源，不使用或展示 Ollama 的真實向量。空白 chunk 被略過時，結果數可以少於原 chunks。等長回應若被 provider 重排，shape 檢查看不出來；live 核對尚未執行。
