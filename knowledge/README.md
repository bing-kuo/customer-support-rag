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
