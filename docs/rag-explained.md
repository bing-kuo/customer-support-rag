# RAG 入門：先找資料，再根據資料回答

想像公共圖書館有一份服務指南，裡面寫著開館時間、借閱規則和雨天活動。讀者問：「下雨時，親子活動在哪裡舉辦？」如果回答者先在指南中找到相關段落，再根據那段文字整理答案，這就是 RAG 的基本想法：先檢索（retrieval），再生成（generation）。

RAG 是 Retrieval-Augmented Generation 的縮寫。檢索負責找出可參考的資料，生成則負責把找到的資料組織成回答。這種流程讓回答有機會依據文件，但 RAG 這個名稱本身不代表引用一定正確、答案一定安全，也不保證系統找得到所需資訊。

## 為什麼要把文件切成 chunks？

如果把整本指南一次交給檢索或生成流程，內容可能太長，也不容易只找出最相關的部分。Chunk（片段）是切分後供索引或檢索使用的文字範圍。切分的目標不是越小越好，而是讓每段保留足夠上下文，同時能被有效尋找。

可以把切分策略想成由大到小尋找合適邊界：

```text
公共圖書館服務指南（文件）
├── 開館時間（section）
│   └── 開館日與時段說明（paragraph）
│       ├── 本館每週二至週日開館。（line）
│       └── 國定假日另行公告。（line）
├── 借閱規則（section）
│   └── 每張借書證最多借閱十冊，借期二十八天。（paragraph）
└── 雨天活動（section）
    └── 遇雨改至兒童閱覽室舉行。（paragraph）
```

這是設計概念的示意，不表示每個系統都會建立 section、paragraph、line、chunk 四種獨立資料型別。實作通常會設定優先順序：盡可能在 section、paragraph 等較完整語意邊界切開；放不下時再退到換行（line），最後才在長行中按字元上限切開。若一律切得很小，單一片段雖然精簡，卻可能失去適用對象、例外條件或前後文，讓檢索結果變得零碎。

### 本 PoC 的切分方式

本 PoC 的 chunker 以 `KnowledgeDocument.body` 為輸入，預設上限為 1,200 個 Python `str` 字元。這是字元數，不是 tokenizer token 數、UTF-8 bytes 數，也不是使用者所見的 grapheme cluster 數。程式先依簡化的 Markdown ATX 標題分 section，再於 section 內優先選上限內最後一個段落邊界，其次找換行，最後才安全地硬切長行。段落或行不會自動各自成為 chunk；它們是選擇切點的邊界。

目前支援 1 到 6 個 `#` 的 ATX 標題，前面最多容許三個空格；不是完整的 CommonMark parser。標題行仍留在原始 `body` 的切片文字中。`heading_path` 則是 chunk 附帶的 metadata（中繼資料），不是額外拼進 chunk 文字的內容。掃描標題時會略過 code fence 內看起來像標題的行，但後續切分仍依段落、換行和字元上限進行，不保證 fence 或程式碼區塊永遠不會被拆開。

## Overlap 能幫什麼忙？

Overlap（重疊）是讓相鄰 chunks 共用一小段文字。例如把長段落切成兩段時，第二段可重複第一段末尾幾個字，減少答案線索剛好落在邊界兩側、單獨檢索其中一段時上下文不足的情況。代價是重複索引與儲存，搜尋也可能回傳相似的重複片段，並把重複文字送入生成上下文。

「沒有 overlap」不等於「可以無損還原」。無損重建的條件是原文被完整涵蓋、沒有缺口、切片沒有被改寫，而且切片依原順序拼接。反過來說，有 overlap 也不代表一定無法還原；若保留原始 offsets，可以依位置去重或合併範圍。

本 PoC 使用 no-overlap。它以半開區間 `[char_start, char_end)` 標示每個 chunk 在 `body` 的字元位置，chunk 文字直接取 `body[char_start:char_end]`。既有測試對非空輸入檢查第一段從位置 0 開始、最後一段到 `len(body)`、相鄰區間連續、每段等於原始切片，且依序串接可回到原文。這些條件支持目前切分對已提供 body 的完整性；no-overlap 本身不是無損的原因，也不是唯一可無損的設計。

`char_start` 和 `char_end` 是 Python 字串的字元位置，不是 UTF-8 byte offset。`body_sha256` 則是整份 `body.encode("utf-8")` 的 SHA-256，每個 chunk 保存同一個 body hash；它不是各 chunk 個別計算的 hash。Chunker 不正規化已收到的 body，測試涵蓋繁體中文字、emoji，以及 LF、CRLF、單 CR 換行，並檢查不會把 CRLF 拆在兩個切片中。這只描述 chunker 收到 body 後的行為，不代表上游讀檔階段對所有來源也都保留換行。

## Embedding 怎麼協助檢索？

Keyword search（關鍵字搜尋）適合找型號、代碼、精確詞句；embedding（嵌入向量）則把文字表示成數值向量，讓檢索可以依語意相近程度排列候選，即使問題和來源用詞不完全相同。兩者各有用途：向量不保證理解所有語意，關鍵字也可能錯過不同措辭。Hybrid retrieval（混合檢索）是可以評估的組合選項，不應只因為它聽起來全面就假定效果更好。

例如來源寫「雨天活動改到室內」，讀者問「下雨時在哪裡舉辦？」向量檢索可能把這段來源排前面；但是否找得到，仍取決於資料、切分、embedding 和檢索設定，這只是原理示例，不是本 PoC 的模型輸出或命中結果。向量本身不會回答活動在哪裡。系統應取回原始來源文字，再根據文字組織答案並附上可追溯的來源。

用一個很小的數字例子理解 cosine 排序：假設問題向量是 `q = (1, 0)`，兩個片段向量分別是 `a = (2, 0)` 與 `b = (1, 1)`。Cosine similarity 的計算是內積除以兩向量長度乘積，因此 `q` 與 `a` 的 cosine 為 1，與 `b` 約為 0.707；在這組人造數字裡，`a` 會排在 `b` 前面。這只演示排序怎麼算，不是真實模型向量，也不表示第一名一定相關或答案正確。Cosine 是排序值，不是信心機率；不能把 0.9 解讀成「九成有把握」。

本 PoC 的 serving 流程依商品範圍篩選已存 embeddings，再做 exact cosine 排序，沒有 confidence cutoff。維護用 evaluator 另提供 cosine rank 與忽略空白的字元 bigram overlap lexical baseline rank 對照；這不等於 serving 已部署 keyword 或 hybrid retrieval，也不足以宣稱 embedding 優於關鍵字。要知道哪種方法適合特定問題，必須在同一資料和固定問題集上比較。

## 可重建，不等於找得對或答得好

驗證 RAG 至少要分兩層。第一層看資料工程有沒有失真：切片完整嗎？offset 是否對得上原文？相同輸入和設定是否得到相同結果？第二層才看檢索與回答品質：需要的來源有沒有排進前 k 名？回答是否包含正確事實和限制？引用是否支持每項主張？該回答時有沒有回答，該交由人員處理時有沒有交接？第一層通過不代表第二層通過。

下表把一般概念、目前 PoC 有的程式／測試證據，以及尚待驗證的品質問題分開。測試來源代表已有測試案例，不表示本輪重新執行過測試。

| 驗證面向 | 一般標準／原理 | 本 PoC 程式與既有測試涵蓋 | 尚未證明／品質驗證建議 |
| --- | --- | --- | --- |
| Lossless Partition | 非空原文由依序、連續且未改寫的切片完整涵蓋，拼接後等於原文；overlap 不是必要條件。 | `tests/test_chunking.py` 有 exact-partition helper，檢查 offsets、原切片與重建文字；也涵蓋空 body。 | 這只驗證切分保真，不代表 chunk 語意品質或檢索品質通過。 |
| Offset Integrity | 位置需明確標示所屬文字及單位；半開區間可避免相鄰位置重複計算。 | 欄位是 `char_start`／`char_end`，索引 Python `body` 字元；`body_sha256` 是完整 body 的 hash。測試檢查 slice、連續性、digest、Unicode 與換行。 | 不代表上游所有讀檔流程保留原始位元組；body hash 也不是 chunk hash。 |
| Determinism | 固定輸入、切分演算法版本和設定，輸出應可重現。 | 既有測試比較相同 document 與 `max_chars` 的 `chunk_document(...)` 結果。 | 不代表未來演算法版本或所有 runtime 永遠相同；品質流程仍應記錄 chunk profile/version。 |
| Edge Cases | 明確涵蓋空輸入、長行、Unicode、換行、標題與格式邊界。 | 既有來源包含 plain text、繁中／emoji、LF／CRLF／單 CR、長行硬切、略過 fence 內標題及跳級標題。 | fence 不保證不可拆；簡化標題辨識不是完整 Markdown parser。 |
| Performance Regression | 先定義可重現的上限或基準，再追蹤回歸；測試 instrumentation 和完整效能基準是不同證據。 | `test_candidate_searches_are_bounded_to_one_chunk_window` 追蹤候選搜尋範圍：4096 個 `a`、`max_chars=32` 時最大搜尋 span 不超過 33，並檢查切成 128 段。 | 這是 bounded regex instrumentation regression，不是完整 chunker benchmark、營運延遲測量或全程線性保證；標題與行掃描另有工作。 |
| Retrieval／回答品質 | 固定問題和相關來源標準，檢查 Recall@k、正確來源排名（如 MRR／rank distribution），再用人工規準判斷回答事實、條件、引用與 handoff。 | evaluator 有 cosine 排名和 lexical baseline 排名程式；本輪沒有執行產品測試或品質評估。 | 尚無本輪實測 Recall@k／MRR，也不能推論 embedding 勝過 keyword、答案正確或 cosine 是信心值。 |

開始調參前，先固定代表性問題集、每題期望來源、商品與知識版本，並留出不參與調參的 held-out 問題。同一份資料、相同問題和同一評分方式先建立 keyword／lexical baseline；每輪只改一項，例如 chunk size、overlap 或 retrieval 組合，並記錄 corpus、query、chunk profile、embedding identity、top-k、filter 等 config 版本。設定用題目調參，held-out 只在設定完成後檢查泛化，不拿來反覆選參數。報告 Recall@k 和來源排名時，也要另外用人工 rubric 檢查必要事實、條件是否保留、引用能否支持主張，以及是否正確回答或 handoff。延遲和資源需依實際測量記錄；usage 未知就記 `null`／未知，不填零或推估費用。

## 本 PoC 做到哪裡

上文的 chunking、embedding 和 retrieval 描述的是程式實作與既有測試來源；本輪沒有執行產品測試，也沒有執行新的 benchmark、Recall@k／MRR 品質評估或真模型實驗。建議的評分流程是後續品質驗證方法，不是已完成的 PoC 成效。想了解這個 PoC 的客服草稿定位、知識更新、歷史 live／mock 證據及目前限制，請閱讀[客服 RAG PoC 做到哪裡：可追溯的草稿，仍需要人判斷](poc-closeout-article.md)。

## 延伸閱讀

- [客服 RAG PoC 做到哪裡：可追溯的草稿，仍需要人判斷](poc-closeout-article.md)：本 PoC 案例、知識更新流程、live／mock 證據與限制。
