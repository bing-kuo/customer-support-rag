# 客服 RAG

以合成客服知識驗證檢索增強生成（RAG）的 proof of concept。程式碼使用 MIT License（見 [LICENSE](LICENSE)）；第三方套件保留各自授權（見 [NOTICE](NOTICE)）。`knowledge/`、`knowledge/drafts/` 與評測 suite 是合成資料，不含客戶紀錄。

## 目前狀態

截至 PR #15 合併後的 main（[`f7ec860a01ad447a9cb639fcb474b597b62f4fc6`](https://github.com/bing-kuo/customer-support-rag/commit/f7ec860a01ad447a9cb639fcb474b597b62f4fc6)），知識維護 CLI、`POST /ask`、`POST /assist` 與本機頁面已在程式中。PoC 仍未完成 P2 驗收：既有安全／語意失敗與未驗項保留，FAQ absent baseline 的舊錯誤根因未知，FAQ present 與 policy v1/v2 尚未完成驗證；sealed 評估不代表通過。既有真 embedding 候選尚未取得 release approval，也未啟用。離線 rehearsal 與 PR #15 的修補都不代表產品驗收完成；沒有解除 security hold。

本 repo 內記錄了不同舊版的歷史 live 生成結果：PR [#11](https://github.com/bing-kuo/customer-support-rag/pull/11) 的單題入口與 PR [#12](https://github.com/bing-kuo/customer-support-rag/pull/12) 舊版一般 `POST /ask` 各有一題回答；兩筆 `usage` 均為 `null`，`usage_code` 為 `unknown_billing_component`。較早的 PR #15 核准版本 [`09e24d1`](https://github.com/bing-kuo/customer-support-rag/commit/09e24d1e48ac71768dab08600feb8b47d04e17ce) 另記錄四個合成 dev 案例當時均成功取得生成回答，四案用量均未知。這些結果分屬不同舊版，不合併計算，也不代表 PR #15 最終合併版本或目前版本的語意品質已通過；最終合併版本沒有重跑 live 問答。PR [#13](https://github.com/bing-kuo/customer-support-rag/pull/13) 記錄了 UI 單題 live，但當時的畫面檔留在本機、未附為 GitHub 可檢視圖片。此 repo 為 private；以上 PR 與 commit 證據連結需要 repo 存取權，外部讀者可能無法開啟。

本輪依 D-018 調整自然 RAG 草稿的保證範圍：程式驗來源／版本／hash、引用 ID、字元 span、answer 範圍完整覆蓋、輸出 schema 與 handoff 欄位一致；這些檢查只能確認結構和定位，不能證明每個主張被來源語意支持、所有必要限制完整，或 status 符合人工處置意圖。模型 instruction 會要求保留條件、禁止、必做與停止指示，但這不是可機械保證。來源文件按已核對的原始 chunks 整份送入；超過既有 4 個 evidence／1200 字元／wire 界線就不送 Luna，回現有 handoff。錯誤或相反建議即使結構合法，仍須人工依來源判讀；目前 UI 沒有強制放行閘門。

P2 歷史生成候選仍綁建置 commit `0cb3a9a`。PR #15 的程式／prompt/schema 變更沒有改標舊證據，也沒有重算既有向量。此 PoC 收尾不包含新的 live 評測、真候選核准或啟用；既有錯誤與未驗狀態保持原樣。

## 從乾淨 checkout 開始

需要 Python 3.12 或 3.13，以及已安裝的 `uv`。在 repository 根目錄執行：

```sh
uv sync --locked --extra dev
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pytest
```

離線 lifecycle rehearsal 使用臨時目錄、mock embedding 與明示的假 approval fixtures：

```sh
uv run --locked python scripts/rehearse_delivery.py
```

腳本會建置／重用候選、檢查無變更 no-op、比對 FAQ 前後及政策 v1/v2、測試候選隔離、只在臨時 workspace 走明確 activate、重新讀取狀態，並驗證失敗切換保留舊 snapshot。它先從政策 v1 執行 compare-failure recovery，再明確 activate policy v2；restart/status 後，active serving 的 `/ask` 與 `/assist` 引用版本都核為 v2，policy-v1 舊 snapshot 仍保留。它也用 `CUSTOMER_SUPPORT_RAG_RELEASE_ROOT` 接上暫存 API：未 activate 時 `/ask` 與填完必要欄位的 `/assist` 都回 `release_unavailable`；active snapshot serving 使用 mock query embedding、mock generator 和明確標示的假答案，並確認缺欄位 assist 不呼叫 adapter。腳本會封鎖 Python socket 連線，不讀真 key、不啟 Ollama、不呼叫 Luna、不做 paid request；輸出 usage 為 `null`。腳本建立的 Marcus approval 僅是假夾具，不是人工審查或正式核准。成功不等於 P2、live 問答或知識發布已驗收。

## 本機 API 與頁面

預設可不設定 key 或資料庫啟動本機服務，用於查看程序健康、API 文件與靜態頁面：

```sh
uv run --locked uvicorn customer_support_rag.api:app \
  --host 127.0.0.1 --port 8000 --no-proxy-headers
```

開啟 `http://127.0.0.1:8000/` 查看頁面，`/docs` 查看 OpenAPI，`/health` 查看程序狀態。Health 只證明 API 程序可回應，不檢查 active release、資料庫、Ollama、key 或 Luna 是否可用。請求須使用 loopback Host；瀏覽器 Origin 必須與本機頁面同源。未知路徑不會回退成首頁。

真實 `POST /ask` 需要已核准的 active release 或相容的既有 SQLite 設定、可用的本機 Ollama embedding，以及 OpenAI key。查詢會先用 Ollama 產生問句向量，再以既有向量檢索；在有來源且完整 bundle 可放入既有界線時，才送一次 `gpt-6-luna` 請求。沒有檢索來源或 bundle 無法完整、安全地提供時，API 回現有 handoff；它不會判斷一般語意何時必須交人工。`POST /assist` 收集必要欄位後也可能進入模型。回答是供客服打開來源核對的自然草稿，並非無人監督客服；現有 UI 沒有強制人工放行功能。下列請求範例只供操作者在已核准的環境手動執行，非測試命令：

```sh
curl -sS http://127.0.0.1:8000/ask \
  -H 'content-type: application/json' \
  -d '{"product_id":"hp-a","question":"HP-A 耳機可以用水洗嗎？"}'
```

模型與來源：embedding adapter 固定使用 `qwen3-embedding:0.6b`、1024 維及已記錄 digest `ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d`；模型來源是 [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)，該模型頁標示 Apache-2.0。Ollama 本機 embedding API 介面依 [Ollama Embed API](https://docs.ollama.com/api/embed)。digest 來自既有記錄，本次 README／rehearsal 沒有啟動模型重查。生成 adapter 的固定 API model 名稱為 `gpt-6-luna`；它是服務端模型，不由本 repository 下載。模型可用性與帳務依實際 provider 設定，不能由 `/health` 推定。

### Key 設定

[`.env.example`](.env.example) 只有一行非秘密 placeholder。將它複製到你自行選擇、repository 外的 key directory，替換成自己的 key，並限制目錄與檔案權限：

```sh
KEY_ROOT=/path/to/your/private-key-directory
mkdir -p "$KEY_ROOT"
chmod 700 "$KEY_ROOT"
cp .env.example "$KEY_ROOT/.env"
chmod 600 "$KEY_ROOT/.env"
```

設定 `CUSTOMER_SUPPORT_RAG_KEY_ROOT` 指向這個目錄；程式只讀其中 `.env` 的 `OPENAI_API_KEY`。不要把真 `.env`、資料庫、approval 或私有評測資料加入 Git。key root 與 release/workspace 路徑由操作者在自己的機器設定；README 不提供機器專屬位置。

使用已核准並啟用的 release serving 時，將 `CUSTOMER_SUPPORT_RAG_RELEASE_ROOT` 設為 maintenance workspace 路徑，並將 `CUSTOMER_SUPPORT_RAG_KEY_ROOT` 設為上述 key directory。兩者是啟動程序的環境變數；`.env.example` 只供 key root 使用，不是服務設定檔。release-root 模式由 `active.json` 載入快照，不需要舊 `CUSTOMER_SUPPORT_RAG_DATABASE`。

### 手動維護命令

維護 CLI 預設使用 mock embedding，不需要 key。`--embed ollama` 會呼叫本機 Ollama；不要在離線 rehearsal 使用它。人工核准檔必須來自實際審查，腳本產生的假 approval 不可拿到此處使用。

```sh
uv run --locked python -m customer_support_rag.maintenance --help
uv run --locked python -m customer_support_rag.maintenance scan --root knowledge
uv run --locked python -m customer_support_rag.maintenance status \
  --workspace /path/to/your/authorized-release-workspace
```

真實候選須先由授權 reviewer 核准文件 bytes 與 release manifest、資料庫及 regression hashes，再使用 `consume-document`／`consume-release`。只有已消費的真實 release approval 才可手動 `activate`。切換前讀取 `status`，`--expect` 必須是目前 `active.json` bytes 的 SHA-256；首次切換才使用 `--expect-none`。若結果為 `activated_unconfirmed`，先再查 `status`；若 pointer 已指向目標 release，不要以舊 `--expect` 重送。舊 snapshot 保留，除非另有明確授權，不要刪除。這些是真實環境操作說明，rehearsal 不會對真 workspace 執行。

完整的初次 build、same-input no-op/reuse、FAQ 更新、evaluate、release 核准、activate 與 serving 啟動命令見[交付指南](docs/reproducible-delivery.md)。其中 `--embed ollama` 會呼叫本機 embedding model；範例是操作者參考流程，不是 P3 rehearsal，亦非本 PR 執行授權。PR #15 合併版本的 rehearsal 沒有呼叫 Luna 或啟用真候選；歷史單題 live 記錄見上方 PR #11、#12、#13，不能視為目前版本或完整品質驗收。

文章完整說明架構取捨、歷史 mock/live 證據及仍未驗證的範圍，見[PoC 收尾文章](docs/poc-closeout-article.md)。

## 故障與恢復

- `paid_blocked`：key 缺少、格式或權限不符，或程序未提供所需設定。確認 key directory 內只有正確的 `OPENAI_API_KEY`、檔案 mode 為 `0600`；不要輸出或貼上 key。
- `dependency_unavailable`／`release_unavailable`：本機資料庫、schema、artifact 或 active snapshot 不可用。先確認設定指向操作者的正確 workspace，並用 maintenance `status` 查看目前 pointer；不要把檔案錯誤當成無證據。
- `embedding_unavailable`：本機 Ollama／模型 digest／embedding 回應未符合固定契約。查明本機模型狀態後再決定下一步；本程式不自動重試。
- `429`（`insufficient_quota`、`rate_limit`、`provider_capacity`）：停止該請求，不自動重送；辨明 provider 回報後才決定是否由操作者發起新請求。
- timeout、client 取消或連線中斷：不代表上游已停止，也不代表沒有計費。先查 provider 狀態與用量，不要因為本機沒收到答案就重送。
- 成功答案但 `usage` 為 `null`：未知用量不是 0；保留答案與 `usage_code`，先查 provider usage，不要重送原問題以猜測帳務。
- pointer 已更新但狀態未確認：執行 `status` 核對目標 release。不要重播舊 expected hash，也不要刪除舊 snapshot。

診斷保存維持明確 opt-in；一般 serving 不會自動保存 provider body。診斷失敗會安全停止，不洩漏 raw exception 或重送。細節與 offline regression 證據由本 PR 及其測試追蹤。
