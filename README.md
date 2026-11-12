# 客服 RAG

以合成資料驗證客服檢索增強生成（RAG）的 proof of concept。此 repository 目前處於早期開發階段，尚未核准用於正式環境。

## 開發環境

- CI 已驗證 Python 3.12 與 3.13
- `uv.lock` 固定應用程式與開發依賴的完整版本及雜湊
- FastAPI 提供 HTTP routing，Pydantic 定義並驗證健康檢查回應，Uvicorn 在本機啟動 ASGI app
- Starlette 固定在相容版本。HTTPX `0.28.1` 是 runtime dependency，供本機 embedding adapter 使用，也支援 FastAPI `TestClient`。NumPy `2.3.5` 是 runtime dependency，只用於已載入向量的 float64 exact cosine。pytest 執行測試，Ruff 負責格式與 lint

Clone 後在獨立 repository checkout 執行一次，啟用 staged secret scan：

```sh
python3 scripts/install_git_hooks.py
```

安裝器只設定此 clone 的 local `core.hooksPath`。它會保留既有自訂 hooks，並拒絕在 linked worktree、已有 hooks 設定或其他 linked worktrees 存在時覆寫；請在建立 linked worktrees 前執行。安裝後一般 `git commit` 會執行 project-local Gitleaks hook。

```sh
uv sync --locked --extra dev
```

## 本機檢查

```sh
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pytest
```

## 啟動 API

```sh
uv run --locked uvicorn customer_support_rag.api:app --host 127.0.0.1 --port 8000 --no-proxy-headers
```

本機啟動必須綁 `127.0.0.1`。`--no-proxy-headers` 避免 `X-Forwarded-Host` 改寫 Host。`GET http://127.0.0.1:8000/health` 回傳服務狀態、應用程式名稱與版本。這個端點只回報 API 程序狀態，不檢查模型、知識庫或儲存服務。`GET /docs` 同樣不讀 key、不開資料庫、也不送問句。

## 本機頁面

同一 FastAPI 程序在 `GET /` 供應 repository 根目錄的 `web/index.html`，固定資產為 `GET /assets/style.css` 與 `GET /assets/app.js`。啟動後以瀏覽器開啟 `http://127.0.0.1:8000/`。程式用套件位置定位 `web/`，請求不能指定 web 根目錄或 key 根目錄。檔案不是一般檔、或名稱不在上述清單時回 JSON 404。未知路徑與未知 POST 維持 JSON／HTTP 錯誤，不會 fallback 成 HTML 首頁。暫時測試用的 web tree 不是產品 UI。

目前頁面是合成耳機 PoC 的單次問答草稿：選 `hp-a`／`hp-b`、送一次 `POST /ask`、以純文字顯示回答或轉人工摘要。這還沒有通過 agent-browser 的最終畫面／鍵盤／窄版驗收，不能當成 P1 完成或 live 問答。

`POST /ask` 的 Host 必須是 loopback（`127.0.0.1`、`localhost`、`::1`，可含 port）。沒有 Origin 的 CLI 可以送一次。瀏覽器若帶 Origin，必須與該次請求的 scheme、host、port 相同。`Origin: null` 與非同源會被拒絕，回應仍是固定 code 與 `request_id`。沒有 wildcard CORS。Host／Origin 被拒絕時不會讀 key 或呼叫模型。`GET /`、`GET /assets/*`、`GET /health` 與 `GET /docs` 也不讀 key、不開資料庫、不送問句。

## 單次問答

`POST /ask` 接受一個沒有對話狀態的 JSON：

```json
{"product_id": "hp-a", "question": "HP-A 耳機可以用水洗嗎？"}
```

`product_id` 只接受 `hp-a` 或 `hp-b`。`question` 必須是非空白字串，最多 1000 個 Python 字元，並保留原問句。history、路徑、URL、artifact key、模型、endpoint、system prompt、`top_k` 和其他額外欄位都會被拒絕。HTTP body 以實際讀到的 bytes 計算，上限 16KiB；超過回 `413`，格式或型別錯誤回 `422`。這兩種錯誤只回固定 code，不回顯問句或多餘欄位的值。非法輸入不會檢索、讀 key 或呼叫模型。

伺服器使用程序設定，不接受請求覆寫：

```sh
export CUSTOMER_SUPPORT_RAG_DATABASE=.data/embeddings.sqlite
export CUSTOMER_SUPPORT_RAG_KEY_ROOT=.
uv run --locked uvicorn customer_support_rag.api:app --host 127.0.0.1 --port 8000 --no-proxy-headers
```

`CUSTOMER_SUPPORT_RAG_DATABASE` 必須是已經存在的 SQLite。這個程序只讀開啟它，缺檔時不會自己建立，也不會寫 schema。預設載入的 artifact keys 是 HP-A care `9a2283a73584261d31460d1b46855145d384c756834c8984c1539e96647cb4ad` 與 HP-B connection `18812b328a8e3614c4c83a849c2bfcd108e695453c8eaf9d57cf87200cefafae`，digest 固定為 `ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d`。key 仍只讀這個 key root 的 `.env` 裡既有的 `OPENAI_API_KEY`。請不要把真的 `.env` 或資料庫提交進 git。

一次成功的呼叫會對本機 Ollama 做問句 embedding（前後各核一次 digest），再用 exact cosine 在該商品的既有向量裡取最多四個整段 chunk。證據合計超過 1200 字時留下排序在前、仍放得進界線的完整 chunk，不截斷原文。長度選取之前先看這次全部候選：同一文件有兩個版本就停，不送 Luna。單一 chunk 超過界線也會停。沒有該商品的候選時直接 `no_evidence`，不呼叫 Luna。有候選才送一次 Luna。cosine 只用來排序，沒有相似度門檻，也不能當成拒答或退款決定。

本機範例：

```sh
curl -sS http://127.0.0.1:8000/ask \
  -H 'content-type: application/json' \
  -d '{"product_id":"hp-a","question":"HP-A 耳機可以用水洗嗎？"}'
```

下面四個 JSON 是說明用的形狀，不是這次變更的 live 回應。引用文字和 offsets 來自本機已存 chunk；回答是模型草稿。`request_id` 每次請求都新產生，不是 `hp_a_once` 的固定 ID，也不表示跨請求去重。

有引用的回答：

```json
{
  "request_id": "ask-example",
  "status": "answer",
  "answer": "請用乾燥的軟布擦拭，耳機不可泡水，也別用溶劑清潔。",
  "reason": null,
  "sources": [
    {
      "source_id": "ev:9a2283a73584261d31460d1b46855145d384c756834c8984c1539e96647cb4ad:1",
      "document_id": "hp-a-care",
      "product_id": "hp-a",
      "version": "1",
      "title": "HP-A 耳機日常保養",
      "chunk_index": 1,
      "heading": "日常保養",
      "char_start": 1,
      "char_end": 71,
      "text": "# 日常保養\n\n使用後以乾燥柔軟的布擦拭耳罩。請勿將耳機浸入水中，也不要使用溶劑清潔。\n\n如需檢查保固條件，請聯絡購買通路並備妥購買證明。\n"
    }
  ],
  "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20, "cached_tokens": null, "cache_write_tokens": null, "ttl_30m_tokens": null, "ttl_12h_tokens": null, "reasoning_tokens": null},
  "usage_code": null,
  "usage_fields": []
}
```

這一筆的 `artifact_key` 是 `9a2283a73584261d31460d1b46855145d384c756834c8984c1539e96647cb4ad`，與 `source_id` 的中段相同。

沒有候選：

```json
{"request_id": "ask-example", "status": "handoff", "answer": null, "reason": "no_evidence", "sources": [], "usage": null, "usage_code": null, "usage_fields": []}
```

模型有回答但計費形狀無法辨識時，回傳和上面的回答相同，`sources` 仍是實際引用的來源。差別只有 `usage` 為 null、`usage_code` 為 `unknown_billing_component`，以及 `usage_fields` 列出安全欄位名，例如 `usage.priority_processing`。不把 usage 填成 0，也不丟掉答案。失敗只回 code 和這次的 request id，例如 `{"code": "rate_limit", "request_id": "ask-example"}`。

| code | HTTP | 意義 |
| --- | --- | --- |
| `invalid_request` | 422 | JSON、型別、商品、空白或超長問句 |
| `body_too_large` | 413 | body 超過 16KiB |
| `ask_not_configured` | 503 | 程序沒有資料庫或 key root 設定 |
| `dependency_unavailable` | 503 | 資料庫缺檔、缺 key、schema 或 digest 不合 |
| `evidence_unavailable` | 503 | 單一來源超界或文件版本衝突 |
| `embedding_unavailable` | 502 | 本機 embedding 失敗；不會再呼叫 Luna |
| `paid_blocked` | 503 | key 無法使用；沒有送出 |
| `insufficient_quota`、`rate_limit`、`provider_capacity` | 429 | provider 拒絕；立刻停止 |
| `logical_timeout`、`provider_transport_error` | 504 | 這一次逾時或連線失敗 |
| `output_refusal`、`invalid_model_schema`、`unknown_citation`、`provider_service_unavailable` | 502 | 沒有可回的答案 |

失敗不會附上原始 exception、provider body、key、本機路徑或半成品答案。沒有 retry、fallback 或自動重送；使用者再送是下一次獨立請求。

送去 Luna 的內容只有這次問題和被選中的 evidence。沒被選到的文件和整份資料庫不會送出。問句或文件裡的 URL、SQL、shell、HTML 和指令只當資料，不會改 endpoint、商品或 artifact scope。schema 與來源檢查不是已經證明的語意防注入。

知識文件仍是合成資料，這個端點沒有啟用正式知識，也沒有在這次變更對 Ollama 或 Luna 做 live 呼叫。mock 測試不需要真的 key：

```sh
uv run --locked pytest tests/test_ask.py
```

GitHub Actions 會使用 `uv.lock` 安裝依賴，在 Python 3.12 與 3.13 執行相同的格式、lint 與測試檢查，並以 Gitleaks 掃描 secrets。

目前包含最小 FastAPI app、健康檢查端點，以及以 TOML front matter 標記的少量合成知識文件範例（見 [`knowledge/README.md`](knowledge/README.md)）。文件讀取成功不等於人工審查、核准或啟用於線上 RAG。

本機 embedding 只做同步 adapter：既有 chunk 的原文送到固定的本機 Ollama 位址。呼叫者可以再把一份文件的來源與向量放進自己指定的 SQLite 檔，之後用同一個 key 讀回；store 不會自己呼叫模型。問句向量與離線 exact cosine 也在同一份說明。呼叫方式、HP-A mock 範例與保存格式見 [`knowledge/README.md`](knowledge/README.md)。這個 README 的範例沒有在本變更中對 Ollama 執行。

離線回答草稿把 caller 已經選好的 hits 收成有界 evidence request，再解析一份假的自然回答。固定 policy 與模型看得到的 JSON 資料分開。HP-A 的來源對照、mock HTTP 自然草稿與無證據 handoff 在 [`knowledge/README.md`](knowledge/README.md)。那是 mock，沒有呼叫 Luna，也還沒有核准啟用。

`hp_a_once` 仍是固定的 HP-A 相容入口，只讀呼叫者指定專案根的 key，準備同一題與既有保養 chunk，最多送一次。`POST /ask` 和它共用同一個 key-only 送出與 parser，差別是 `/ask` 接受一般的單次問句。兩條路徑都不讀 policy、不開舊 ledger，也不用費率、token 成本或預留當作准入。`max_output_tokens` 只限制這一次的輸出長度。429 會立刻停，並依 provider code 分成 `insufficient_quota` 或 `rate_limit`；認不出的 429 不是超額。認不出的計費欄位不會丟掉已通過模型、狀態與來源檢查的回答，也不會把 usage 填成 0。說明與邊界見 [`knowledge/README.md`](knowledge/README.md)。

## 知識維護（mock 切片，尚未核准啟用）

這一支把合成草稿、候選 release、mock 回歸與明確啟用接到現有的 `POST /ask`。Marcus 仍須以 exact bytes 核准文件與 release 之後，那些紀錄才能被消費。本目錄沒有已核准的正式紀錄。測試只在暫存 workspace 寫 reviewer 為 `marcus` 的夾具，那些夾具不是 demo 核准，也不能當作生成驗收。

`scan` 只描述 `knowledge/hp-a/headphone-care.md` 與 `knowledge/hp-b/connection-troubleshooting.md`。`knowledge/drafts/` 的充電 FAQ、退貨政策 v1（7 日）與 v2（14 日）留在草稿。`knowledge/fixtures/injection-note.md` 是未入選的注入範例。一份 release 要列出完整來源集合。

文件核准 schema 是 `synthetic-document-approval`，綁文件身份、路徑、檔案 hash 與正文 hash。release 核准 schema 是 `synthetic-release-approval`，綁 `release_id`、manifest、資料庫、regression、40 字元 code SHA 與 suite version。兩種紀錄不能互換。CLI 沒有 `approve`。消費時 reviewer 必須恰好是 `marcus`，多出來的欄位會被拒絕。這是同一 OS 帳號上的程序隔離。

`release_id` 是來源身份與 mock embedding 設定的 SHA-256，不含資料庫位元組。artifact key 也不含向量；啟用另外比對 manifest 裡的資料庫檔案 hash。mock digest 是字串 `customer-support-rag-mock-embed-v1` 的 SHA-256，manifest 的 transport 是 `mock`。已發布的 Ollama digest 仍是 `ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d`。這個 CLI 只建 mock 候選。

候選先寫進暫時目錄，再在真實的 `releases/` 裡 rename 成 `releases/<id>/`。`control/`、`document-approvals/`、`release-approvals/` 與 `releases/` 的任何一層若是 symlink，寫入會被拒絕，外面的目錄不會多出檔案。失敗會清掉暫時目錄，並留下原本的 `active.json`。啟用會取得 workspace 的 flock，核對已消費的 release 核准、三個檔案 hash，以及這份候選自己的 regression：`release_id`、manifest hash、資料庫 hash、`suite_version`、`rubric_version` 與 `code_sha` 都必須對上這份候選。別的候選的 regression 不能拿來啟用。還會核對資料庫裡的 key。複製既有向量前會核對來源 manifest、資料庫檔案 hash、mock digest 與文件身份，並把每個向量跟該 chunk 的 mock 向量比較；對不上就捨棄該來源、重新 embedding，不會沿用壞向量。`--expect` 必須是目前指標檔案的 SHA-256；第一次啟用用 `--expect-none`。預期不符就不會寫。`os.replace` 之前失敗時，舊指標還在，可以原命令重試。replace 已經成功、但指標目錄 fsync 失敗時，結果是 `activated_unconfirmed`（結束碼 0），並印出指標檔的 SHA-256。這時指標已經指向新 release，不能當成回滾。請先看 `status`；若已經是這份 release，不要再用舊的 `--expect` 重送。帶新的指標 hash 再執行一次 `activate` 會是 `noop`。候選 rename 成功之後的目錄 fsync 失敗仍會被略過。flock 是這台機器、同一個檔案系統上的建議鎖。作業系統若忽略 fsync，這段流程不能證明斷電安全。

```sh
uv run --locked python -m customer_support_rag.maintenance hashes \
  --root knowledge --path drafts/hp-a/charging-faq.md
```

子命令還有 `scan`、`consume-document`、`build`、`evaluate`、`consume-release`、`activate` 與 `status`。`evaluate` 要帶 `--code-sha`。`eval/dev-suite.json` 的 suite version 是 `dev-suite-v1`。檢索案例只在候選的來源路徑集合與該案例的 source set 相同時執行，否則記 `not_applicable`。生成案例的結果維持 `unverified`，`model_called` 為 false。激活會拒絕一份把生成結果寫成 pass 的 regression。字符 bigram baseline 用 Dice，平手時先比 artifact key，再比 chunk index，`top_k` 是 4。這份回歸記錄控制流程。sealed holdout 由 Marcus 另外準備。

設定 `CUSTOMER_SUPPORT_RAG_RELEASE_ROOT` 之後，`POST /ask` 只讀該 workspace 的 active snapshot。這時程序仍要有 `CUSTOMER_SUPPORT_RAG_KEY_ROOT`。指標壞掉回 `503` `release_unavailable`。未設定 release root 時，既有的資料庫與範例 artifact key 路徑維持原樣。

## 有界追問

`POST /assist` 與單次 `POST /ask` 分開。JSON 只有 `action`（`start`、`continue`、`reset`）、可省略的 `slot_token`、`product_id`、`question`、`steps_tried`。其他欄位，包括 role、history、release 與 attempt，都會被拒絕。

伺服器發出 32 個 hex 字元的 slot token，用來尋找這個程序裡的記憶。澄清最多兩次，再來是 `insufficient_information`。每個欄位最多 1000 個 Python 字元，問句與已試步驟合計最多 2000。商品只接受 `hp-a` 或 `hp-b`，不會從問句推斷。HTTP body 上限 16KiB。token 從建立起 900 秒有效，繼續對話不會延長。同時最多 32 個 slot；滿了回 `503` `slot_capacity`。未知或過期 token 回 `422` `invalid_slot`。只有 `reset` 會換成一張新的空 slot。這次回應仍是 `clarify`，但 attempt 是 1，並且算進最多兩次澄清。空的 `continue` 接著是 `clarify` attempt 2，再來是 `handoff` attempt 2，之後維持 handoff，不會出現第三次 `clarify`。reset 之後第一次 `continue` 若三個欄位都齊，仍會進入模型。`start` 不帶欄位仍是 `clarify` attempt 1、第二次 `clarify` attempt 2，然後 handoff。`start` 帶 token，或 `reset` 附帶欄位，都是 `invalid_request`。

三個欄位都齊了才會進入模型，而且只會一次。之後的 `continue` 是 `reset_required`，回應不含上一則回答。還在澄清時更換商品會清掉先前的問題與步驟，已用掉的澄清次數保留。已經呼叫過模型之後再換商品，同樣是 `reset_required`。錯誤不重試。`user_statements_unverified` 固定為 true。handoff 的 `summary` 是繁體中文，錯誤 code 是英文。

Host 與 Origin 規則和 `POST /ask` 相同。跨來源的 `POST /assist` 回 `422`，而且不會呼叫模型。這次變更沒有改 `web/`。

澄清一步的形狀：

```json
{
  "request_id": "assist-example",
  "status": "clarify",
  "slot_token": "slot-example",
  "attempt": 1,
  "missing": ["product_id"],
  "product_id": null,
  "question": "可以用水洗嗎？",
  "steps_tried": null,
  "summary": null,
  "user_statements_unverified": true,
  "answer": null,
  "reason": null,
  "sources": [],
  "usage": null,
  "usage_code": null,
  "usage_fields": []
}
```

欄位齊全且模型回了引用時，`status` 為 `answer`，`sources` 與 `POST /ask` 相同。模型轉人工時，`status` 為 `handoff`，`summary` 帶上該次 reason。缺欄位超過兩次時，reason 是 `insufficient_information`。
