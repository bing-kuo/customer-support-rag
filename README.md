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

同一 FastAPI 程序在 `GET /` 供應 repository 根目錄的 `web/index.html`，固定資產為 `GET /assets/style.css` 與 `GET /assets/app.js`。程式用套件位置定位 `web/`，請求不能指定 web 根目錄或 key 根目錄。沒有 `web/`、檔案不是一般檔、或名稱不在上述清單時回 JSON 404。未知路徑與未知 POST 維持 JSON／HTTP 錯誤，不會 fallback 成 HTML 首頁。暫時測試用的 web tree 不是產品 UI；目前 repository 還沒有 `web/`，這不是已完成的畫面。

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
