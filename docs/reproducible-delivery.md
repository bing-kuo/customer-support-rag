# 可重現交付與安全邊界

## 離線 rehearsal

執行 `uv run --locked python scripts/rehearse_delivery.py`，從 repository root 使用已安裝的工具。腳本只呼叫既有 maintenance CLI，所有 knowledge、workspace、approval fixtures 與 fake `.env` 都位於自動清除的 temporary directory。文件及 release approval 會清楚標成 rehearsal fixture，不代表 reviewer 的真實核准。Serving check 只在此 temporary release root 設定環境，未 activate 前確認 `/ask` 和完整欄位的 `/assist` 都回 `release_unavailable`；缺欄位 assist 先回 clarification，不呼叫 adapter。status 核准 active release 後，mock query embedding、mock generator 和標記為假答案的 response 再驗證 `/ask` 與 `/assist`。腳本最後從 policy v1 明確切換到 policy v2，重啟讀到同一 active release，兩個 API 的引用版本皆為 `2`，且舊 policy-v1 snapshot仍存在；沒有呼叫 Ollama 或 Luna。

腳本要求乾淨且已提交的 Git worktree，以當前 commit 作 evaluation `code_sha`。它複製合成知識，對 baseline、FAQ、return policy v1/v2 建立 mock 候選並 evaluate；重跑同一 build 必須是零 embedding no-op。generation 報告維持 `unverified`、`model_called=false`、usage 為未知。它只在暫存 workspace 消費假核准並 activate，確認 build 不會改 active pointer、failed pointer compare 保留舊 snapshot、重新讀取狀態可恢復，再用 status 確認過的 pointer hash 完成示範切換。

主腳本及 CLI 子程序都封鎖 socket 連線；`--embed` 明確使用 `mock`。Serving test 也只用 fake-key fixture、mock query embedding 與 deterministic mock generation，並輸出各自呼叫數。它不讀真 `.env`、資料庫、模型或 private evaluation；不呼叫 Ollama、Luna、網路服務或 provider usage，也不寫入使用者 release workspace。Rehearsal 只證明本機控制流程和 mock adapter 接線，不能取代真實 embedding、模型生成、holdout、approval 或發布驗收。

## 手動服務操作

本機頁面、`/health` 與 `/docs` 可以在未設定資料庫和 key 時啟動。Health 只檢查 API 程序。實際 `/ask` 需要操作者提供自己的 release/database 設定與 key root；有檢索候選時會做一次本機 Ollama query embedding 和一次 `gpt-6-luna` API request，無自動 retry。`/assist` 只有在欄位齊全時才進入模型。

### 從核准文件到 serving

以下是提供給具備既有核准與環境權限的操作者的命令範本。本次 P3 沒有執行這些真實建置、Ollama embedding、release 核准或 activation。P2 的真生成與 holdout 尚未完成驗收；使用 `--embed ollama` 之前，須依團隊流程另行確認是否可執行。不能拿離線 rehearsal 的假 approval 來執行本節命令。

先在 repository root 指定本機自己的知識、release workspace 與 key 目錄。路徑只是 placeholder，不會由 repository 建立或自動選取：

```sh
KNOWLEDGE_ROOT="$PWD/knowledge"
RELEASE_WORKSPACE="/path/to/your/private-release-workspace"
KEY_ROOT="/path/to/your/private-key-directory"
mkdir -m 700 -p "$RELEASE_WORKSPACE"
```

首次準備每份文件時，輸出 hashes 供 reviewer 核對 bytes，之後只消費 reviewer 交付且與該檔相符的 approval。範例以 baseline 兩份文件為例；新增 FAQ 或政策版本時，對該檔重做相同流程：

```sh
uv run --locked python -m customer_support_rag.maintenance hashes \
  --root "$KNOWLEDGE_ROOT" --path hp-a/headphone-care.md
uv run --locked python -m customer_support_rag.maintenance consume-document \
  --workspace "$RELEASE_WORKSPACE" --root "$KNOWLEDGE_ROOT" \
  --approval "/path/to/reviewer-provided/hp-a-headphone-care.json"

uv run --locked python -m customer_support_rag.maintenance hashes \
  --root "$KNOWLEDGE_ROOT" --path hp-b/connection-troubleshooting.md
uv run --locked python -m customer_support_rag.maintenance consume-document \
  --workspace "$RELEASE_WORKSPACE" --root "$KNOWLEDGE_ROOT" \
  --approval "/path/to/reviewer-provided/hp-b-connection-troubleshooting.json"
```

建立 baseline candidate 時明確指定來源。要產生可供真實檢索用的向量，`--embed ollama` 會使用本機已安裝且 digest 符合的 `qwen3-embedding:0.6b`，且可能發出多次本機 embedding request；此範例不會自動下載模型：

```sh
uv run --locked python -m customer_support_rag.maintenance build \
  --workspace "$RELEASE_WORKSPACE" --root "$KNOWLEDGE_ROOT" \
  --source hp-a/headphone-care.md \
  --source hp-b/connection-troubleshooting.md \
  --embed ollama
```

從 JSON 輸出抄錄實際 `release_id`，不要猜測或手造。以同樣來源再執行 build 可檢查 no-op/reuse：只有文件 identity、source bytes、transport、模型 digest 與其餘 embedding identity 全相符時，才會重用既有資料，輸出應顯示 `noop: true` 且不新增 document embedding。不同 identity 或內容變更會形成另一個 candidate，不覆寫舊 snapshot。

評估指定 candidate 的 retrieval suite。這裡明確使用 `--embed ollama`，只代表會將 suite 中的檢索問題送給本機 embedding adapter；generation cases 仍是 `unverified`，不會呼叫 Luna：

```sh
RELEASE_ID="copy-the-release_id-from-build-output"
CODE_SHA="$(git rev-parse HEAD)"
uv run --locked python -m customer_support_rag.maintenance evaluate \
  --workspace "$RELEASE_WORKSPACE" --root "$KNOWLEDGE_ROOT" \
  --release "$RELEASE_ID" --suite eval/dev-suite.json \
  --code-sha "$CODE_SHA" --embed ollama
```

若要增加 FAQ，先檢查與消費 FAQ 文件 approval，然後以完整 baseline + FAQ source set build。符合完整 identity 的既有 baseline chunks 可重用，其餘來源才需 embedding：

```sh
uv run --locked python -m customer_support_rag.maintenance hashes \
  --root "$KNOWLEDGE_ROOT" --path drafts/hp-a/charging-faq.md
uv run --locked python -m customer_support_rag.maintenance consume-document \
  --workspace "$RELEASE_WORKSPACE" --root "$KNOWLEDGE_ROOT" \
  --approval "/path/to/reviewer-provided/hp-a-charging-faq.json"
uv run --locked python -m customer_support_rag.maintenance build \
  --workspace "$RELEASE_WORKSPACE" --root "$KNOWLEDGE_ROOT" \
  --source hp-a/headphone-care.md \
  --source drafts/hp-a/charging-faq.md \
  --source hp-b/connection-troubleshooting.md \
  --embed ollama
```

Return policy v1 and v2 are mutually exclusive source sets. Obtain and consume each policy document approval, then build and evaluate each version as a separate candidate; use the reviewer-provided release approval for that specific candidate:

```sh
for POLICY_VERSION in v1 v2; do
  POLICY_SOURCE="drafts/hp-a/return-policy-${POLICY_VERSION}.md"
  uv run --locked python -m customer_support_rag.maintenance hashes \
    --root "$KNOWLEDGE_ROOT" --path "$POLICY_SOURCE"
  uv run --locked python -m customer_support_rag.maintenance consume-document \
    --workspace "$RELEASE_WORKSPACE" --root "$KNOWLEDGE_ROOT" \
    --approval "/path/to/reviewer-provided/return-policy-${POLICY_VERSION}.json"
  uv run --locked python -m customer_support_rag.maintenance build \
    --workspace "$RELEASE_WORKSPACE" --root "$KNOWLEDGE_ROOT" \
    --source hp-a/headphone-care.md \
    --source "$POLICY_SOURCE" \
    --source hp-b/connection-troubleshooting.md \
    --embed ollama
done
```

對 FAQ 與各 policy candidate 分別執行上一段 `evaluate` 範本，將 `RELEASE_ID` 換成該次 build 的實際輸出，並保留對應 regression bytes。若重跑 evaluation 改變 report hash，必須要求新的 release review/approval；不能把先前候選的 approval 套到新 report。

Policy v1 與 v2 必須各自使用 reviewer 已核准的版本與 source set，分別 build/evaluate，不能同時把兩個互斥政策版本放進一個同產品 candidate。每個 candidate evaluate 後，reviewer 必須針對該次 `release_id`、manifest/database/regression hashes 與 code SHA 另交 release approval；如果新 evaluate 改變 regression bytes，舊 release approval 不再適用。CLI 不建立 approval 檔：

```sh
uv run --locked python -m customer_support_rag.maintenance consume-release \
  --workspace "$RELEASE_WORKSPACE" \
  --approval "/path/to/reviewer-provided/exact-release-approval.json"
uv run --locked python -m customer_support_rag.maintenance status \
  --workspace "$RELEASE_WORKSPACE"
```

首次 activate 前 `status` 應顯示尚未 active，然後使用 `--expect-none`。更新已有 active candidate 前，先從目前 `active.json` bytes 算 SHA-256，帶入 `--expect`；成功後再用 `status` 核對 `release_id`、embedding identity 及 hashes：

```sh
uv run --locked python -m customer_support_rag.maintenance activate \
  --workspace "$RELEASE_WORKSPACE" \
  --release "$RELEASE_ID" --expect-none

# 後續更新時先算現有 pointer hash，再執行：
NEXT_RELEASE_ID="copy-the-new-approved-release_id-from-build-output"
EXPECTED_ACTIVE="$(uv run --locked python -c 'import hashlib,pathlib,sys; print(hashlib.sha256((pathlib.Path(sys.argv[1]) / "active.json").read_bytes()).hexdigest())' "$RELEASE_WORKSPACE")"
uv run --locked python -m customer_support_rag.maintenance activate \
  --workspace "$RELEASE_WORKSPACE" \
  --release "$NEXT_RELEASE_ID" --expect "$EXPECTED_ACTIVE"
uv run --locked python -m customer_support_rag.maintenance status \
  --workspace "$RELEASE_WORKSPACE"
```

狀態確認後才能用該 workspace 啟動 serving；`CUSTOMER_SUPPORT_RAG_RELEASE_ROOT` 指向 workspace 本身，服務從 `active.json` 讀快照，不需要舊 database 變數。key 由另一個只含 `.env` 的 key root 提供：

```sh
unset CUSTOMER_SUPPORT_RAG_DATABASE
export CUSTOMER_SUPPORT_RAG_RELEASE_ROOT="$RELEASE_WORKSPACE"
export CUSTOMER_SUPPORT_RAG_KEY_ROOT="$KEY_ROOT"
uv run --locked uvicorn customer_support_rag.api:app \
  --host 127.0.0.1 --port 8000 --no-proxy-headers
```

此時 `/health` 仍只代表程序可回應。要讓 `/ask` 使用 query embedding，並在有檢索證據時呼叫 Luna，必須等候選、生成語意、holdout 及必要 release review 都已完成驗收並取得明確執行授權；本指南與 mock rehearsal 都沒有執行該問答。

維護命令可在具備核准權限的真實環境中手動執行：

```sh
uv run --locked python -m customer_support_rag.maintenance status \
  --workspace /path/to/your/authorized-release-workspace
uv run --locked python -m customer_support_rag.maintenance consume-release \
  --workspace /path/to/your/authorized-release-workspace \
  --approval /path/to/your/reviewed-release-approval.json
uv run --locked python -m customer_support_rag.maintenance activate \
  --workspace /path/to/your/authorized-release-workspace \
  --release YOUR_RELEASE_ID --expect CURRENT_ACTIVE_JSON_SHA256
```

命令需由操作者替換範例路徑、release ID 與當前 pointer hash；approval 必須來自實際 review，不能使用 rehearsal fixtures。指標切換前先 `status`。遇到 `activated_unconfirmed`，先重新 `status`；若已指向目標 release，不要重播舊 `--expect`。失敗 compare 不會換 pointer。舊 snapshot 保留，沒有清除步驟。

## 問答錯誤與恢復

- Missing key 或 key permission 不合時，確認 `CUSTOMER_SUPPORT_RAG_KEY_ROOT/.env` 只含 `OPENAI_API_KEY` 且 mode 為 `0600`。不要將 key 貼入 issue、log 或 shell history。
- `dependency_unavailable`、`release_unavailable` 代表 local DB／snapshot 等依賴未能讀取。先確認程序設定與 `status`；不要當成 `no_evidence`。
- `429` 時停止。`insufficient_quota`、`rate_limit` 和 `provider_capacity` 不會觸發自動 retry；先查 provider 狀態再決定是否發起新的請求。
- timeout、client cancel 與 connection error 無法證明上游未執行、停止或未計費。查 provider usage/status 後再處理未知結果，不直接重送。
- Answer 可在 `usage=null` 且 `usage_code` 有值時回傳。未知 usage 不等於零；保存 request ID 和安全錯誤碼，向 provider 核對後再決定下一步。
- 不刪除舊 release snapshot。狀態未知時先執行 `status`，不以舊 expected pointer 重播 activate。

## 自然草稿契約與人工判讀（D-018 方案 A）

內部 instruction 版本為 `grounded-draft-v4`，policy UTF-8 SHA-256 為 `4d1906435bcc03df1d9f595ccc7559d0c6d923dbd748cfaad927c9297bd4f2a5`；canonical schema 是 677 bytes，SHA-256 `3bb66f4947eb708ff91476ebb60fc7542703bd112eb79c9941dc154f2e20f341`。公開 `AskResponse`、`CitedSource`、`/assist` 與 web API 形狀不變。Responses schema 仍是 flat object：root `status`、`answer`、`reason`、`claims`；每個 claim 只有 answer/source 字元 offsets 與 source ID，沒有 `required_limits`、`required_handoff` 或 `limit_ids`。嚴格 parser 拒絕額外欄位、未知來源、無效或不完整 span、未覆蓋的 answer，以及不一致的 handoff 欄位；它不做自然語言蘊含判斷。

模型 instruction 要求自然改寫，保留本題相關的條件、禁止、必做與停止指示，允許語意等價說法，不能新增或反轉許可；證據不足、衝突或只有人工可完成的處置時要求 handoff。這是生成指令，不是 schema 或本機 parser 能保證的規則。草稿需要客服打開實際來源核對；現有 UI 沒有強制人工放行，不能稱為無人監督服務。

### 程式可驗與需人工判讀

| 檢查 | 程式可驗 | 不可由 span／關鍵字自動保證 |
| --- | --- | --- |
| 來源 | artifact、商品、版本、正文 hash、chunk offsets 與原始 body 對得上 | 檢索是否找到了跨文件的所有相關內容 |
| 引用 | citation ID 必須屬於本次完整 evidence；claim span 在 answer 與來源範圍內；claims 完整覆蓋 answer | span 的內容是否真的支持該 claim、否定／條件是否被解反 |
| 限制 | 從同一已命中文件提供全部非空原始 chunks，不挑 marker 句、不截斷；整包需在原 4 evidence、1200 字元與 wire limits 內 | 必要限制是否適用本問題、自然改寫是否保留其意思 |
| 處置 | answer/handoff JSON 的欄位組合、reason enum 與 schema | 是否應交人工、同義異常、否定條件及可選聯絡語句的意圖 |

`/ask` 僅對已由原檢索選中的文件擴展完整 chunk bundle，不宣稱找到所有跨文件依賴。store 回讀的 artifact／product／version／body hash 需一致，並以相同 chunker 驗證所有原始 chunks、順序、文字與 offset。若 bundle 缺段、身分不符、超過 4 個 chunks、1200 字元、模型資料或 wire 上限，會在呼叫 Luna 前 fail closed；不提高上限、不靜默省略、不截句，也不以 marker 猜限制無關。這種容量 handoff 與語意判定 handoff 不同；UI 的現有 reason enum 不提供更細的使用者解釋。

### 全新天文館 mock 與人工 rubric

synthetic source A：「終端只使用公開演示目錄；不得輸入訪客姓名；不得更改帳號權限。」source B：「目錄每週二更新；權限異常由館員確認。」以下例子只示範 parser 邊界和人工 rubric，不是模型能力結果：

- **結構合法、語意錯誤：**回答「目錄每週二更新。」卻引用只寫終端位置的 A，或對「不得輸入訪客姓名」回答「可以輸入」。span 合法且 parser 會接受；rubric 分別判「claim support fail」與「極性／限制 fail」。不可將 parser 綠燈算成 D24／D25 通過。
- **人工處置：**「權限異常要怎麼處理？」若模型回 `status=answer`，即使寫出館員資訊也由 rubric 判 status/意圖不符；依語意等價方式回答館員處置的 `status=handoff` 可供人工核查。單純問「目錄哪天更新？」時，source B 的「如需更多資訊可聯絡館員」屬可選語句，不應只因關鍵字自動 handoff。
- **自然答案：**`「目錄每週二更新。」` 配 source B 是正常 mock 結構案例；實際語意正確性仍依原文人工確認。
- **D25 限制：**來源寫「資料錯置時，立即停止輸入，交館員確認；請勿提供證件號碼。」回答漏掉停止／隱私限制或反向要求輸入，即人工 fail；不要求對無關的日期問題逐字重述整份來源。

這輪從舊 parser 移除 marker-based 限制覆蓋、模型自報 limit IDs 與 bigram-based 人工分支。固定 instruction 保留生成要求，但不以模型自稱已涵蓋作為證明。測試分開記錄自動結構檢查與作者設定的 synthetic 人工 rubric；後者不是獨立 reviewer 對真生成的品質驗收。真生成改善尚未驗證，D24–D26 的語意品質不能標 pass。P2 acceptance-open/security hold、候選 build SHA `0cb3a9a`、09e 歷史評估及其錯誤／污染狀態均未改寫、重算或啟用。

固定 HP-A wire fixture 在 v4 policy/schema 下為 2,082 bytes；policy 623、data 377、schema 677、framing 405 bytes。模型輸入保留 `question`、`product_id` 與完整 `evidence`；`prompt_revision` 仍由程式在送出前驗證，但不再當作模型資料傳送。byte-based reservation headroom 為 478 units。policy-v2 合成問題的完整政策及保養文件共 181 字元；省略這個內部版本標籤後 wire 為 2,531 bytes，含既有 1,024-unit 預留共 3,555／3,584，餘 29 units。相稱單元回歸已驗證沒有截斷或省略 evidence；clean-HEAD 完整 rehearsal 尚待核准提交後重跑。較長 evidence 仍會依原界線拒絕或在證據準備階段 fail closed；沒有提高預留或自動截斷。
