# 可重現交付與安全邊界

## 離線 rehearsal

執行 `uv run --locked python scripts/rehearse_delivery.py`，從 repository root 使用已安裝的工具。腳本只呼叫既有 maintenance CLI，所有 knowledge、workspace、approval fixtures 與 fake `.env` 都位於自動清除的 temporary directory。文件及 release approval 會清楚標成 rehearsal fixture，不代表 reviewer 的真實核准。

腳本要求乾淨且已提交的 Git worktree，以當前 commit 作 evaluation `code_sha`。它複製合成知識，對 baseline、FAQ、return policy v1/v2 建立 mock 候選並 evaluate；重跑同一 build 必須是零 embedding no-op。generation 報告維持 `unverified`、`model_called=false`、usage 為未知。它只在暫存 workspace 消費假核准並 activate，確認 build 不會改 active pointer、failed pointer compare 保留舊 snapshot、重新讀取狀態可恢復，再用 status 確認過的 pointer hash 完成示範切換。

主腳本及 CLI 子程序都封鎖 socket 連線；`--embed` 明確使用 `mock`。不讀真 `.env`、資料庫、模型或 private evaluation；不呼叫 Ollama、Luna、網路服務或 provider usage，也不寫入使用者 release workspace。rehearsal 的輸出只證明本機控制流程與 mock fixtures，不能取代真實 embedding、模型生成、holdout、approval 或發布驗收。

## 手動服務操作

本機頁面、`/health` 與 `/docs` 可以在未設定資料庫和 key 時啟動。Health 只檢查 API 程序。實際 `/ask` 需要操作者提供自己的 release/database 設定與 key root；有檢索候選時會做一次本機 Ollama query embedding 和一次 `gpt-6-luna` API request，無自動 retry。`/assist` 只有在欄位齊全時才進入模型。

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

## 生成輸出契約版本

P3 request-side handoff/answer 指示版本為 `grounded-draft-v2`，固定 policy UTF-8 SHA-256 為 `e5d1bb83e6b8fd406b0bfe6af4d8a7879d3cc4473997702bfda22f970153f956`。Flat response schema description 標示 `grounded-draft-schema-v2`；以 UTF-8、`sort_keys=true`、`separators=(',', ':')` 序列化後 679 bytes，SHA-256 為 `875d3f42fa7bbcd8734b064b96ef984a5adc8d167527d099878683f1cb092ae7`。Structured Outputs schema 保持 root object、四個 required 欄位及 `additionalProperties=false`；欄位 description 只是指示，沒有跨欄位 schema constraint。Parser 是強制拒絕矛盾 handoff、空答案、錯誤 reason、空／重複／未知 source ID 的最後界線；拒絕後不修補、不 retry。

此版本變更只影響 P3 新請求。它不改寫任何舊候選、舊 regression 或既有生成結論。未來若用此版本對既有 candidate 做已授權評測，evaluation sidecar 應另外連結 candidate build code SHA、manifest/DB hashes、evaluation code SHA/tree、suite/rubric、prompt/schema revision 或 hash，以及 request/response artifact hashes；不可重算既有向量後冒充相同 candidate。

固定 HP-A wire fixture 現為 2,488 bytes；沿用既有 byte-based reservation 後 headroom 是 72 units。較長的 question/evidence 仍依既有界線拒絕，沒有提高預留或自動截斷；該拒絕不是模型驗收。
