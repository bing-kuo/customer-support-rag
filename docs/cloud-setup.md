# 雲端生成：只需專案 key

在專案根目錄建立或編輯 `.env`，只需要：

```text
OPENAI_API_KEY=[本案專用 API key]
```

不覆蓋檔案、不把 key 貼到聊天。舊 `CLOUD_*` 欄位不用填，runtime 忽略它們；不使用 global environment、Codex OAuth 或其他專案 key，也不能用 `.env` 改 model／endpoint。固定模型是 `gpt-6-luna`，Ollama 只負責既有 embedding。

在產品根目錄執行：

```sh
.venv/bin/python -m scripts.cloud_setup
.venv/bin/python -m scripts.cloud_setup --diagnose
```

這兩個命令只檢查本機格式與私有 policy，不連線、不讀帳務 ledger、不 reserve，不驗 key 是否可用。一般輸出為布林 status；diagnose 只回固定錯誤分類與 `OPENAI_API_KEY` 欄位名。若回 policy 錯誤，交 Ethan 處理，不需要使用者填技術欄位、手算價格或額外批准 booleans。

Bing 已授權的本案累計上限只存放 gitignored `runtime/cloud-policy.json`，由程式維護，使用者不需編輯。spent／reserved／unknown 永久保留於 `runtime/cloud-budget.sqlite3`；換 key、重啟、換 policy 或 source config 不會歸零。達上限前停止，不自動充值、放大 cap、換模型或重試。未知費用保留最大預留並停止新請求；不可以刪 ledger 重試。舊 ledger 只有在無 pending 且 schema／usage 相容時才能明確離線遷移；否則回安全的 reconciliation／schema 狀態。

費率依 [官方 model](https://developers.openai.com/api/docs/models/gpt-6-luna) 與 [官方 pricing](https://developers.openai.com/api/docs/pricing) 於本次實作獨立查閱。保守輸入預留涵蓋 uncached 加長期 cache write，再納入已公告 regional uplift；輸出亦留 uplift，不採折扣。固定 `service_tier=default`，只送文字、無 tools，拒絕非 default returned tier／未知 usage。pricing 有效性是程式維護且有期限，不要求使用者填寫；到期須重新查閱並 review。供應商將來改價、帳務差異或非預期費項仍可能超出本機估計；local guard 不是 provider invoice 硬上限。

此實作任務沒有 paid call。Dana review 後由 Calvin 派初始單次 smoke，再依同一累計預算分階段測試，無需反覆向使用者索取技術欄位或相同 cap 授權。每一階段仍須符合 task／command 範圍，不因 checker 通過自動啟動 batch／serve／holdout。

安全結構檢查（不讀 key／policy）：

```sh
.venv/bin/python -m scripts.cloud_smoke --dry-run
```

未在本次執行的初始單次命令（等待 Dana review 及 Calvin 派工）：

```sh
.venv/bin/python -m scripts.cloud_smoke --execute-authorized-new-request
```

原 smoke ID 永久禁止重送；新相容性驗證使用不同固定 ID／body，仍只有一個合成問題／證據、空 history，input 上界 3072、output 512、total 3584；單次入口沒有 loop、batch、自動 retry 或後續執行。固定 request ID 由 durable ledger 防重送，不需要 active release。admission／pre-send 重讀原設定，任何 policy 或 key 的途中變動都停止；未送出只釋放自己的 reservation，不清除其他 sent／unknown。成功只輸出 token usage，失敗只回 safe stopped，不輸出 provider body。

掃描包含公開 source、合成 knowledge/documents 與 docs/licenses/notices；排除 .env、runtime/private policy、ledger、knowledge approvals、所有 eval/private raw、symlinks 和 caches。不宣稱整台機器或私有 evidence 都已掃過。
