本機設定與未執行的單次 smoke 命令見 [cloud setup](docs/cloud-setup.md)；填好設定不代表授權呼叫。

# Current D-010：cloud generation offline implementation

目前回答provider固定gpt-6-luna（Responses），Ollama只保留qwen3-embedding:0.6b；14B generation及舊效能紀錄僅供歷史，不作fallback或新provider驗收。Bing已授權本案固定模型的累計雲端預算；本輪實作仍只有mock測試，沒有paid integration或推論，後續由Calvin依review結果分階段派工。

產品根目錄`.env`只需`OPENAI_API_KEY`；已授權budget由gitignored runtime policy與durable ledger管理，使用者不需填其他欄位，不讀global環境或project.env。詳見[本機setup](docs/cloud-setup.md)。不要將key或financial cap貼在task/聊天/git。

以下既有PoC資料保留為歷史與功能說明；舊provider評測不能rebind到cloud，active仍須新的獨立核准。

---

# 合成客服 RAG 展示

Python／FastAPI、SQLite immutable snapshot、FTS5 與 Ollama 的 localhost PoC。只使用合成耳機與政策資料，產品 ID 為 hp-a／hp-b；介面品牌由 web/ 維護。無訂單、退款執行或外部客服通道。

目前 Develop 尚未驗收：corpus v2 已由 Marcus 核准，candidate 尚未取得獨立 release approval，active 維持 null；正常對話 API 因此回 503，不會假裝已生成有效客服答案。最新離線實測見 eval/，凍結資料與門檻不因結果改寫。

## 本機啟動

使用 Python 3.9 以上，在專案根目錄執行：

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/python -m scripts.serve --port 8768
```

serve 固定 127.0.0.1、單 worker，Origin allowlist 與指定 port 一致。8000 若已被其他服務占用，請指定空閒本機 port；不停止其他應用。一般網址使用實際 API；`?demo=1` 僅展示前端固定 fixture，不能當模型驗收。不可將本機無登入服務直接部署多人使用。

## 知識生命週期

```sh
.venv/bin/python -m scripts.knowledge validate
.venv/bin/python -m scripts.release build
.venv/bin/python -m scripts.release inspect
```

Markdown metadata 在 knowledge/manifest.json；核准紀錄在 knowledge/approvals/。文件 identity 綁 doc_id／revision、完整 metadata 與文字 hash，整份 corpus 再綁 canonical hash。版本、有效期間或任一內容改動均需 Marcus 重新審查。build 只產生 quarantine candidate，不改 active；相同 manifest/config 去重。完整原文 snapshot、heading-aware chunk 的絕對 offset／length／text hash 皆保存，讀取時驗證來源一致性。FTS 每個 eligible release 獨立建立統計，先按商品、locale、日期過濾再排名。

`python -m scripts.faq_draft --question '合成問題' --proposed-answer '待人工審查文字'` 僅在 eval/drafts/ 產生未索引草稿；不自動核准或修改 corpus。正式納入前需產生新文件／revision、核准與評測。

ready 需要完整 regression＋Marcus 獨立 holdout 的 dataset/gold/逐題結果與配置 hash；程式會重播 lexical 排名、重算分子／分母，拒絕任意 passed=true。接著 activate／rollback 還需 Marcus 的獨立 release approval，綁 release manifest、corpus approval、evaluation 與 holdout hash。CLI 參數請用 `python -m scripts.release --help`。不得自行製作真正核准紀錄；tests/ 中僅有明示的 synthetic unit fixtures。

切換採 SQLite transaction、expected_active CAS 與 event；磁碟／中斷不留半更新。撤回是跨 release 的 doc_id denylist，來源只回 redacted 狀態，rollback 與 replay 不可復活撤回文字。保留 snapshots，不自動刪除。

## 邊界與回答方式

Host／Origin allowlist、16KiB body、strict schema、2000字元 input；模型固定 loopback 與兩個完整 digest，無付費 fallback 或模型下載。embedding 1024維、truncate=false；context 4096、output 512、evidence 1600／history 600 皆以 UTF-8 bytes 作保守上界，超量拒絕。

模型使用短候選代號，server 驗證後映射成 release-bound 引用。回答採保守逐字引用，改寫或無法對應引用片段即 fail closed；這使 supported-claim 檢查可用 exact spans 重現，仍需 Marcus 獨立人工語意／安全 QA。缺資訊最多追問兩次，退款或無證據交人工摘要。server 保存最近三輪 slots，idle30分鐘、最多20完成 turns；相同 ID／payload 重播、異 payload 或處理中回409。

一個生成 slot、兩個等待工作；60秒生成 deadline／75秒整輪 deadline。已開始的上游在 timeout／斷線後保留 slot 到回應結束；尚未開始的逾時工作取消，避免事後背景生成。三次依賴失敗斷路30秒；讀取連線異常且無法確認上游結束時停止接納推論，需人工確認上游狀態後重啟服務。沒有自動重試生成。

## 驗證

```sh
.venv/bin/python -m pytest -q
node web/contract.test.cjs
.venv/bin/python -m scripts.verify_local
.venv/bin/python -m scripts.http_smoke
```

verify_local 使用已存在的固定 digest Docker gitleaks image，`--pull=never`，不安裝系統工具；產生 test XML／scan JSON 與配置 hash。scripts/pre-commit 可作本機 hook，CI 定義 test／secret-scan；配置與實際 CI 狀態分開，尚未 push 或跑遠端 CI。

離線評測需要 Marcus genuine corpus approval：`python -m scripts.evaluate_candidate --split dev|regression --corpus-approval <record>`。不可覆寫既有 checkpoint；有失敗 run 先明確封存再重跑。原始 dev40／regression60 保存在 eval/frozen/；目前使用已知公開的 public-v3 dev50／regression70，固定 hash、類別數及 gold 變更記錄在 eval/frozen-v3/。這不是新的盲測，數值門檻不變。private holdout 由 Marcus 保管及執行，不拿來調參，不放入公開展示或文章。

尚未取得 release approval，因此真 active API 回答、20輪 cold/warm HTTP 端到端、完整 UI／clipboard QA 仍待獨立 release 關卡。eval/model-20-rounds.json 是 adapter probe，不能替代端到端驗收。失敗紀錄與資源限制亦保留，不將低樣本 probe 當正式效能結論。

## 授權

專案碼 MIT，見 LICENSE。第三方實際安裝 metadata 與 LICENSE／NOTICE 文本在 eval/dependency-notices.json、eval/licenses/；模型不隨 repo 分發，見 NOTICE 與 eval/model-notices.json。

通用政策可從適用的 general evidence 回答；型號特定排障仍需確認型號。null／省略 product_id 保留已知型號，明確更換非 null 型號才重設相關 slots，新 session 為明確 reset。政策模式由模型選擇是否有支持與來源，再引用完整核准 block，保留資格與限制；unsupported 仍轉人工。品質拒絕與 transport／model identity 故障分開計數，不能把逐字支持當成獨立語意驗收。最新 QA 介面見 eval/evaluation-interface.md。

模型生命週期採 request-local `keep_alive=60s`，不改 Ollama 全域設定。離線 dense 評測先計算 query／文件 pool vectors，再確認卸載本批載入的 embedding，才進入 14B generation；結束且上游已完成時再卸載本批模型。這需要 CLI 明確傳入 `--exclusive-model-slot`，且 `/api/ps` 起始空載；既有同名模型也不推定屬於本 app，遇到外部 residency 或卸載未確認則拒絕繼續。`/api/ps` 不提供 request owner，獨占 slot 仍須由操作者協調，不能靠 metadata 證明同名模型沒有其他使用者。

服務選用 lexical retrieval，因此每個 served turn 不會為 query 額外載入 embedding；真實 embedding／dense 比較仍保留於離線知識評測。若將來選用 dense serving，phase 切換會增加 query embedding 與生成模型重載成本，需新配置與實測，不能沿用目前延遲。資源比較 CLI 為 `python -m scripts.resource_trial --help`，只跑固定公開 dev 子集、不是完整或盲測驗收。舊結果不可綁到新配置。
