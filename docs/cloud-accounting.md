# Cloud accounting 修補與 unknown 邊界

首次 smoke 僅送出一次；當時留下 unknown，缺少可信 usage／latency 與精確 rejection code。已記錄 provider identity 不代表終結或帳務已確認。**實際 root cause 仍未知**，本次未重送、未 retrieve、未讀 key／私有 policy，也不釋放原 reservation。

已確認的獨立相容性缺陷：舊 parser 拒絕官方合法 cache-write breakdown，包括零值。依 [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) 的 Responses usage 形狀，現在明確接受 cached_tokens、cache_write_tokens 與 TTL 細分；這是公開 synthetic fixture 可重現的修補，不能倒推它必然造成第一次 smoke。

核心 input/output/total 必須是有界、非負的整數（bool／float／NaN／巨值拒絕），total 必須等於 input＋output。cached＋write 不得超出 input，reasoning 不得超出 output；TTL 細分存在時需完整且合計等於 write，不能重複計為額外 input。未提供或 null 的 optional breakdown 保留未知，不偽裝已確認為零；explicit component 值需是合法整數。未知費項／未知 keys 仍 fail closed。所有 input 按現有最壞上界估算，不因缺 breakdown 或 cache hit 給折扣；保守上界已涵蓋已知 cache-write 及 uplift，不要求使用者計算。

Accounting terminal 與 application quality 分開：可信終結及 usage 先原子 settle，再檢查 refusal／schema／citation／qualifier。incomplete、failed、cancelled 或 refusal 仍可能有已確認費用；不能用「沒有答案」推論沒有計費。usage 不可信、null／未知 tier、超 reservation、transport ambiguity 或 settlement commit failure 留 unknown；不採自動 retry／fallback。

新 diagnostics 是 private ledger 的有限 enum／數值 metadata：stage/code、HTTP status、safe provider status/tier、已知 usage 欄位名稱與合法 token counts、elapsed、terminal_verified／settled；request／response IDs 與 sanitized model 仍在對應 request row。未知欄位名稱不直接記錄，沒有 provider body、prompt、answer、key 或 financial values 進入一般輸出。正常 settlement 與完整 usage metadata 在同一 transaction 寫入。diagnostics 寫入失敗不掩蓋原錯誤或釋放 unknown；若送出前診斷存不下，停止 transport 並保守保留 marker。硬體／storage 失效時不能保證新增診斷成功，既有 durable sent 仍阻擋重啟後新 admission。首次缺失的 latency 不事後補造成量測。

`app.reconciliation.preview` 只是 pure synthetic dry-run，沒有 ledger path、SQL、network 或 apply 方法，永遠回報不 mutation、不授權真 reconciliation。比對同一 project/local/provider IDs、body/source/config/pricing/approval bindings、terminal time/model/tier、完整已知 component usage、reservation／ceiling，以及另供的獨立 review manifest hash。拒絕未知費項、篡改、缺 component、wrong IDs、超預留、ceiling 變更及非 unknown snapshot；重複或並行 preview 不改任何輸入。

一個 hash 或 reviewer 名字不能證明 provider authenticity：preview 只驗結構，明確回報 authenticity 尚未驗證，不能拿 synthetic fixture review 操作真實資料。真實 reconciliation 還缺同一 request 的可驗來源完整終結與 usage evidence，以及獨立人工 provenance review；聚合 usage、手填 tokens、餘額變化或截圖本身不足。不要求不存在的 dashboard 功能、不開 store=true、不為補證據偷偷付費或 retrieve。未有可信同筆 evidence 與另審操作程序前，原 unknown 與 paid block 維持；這不是 release／RAG 品質核准。

## 有條件的 permanent conservative-debit operator

Bing 已同意此方向：若獨立 review 能證明原請求已無在途工作，且原 reservation 涵蓋全部適用費項，將整份 reservation 永久視為消耗，總 cap 不變。此決定無需再重問，但不能略過前提證據。

離線實作為 `Ledger.conservative_debit(..., dry_run=True)`，沒有自動 CLI、網路或帳戶操作。synthetic tests 才會用 `dry_run=False`，真實 operation 仍須 Dana 審 bound／completion evidence 與 Calvin 另派。operator 輸入為精確原 request/body/pricing/budget/ceiling/reservation fingerprint、原 configuration、同筆 provider IDs、可驗終結及 fee-bound artifacts，另由獨立 review manifest 綁定 decision hash。reviewer 名稱、bool 或 hash 本身不會認證 provider；授權操作者必須先驗來源並將原 decision／review／artifacts 保存在私有 immutable evidence，不能拿測試 fixture 當真批准。

transition 在 BEGIN IMMEDIATE 中只新增 conservative_debits disposition 與永久消耗事件。原 requests.state 仍是 unknown，used_input/output/cost 保持 null，不捏造 actual usage、provider-settled、退款或退款後餘額。支出加總持續計原 res_cost 整份一次；獨立 disposition 的 res_cost 是 conservative debit，不是 observed charge。完全相同 decision 重播不重扣，衝突 decision 拒絕，storage failure rollback；原 ID 或原 body 即使改 ID 仍永久拒絕重送，晚到 settle 也不能把永久扣滿降低。

只有匹配全部原 reservation 欄位的 reviewed disposition 才能排除該筆 paid block，其他 unknown／sent／reserved 仍阻擋。新不同用途 request 仍照既有累計 strict cap 預留，等於邊界亦拒絕。換 key 不清帳；cap/budget binding 改變不允許新 admission，既有 migrate_policy 對這種 unknown 保守拒絕而不移除 disposition。這一版不擴充新 migration 或 billing 平台。

目前真實前提仍不成立：原 metadata 沒有保存 returned tier／terminal status／可驗 completion artifact，亦未保留 serialized request。以目前固定合成模板重建 logical body 可匹配原 body hash，只證明模板與記錄一致；不證明實際 tier、全部費項或 provider 無在途。官方 [Responses reference](https://developers.openai.com/api/reference/resources/responses/methods/create) 說明 response tier 可能不同於 requested tier；[Fast mode guide](https://developers.openai.com/api/docs/guides/fast-mode) 說明 response 的 tier 識別實際處理類別。因此不能只因 request 指定 default 就斷言 standard reservation 必然足夠，也不能用 worker 已退出推論終結。真實 unknown、reservation 與 paid block 未動；等待最小同筆可驗 completion／tier 或適用合約上界證據後再 review，不偷偷 retrieve 或再發 paid request。

## 最新使用者決定：接受原筆未知風險，繼續小量新請求

本段取代前段「必須取回原 terminal／tier 才能繼續」的限制。Bing 明確接受原筆可能已計費與上界不確定性，同意保留原紀錄、永久計滿原 reservation、禁止原請求重送，在原累計 cap 內繼續有界的新請求。Dashboard 279 tokens 僅是 USER-REPORTED 資訊，不是 authenticated provider usage、terminal、tier 或 settled invoice。

既有 operator 新增 `user_risk_acceptance` decision basis，不另建帳務系統。必須有明確 decision ID、`explicit_user_decision` source、Orca task reference、接受不確定性的聲明與完整保留原預留；`actual_usage=unknown`、`provider_verified=false` 固定。獨立 closeout 只核對這個已批准方向、exact decision hash 與原 request/config/reservation/cap binding，不再要求使用者找已不可得的原 payload／status。decision JSON 寫入既有 disposition metadata，279 不填到 used_*。相同 decision 冪等，原 cap／budget identity／舊 pricing hash／原 reserve 不變，不退款、重算、reset 或 top-up。

未來請求仍固定 `gpt-6-luna`、default tier、4096 總上界內的純文字、無 tools。依 [官方模型文件](https://developers.openai.com/api/docs/models/gpt-6-luna) 所列支援的短文字 tiers，未來 reserve 在既有 uncached＋最長 cache write＋regional bound 上納入最高已公告 Fast multiplier，覆蓋 default、Fast、Flex、Batch 而不借 cache 折扣。長 context 門檻遠高於本案 budget，其他 modality／tools 被禁止；unknown 新費項仍停止。新的 price-bound revision 進 config／pricing hash；這不改原筆 reservation，也不宣稱 supplier invoice 無任何風險。

本次只有 implementation／synthetic dry-run，真實 ledger 未變。下一次由 Calvin 在一次獨立 closeout 後派：載入私有 decision 與經 review 的 manifest，先 operator dry-run，再 apply 同一 decision 一次，確認原 usage unknown／原預留永久計滿／原 cap 不變，最後執行**不同 ID、不同 body、不同相容性驗證目的**的新單次命令：

```sh
.venv/bin/python -m scripts.cloud_smoke --execute-authorized-new-request
```

新 request 固定 ID `developer-smoke-cache-compatible-v2`，最多 input 3072/output 512/total 3584，沒有 batch 或 retry。任何已存在相同新 ID 不再送，原 `developer-smoke-one-shot-v1` 永久不用。可先執行 `--dry-run-new-request`，不讀設定、不 reserve、不連線。此文件不是本次實際執行紀錄；等待 Calvin 派工後才會有真 transition 與新 request。
