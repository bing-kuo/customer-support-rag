# Source-ID selection 修補與小量重測

先前 dev-public-v2-07 的 hp-a 前置回答遭 non_verbatim_sentence 拒絕，hp-b main 尚未送出。被拒原文未保留，確切原因仍未知；不能當成 harmless paraphrase，亦沒有降低逐句、引用、資格或 critical 要求。

一般程序現在使用 strict SourceSelection，policy 使用原 PolicySelection；兩者 wire JSON 都只有 answerable／source_ids。模型仍讀當輪問題、bounded history 與動態 eligible evidence，負責選出最小的相關 source IDs 或明確不支持。沒有關鍵字到答案的靜態 mapping，也不固定選第一段。server 只渲染所選當輪已核准 source 的完整文字，不接受模型新增 answer/prose/product 欄位。錯 alias／舊產品 source ID、缺 critical source、偽造文字一律拒絕。

原 service intent、日期／產品 eligible pool、模型缺失 clarification、source 撤回檢查、人工資格限制及完整 stop-use/contact 義務保持。明確 hp-a→hp-b 時清除舊 slots/history，null product 則沿用已確認產品；兩種路徑用不同3秒盒按鈕／5秒兩耳觸控的核准內容做 mock 驗證。critical 安全仍是原確定性路徑，不依賴模型是否選中。

這是 **model selection + server render**，不是自由生成自然語句的品質通過。metrics 的 model_output_kind=source_selection、answer_origin=server_approved_text 與 render_mode 明確區分 policy/procedural quote。整個 block 引用保留限制，但可能附上來源中對當前問題不必要的資格說明；模型也可能在多個 eligible block 間選得不夠相關，仍需獨立語意 review。source ID 合法不能自動證明所有相關性。

## 受限 rejection evidence

一般 logger／ledger 只記有限 code、candidate aliases、selection render mode，沒有 rejected text。只有明確 PUBLIC SYNTHETIC manifest 的 public_eval RetestAdapter 會 opt-in PublicSyntheticCapture；serve、holdout 與其他 scope 不擷取。每筆最多4096 bytes，rejected text 本身最多2048 bytes，只含當輪 evidence／output，不含 prompt、provider headers、key 或 environment。已知 key／credential pattern／內部路徑或超大輸出會抑制 raw，只留安全 metadata。

私有路徑是 gitignored runtime/public-synthetic-eval-rejections，目錄700／檔案600；不進一般 log／public result／web。這是診斷 capture 的邊界，不是任意 PII 自動去識別保證，前提仍是核准純合成 input/evidence。既有 raw 不可恢復，未來檔案僅供有權 review 者比對、不可冒稱當前已捕獲真 provider 拒絕。

## 已凍結的未執行重測

`python -m scripts.source_selection_retest` 僅 freeze plan，不讀 key、不 reserve、不連線。manifest 在 eval/source-selection-retest-v1/manifest.json，原 frozen-v4 questions/gold 不改，known-not-blind。

未來由 Calvin 另派新 stage，使用 RetestAdapter（hard maximum FOUR admission attempts，包含錯誤、無retry）、此 manifest 與新 candidate/config/check binding，不能 resume 舊 public-dev10：

- dev-public-v2-07：真正 hp-a prior → hp-b main，最多兩次；prior 拒絕就停，不跳過 prior 假裝 switch 通過。
- dev-public-v2-02：一般保固及全部資格限制，最多一次。
- dev-normal-07：完整安全指令 negative control，預期零模型。
- dev-no_answer-00：退款 action handoff negative control，預期零模型。
- dev-injection-00：忽略惡意指令且只選正確配對 evidence，最多一次。

mock 已涵蓋 null 沿用、wrong-product/source、added sentence、缺資格與 critical 拒絕，不為這些拒絕刻意增加 paid calls。後續仍先檢查同一私有 cap／spent／reserved，可支付整段最壞預留；每一筆用原 durable guard，不改模型、費率、budget identity 或 user approval。這輪無 paid/Ollama call，active null。

先前 public raw scan exit1 與 Marcus 對 exact tests/test_key_only.py source SHA 的單筆誤報簽結原樣保留；不加 broad ignore，也不把 exit1 改稱機械通過。
