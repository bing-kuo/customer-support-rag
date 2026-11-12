(() => {
  "use strict";

  const form = document.querySelector("#ask-form");
  const productSelect = document.querySelector("#product");
  const questionInput = document.querySelector("#question");
  const stepsInput = document.querySelector("#steps-tried");
  const stepsField = document.querySelector("#steps-field");
  const assistStatus = document.querySelector("#assist-status");
  const characterCount = document.querySelector("#character-count");
  const submitButton = document.querySelector("#submit-button");
  const newQuestionButton = document.querySelector("#new-question");
  const result = document.querySelector("#result");
  const resultCaption = document.querySelector("#result-caption");
  const resultContent = document.querySelector("#result-content");
  const sourceDialog = document.querySelector("#source-dialog");
  const sourceMetadata = document.querySelector("#source-metadata");
  const sourceText = document.querySelector("#source-text");
  const toast = document.querySelector("#toast");
  const safeCodes = {
    invalid_request: "資料格式不符合要求，請檢查後再試。",
    invalid_slot: "這段對話已失效，請開始新問題。",
    body_too_large: "這次內容過長，請縮短後再試。",
    request_too_large: "這次內容過長，請縮短後再試。",
    slot_capacity: "目前本機對話已滿，稍後再試。",
    ask_not_configured: "問答服務目前尚未設定完成。",
    dependency_unavailable: "問答服務暫時無法使用。",
    evidence_unavailable: "目前無法讀取回答依據。",
    paid_blocked: "目前無法完成這次回答。",
    policy_rejected: "目前無法完成這次回答。",
    runtime_unavailable: "問答服務暫時無法使用。",
    embedding_unavailable: "目前無法搜尋回答依據。",
    insufficient_quota: "服務額度不足，這次沒有完成回答。",
    rate_limit: "服務目前忙碌，這次沒有完成回答。",
    provider_capacity: "服務目前忙碌，這次沒有完成回答。",
    logical_timeout: "等待服務回應逾時，這次沒有完成回答。",
    provider_transport_error: "連線等待逾時，這次沒有完成回答。",
  };
  let busy = false;
  let slotToken = null;
  let serverFields = null;
  let previousProductId = productSelect.value;
  let previousDialogFocus = null;
  let toastTimer = null;

  function element(tag, className, value) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (value !== undefined) node.textContent = value;
    return node;
  }

  function characterLength(value) { return Array.from(value).length; }
  function guided() { return form.elements.mode.value === "guided"; }

  function updateCount() {
    const count = characterLength(questionInput.value);
    characterCount.textContent = `${count} / 1000`;
    characterCount.classList.toggle("over-limit", count > 1000);
  }

  function updateMode() {
    const isGuided = guided();
    stepsField.hidden = !isGuided;
    productSelect.required = !isGuided;
    questionInput.required = !isGuided;
    document.querySelector('label[for="question"]').textContent = isGuided
      ? "顧客確認的問題"
      : "顧客問題";
    questionInput.placeholder = isGuided ? "先寫下顧客原本提出的問題" : "例如：HP-A 耳機可以用水洗嗎？";
    document.querySelector("#question-help").textContent = isGuided
      ? "最多 1000 字。此內容是未驗證的使用者陳述。"
      : "最多 1000 字。送出後只會詢問一次，不會自動重試。";
    submitButton.querySelector(".button-label").textContent = isGuided
      ? (slotToken ? "補充資料" : "開始逐步整理")
      : "整理回答";
    newQuestionButton.hidden = !isGuided || !slotToken;
  }

  function setBusy(value) {
    busy = value;
    submitButton.disabled = value;
    productSelect.disabled = value;
    questionInput.disabled = value;
    stepsInput.disabled = value;
    document.querySelectorAll('input[name="mode"]').forEach((input) => { input.disabled = value; });
    result.setAttribute("aria-busy", String(value));
    if (value) submitButton.querySelector(".button-label").textContent = "整理中…";
  }

  function currentAssistFields() {
    return {
      product_id: productSelect.value || null,
      question: questionInput.value.trim() || null,
      steps_tried: stepsInput.value.trim() || null,
    };
  }

  function needsServerReset(fields) {
    if (!slotToken || !serverFields) return false;
    return serverFields.product_id && serverFields.product_id !== fields.product_id;
  }

  function startResult(kind, label, caption) {
    resultCaption.textContent = caption;
    resultContent.replaceChildren();
    resultContent.append(element("span", `result-badge ${kind}`, label));
  }

  function addUsage(payload, generated) {
    const box = element("section", "usage-box");
    box.append(element("p", "usage-title", payload.usage ? "本次用量" : "本次用量未知"));
    if (payload.usage) {
      const usage = payload.usage;
      box.append(element("p", "usage-detail", `輸入 ${usage.input_tokens} · 輸出 ${usage.output_tokens} · 合計 ${usage.total_tokens} tokens`));
    } else if (payload.usage_code) {
      box.append(element("p", "usage-detail", `服務未提供可確認的用量（${payload.usage_code}）。`));
    } else {
      box.append(element("p", "usage-detail", generated === false
        ? "本步驟沒有呼叫生成模型。"
        : "服務未回傳可確認的用量。"));
    }
    resultContent.append(box);
  }

  function addRequestId(payload) {
    if (typeof payload.request_id === "string") resultContent.append(element("p", "request-id", `請求編號：${payload.request_id}`));
  }

  function openSource(source, button) {
    previousDialogFocus = button;
    sourceMetadata.replaceChildren();
    const fields = [["標題", source.title], ["版本", source.version], ["商品", source.product_id],
      ["文件 ID", source.document_id], ["Chunk", String(source.chunk_index)],
      ["段落", source.heading], ["原文字元位置", `${source.char_start}–${source.char_end}`]];
    for (const [label, value] of fields) {
      const pair = element("div");
      pair.append(element("dt", "", label), element("dd", "", value));
      sourceMetadata.append(pair);
    }
    sourceText.textContent = source.text;
    sourceDialog.showModal();
    document.querySelector("#close-dialog").focus();
  }

  function validSource(source) {
    return source && ["source_id", "artifact_key", "document_id", "version", "title", "text"].every((key) =>
      typeof source[key] === "string" && source[key].trim().length > 0) &&
      typeof source.heading === "string" && ["hp-a", "hp-b"].includes(source.product_id) &&
      ["chunk_index", "char_start", "char_end"].every((key) => Number.isInteger(source[key])) &&
      source.chunk_index >= 0 && source.char_start >= 0 && source.char_end > source.char_start;
  }

  function addSources(sources) {
    if (!sources.length) return;
    resultContent.append(element("div", "result-divider"), element("h3", "subheading", "回答引用的原文"));
    const list = element("div", "source-list");
    for (const source of sources) {
      const button = element("button", "source-button");
      button.type = "button";
      const text = element("span");
      text.append(element("span", "source-title", source.title), element("span", "source-meta", `${source.heading} · 第 ${source.chunk_index} 段 · 版本 ${source.version}`));
      button.append(text, element("span", "source-open", "查看原文 ↗"));
      button.addEventListener("click", () => openSource(source, button));
      list.append(button);
    }
    resultContent.append(list);
  }

  function validUsage(usage) {
    const required = ["input_tokens", "output_tokens", "total_tokens"];
    const optional = ["cached_tokens", "cache_write_tokens", "ttl_30m_tokens", "ttl_12h_tokens", "reasoning_tokens"];
    return usage && typeof usage === "object" && required.every((key) => Number.isSafeInteger(usage[key]) && usage[key] >= 0) &&
      optional.every((key) => usage[key] === null || (Number.isSafeInteger(usage[key]) && usage[key] >= 0));
  }

  function validResponse(payload) {
    if (!payload || typeof payload !== "object" || !Array.isArray(payload.sources) || !(payload.usage === null || validUsage(payload.usage))) return false;
    if (typeof payload.usage_code !== "string" && payload.usage_code !== null) return false;
    if (!Array.isArray(payload.usage_fields)) return false;
    if (payload.status === "answer") return typeof payload.answer === "string" && payload.answer.trim().length > 0 && payload.sources.length > 0 && payload.sources.every(validSource);
    if (payload.status === "clarify") return typeof payload.slot_token === "string" && Array.isArray(payload.missing) && payload.answer === null && payload.sources.length === 0;
    if (payload.status === "handoff") return payload.answer === null && payload.sources.length === 0 &&
      (typeof payload.summary === "string" || ["no_evidence", "conflicting_evidence", "refund_request"].includes(payload.reason));
    return false;
  }

  function showAnswer(payload) {
    startResult("", "回答草稿", "請先核對回答與引用原文，再決定是否使用。希望更換商品或開始新題，請開新問題以清除本題資料。");
    resultContent.append(element("p", "answer-text", payload.answer));
    addSources(payload.sources);
    resultContent.append(element("div", "result-divider"));
    addUsage(payload, true);
    addRequestId(payload);
  }

  function renderSummary(payload) {
    const summary = payload.summary || "商品：未提供\n顧客問題（未驗證的使用者陳述）：未提供\n已試步驟：未提供\n缺少資訊：未提供";
    const note = element("div", "notice-box");
    note.append(element("p", "notice-title", "提供給客服同仁的摘要（使用者陳述未經驗證）"), element("p", "notice-detail", summary));
    resultContent.append(note);
    const copy = element("button", "copy-button", "複製摘要");
    copy.type = "button";
    copy.addEventListener("click", () => copySummary(summary));
    resultContent.append(copy);
  }

  function showHandoff(payload) {
    const resetNeeded = payload.reason === "reset_required";
    startResult("handoff", "建議轉由專人處理", resetNeeded ? "更換商品或本題已完成，請明確開啟新問題。" : "目前欄位仍不足，請將摘要交由客服同仁確認。");
    if (payload.summary) renderSummary(payload);
    else if (payload.status === "handoff") {
      const reasons = { no_evidence: "目前找不到足夠的原文依據。", conflicting_evidence: "找到的資料內容彼此不一致。", refund_request: "退款問題需要由專人確認。" };
      renderSummary({ summary: `商品：${(productSelect.value || "未提供").toUpperCase()}\n顧客問題（未驗證的使用者陳述）：${questionInput.value}\n已試步驟：${stepsInput.value || "未提供"}\n轉接原因：${reasons[payload.reason] || "需要專人確認。"}` });
    }
    const noGenerationReasons = ["insufficient_information", "reset_required"];
    addUsage(payload, noGenerationReasons.includes(payload.reason) ? false : null);
    addRequestId(payload);
    if (guided()) newQuestionButton.hidden = false;
  }

  function showClarify(payload) {
    slotToken = payload.slot_token;
    const missing = new Set(payload.missing);
    const labels = { product_id: "商品", question: "顧客確認的問題", steps_tried: "已試步驟" };
    const prompt = Array.from(missing, (field) => labels[field] || field).join("、");
    assistStatus.textContent = `請補充：${prompt}。這是第 ${payload.attempt} 次明確追問（最多兩次）；目前輸入只視為未驗證陳述。`;
    startResult("", "需要補充資料", "尚未產生回答；補足必要欄位後，服務才會進行一次回答請求。");
    resultContent.append(element("p", "answer-text", `請補充${prompt}。`));
    addUsage(payload, false);
    addRequestId(payload);
    newQuestionButton.hidden = false;
    if (missing.has("product_id")) productSelect.focus();
    else if (missing.has("question")) questionInput.focus();
    else if (missing.has("steps_tried")) stepsInput.focus();
  }

  function showError(message, requestId, label = "這次沒有完成", code = null, status = null) {
    startResult("error", label, "輸入已保留；服務不會自動重試。確認後可由你手動再送一次。");
    resultContent.append(element("p", "answer-text error-detail", message));
    if (status) resultContent.append(element("p", "request-id", `服務回應 ${status}`));
    if (code) resultContent.append(element("p", "request-id", `錯誤代碼：${code}`));
    if (requestId) resultContent.append(element("p", "request-id", `請求編號：${requestId}`));
  }

  async function copySummary(value) {
    try {
      if (!navigator.clipboard || !navigator.clipboard.writeText) throw new Error("clipboard unavailable");
      await navigator.clipboard.writeText(value);
      announce("摘要已複製，不會自動寄出。", true);
    } catch {
      const fallback = element("textarea", "copy-fallback");
      fallback.value = value;
      fallback.setAttribute("aria-label", "可手動選取並複製的摘要");
      resultContent.append(fallback);
      fallback.focus();
      fallback.select();
      announce("無法使用剪貼簿，摘要已選取，請手動複製。", true);
    }
  }

  function announce(message, show = false) {
    toast.textContent = message;
    if (toastTimer) window.clearTimeout(toastTimer);
    toast.classList.toggle("visible", show);
    if (show) toastTimer = window.setTimeout(() => toast.classList.remove("visible"), 3600);
  }

  async function request(path, body) {
    const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    let payload;
    try { payload = await response.json(); } catch { payload = null; }
    if (!response.ok) {
      const code = payload && typeof payload.code === "string" ? payload.code : "";
      const requestId = payload && typeof payload.request_id === "string" ? payload.request_id : null;
      if (code === "invalid_slot") slotToken = null;
      showError(safeCodes[code] || (response.status === 422 ? "資料格式不符合要求，請檢查內容後再試。" : "服務暫時無法完成這次請求，請確認後再手動重試。"), requestId, "這次沒有完成", code || null, response.status);
      return false;
    }
    if (!validResponse(payload)) {
      showError("服務回應格式無法確認，請稍後檢查服務狀態。", payload && payload.request_id, "回應格式異常");
      return false;
    }
    if (typeof payload.slot_token === "string") {
      slotToken = payload.slot_token;
      serverFields = {
        product_id: payload.product_id || null,
        question: payload.question || null,
        steps_tried: payload.steps_tried || null,
      };
    }
    if (payload.status === "clarify") showClarify(payload);
    else if (payload.status === "handoff") showHandoff(payload);
    else {
      slotToken = payload.slot_token || slotToken;
      showAnswer(payload);
      newQuestionButton.hidden = false;
    }
    return true;
  }

  async function submit(event) {
    event.preventDefault();
    if (busy) return;
    const productId = productSelect.value || null;
    const question = questionInput.value;
    const steps = stepsInput.value;
    if (!guided() && !question.trim()) { showError("請先輸入顧客問題。", null, "尚未送出"); questionInput.focus(); return; }
    if (characterLength(question) > 1000 || characterLength(steps) > 1000) { showError("每個欄位最多 1000 字，請縮短後再送出。", null, "尚未送出"); questionInput.focus(); return; }
    setBusy(true);
    startResult("", "處理中", "正在整理這次問題，請稍候。");
    resultContent.append(element("p", "answer-text", "請稍候，正在確認回答與原文依據。"));
    try {
      if (guided()) {
        const fields = currentAssistFields();
        if (needsServerReset(fields)) {
          const oldToken = slotToken;
          const reset = await request("/assist", { action: "reset", slot_token: oldToken });
          if (reset) {
            serverFields = { product_id: null, question: null, steps_tried: null };
            assistStatus.textContent = "已明確重設商品對話，避免沿用舊商品資料；請再次送出目前資料。";
          }
          return;
        }
        const action = slotToken ? "continue" : "start";
        const body = { action };
        if (slotToken) body.slot_token = slotToken;
        if (productId) body.product_id = productId;
        if (slotToken && serverFields && serverFields.question && !question.trim()) body.question = null;
        else if (question.trim()) body.question = question.trim();
        if (slotToken && serverFields && serverFields.steps_tried && !steps.trim()) body.steps_tried = null;
        else if (steps.trim()) body.steps_tried = steps.trim();
        await request("/assist", body);
      } else {
        const response = await fetch("/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ product_id: productId, question: question.trim() }) });
        let payload;
        try { payload = await response.json(); } catch { payload = null; }
        if (!response.ok) {
          const code = payload && typeof payload.code === "string" ? payload.code : "";
          showError(safeCodes[code] || "服務暫時無法完成這次請求，請確認後再手動重試。", payload && payload.request_id, "這次沒有完成", code || null, response.status);
        } else if (!validResponse(payload)) showError("服務回應格式無法確認，請稍後檢查服務狀態。", payload && payload.request_id, "回應格式異常");
        else if (payload.status === "answer") showAnswer(payload);
        else showHandoff(payload);
      }
    } catch {
      showError("無法連線到本機問答服務。確認服務狀態後，可由你手動重新送出。", null, "連線中斷");
    } finally {
      setBusy(false);
      updateMode();
      if (guided()) submitButton.querySelector(".button-label").textContent = slotToken ? "補充資料" : "開始逐步整理";
    }
  }

  async function resetConversation() {
    if (busy) return;
    const oldToken = slotToken;
    const oldServerFields = serverFields;
    let resetSlotToken = null;
    let resetConfirmed = !oldToken;
    setBusy(true);
    newQuestionButton.disabled = true;
    try {
      if (oldToken) {
        resetConfirmed = await request("/assist", { action: "reset", slot_token: oldToken });
        if (resetConfirmed) resetSlotToken = slotToken;
      }
    } catch {
      resetConfirmed = false;
    } finally {
      slotToken = resetConfirmed ? resetSlotToken : slotToken;
      if (resetConfirmed) serverFields = { product_id: null, question: null, steps_tried: null };
      else serverFields = serverFields || oldServerFields;
      productSelect.value = "";
      previousProductId = "";
      questionInput.value = "";
      stepsInput.value = "";
      assistStatus.textContent = "資料只保留在目前頁面與本機暫存對話中。";
      newQuestionButton.disabled = false;
      updateCount();
      updateMode();
      if (resetConfirmed) {
        startResult("", "回答整理結果", "送出問題後，結果會顯示在這裡。");
        resultContent.append(element("div", "empty-state", "新問題已清空，可開始整理。"));
      } else {
        showError("本機已清除畫面資料，但服務未確認舊對話重設。下次送出會建立新暫存對話。", null, "重設未確認");
      }
      setBusy(false);
      updateMode();
      newQuestionButton.hidden = resetConfirmed || !slotToken;
    }
  }

  questionInput.addEventListener("input", updateCount);
  productSelect.addEventListener("change", async () => {
    const changedExistingProduct = previousProductId && previousProductId !== productSelect.value;
    if (changedExistingProduct) {
      const oldToken = slotToken;
      questionInput.value = "";
      stepsInput.value = "";
      assistStatus.textContent = "商品已變更；正在明確重設舊問題，避免沿用舊欄位。";
      if (sourceDialog.open) sourceDialog.close();
      sourceMetadata.replaceChildren();
      sourceText.textContent = "";
      startResult("", "回答整理結果", "商品已變更，請輸入新的顧客問題。");
      resultContent.append(element("div", "empty-state", "商品已變更，舊回答與引用已清除。"));
      if (oldToken) {
        setBusy(true);
        try {
          const reset = await request("/assist", { action: "reset", slot_token: oldToken });
          assistStatus.textContent = reset
            ? "商品已變更；舊問題與已試步驟已清除，伺服器對話也已重設。請輸入新問題。"
            : "商品已變更；畫面已清除，但伺服器未確認舊對話重設。請再次開始新問題後再送出。";
        } catch {
          assistStatus.textContent = "商品已變更；畫面已清除，但伺服器未確認舊對話重設。請確認服務狀態後再送出。";
        } finally {
          setBusy(false);
          updateMode();
        }
      }
    }
    previousProductId = productSelect.value;
    updateCount();
    updateMode();
  });
  form.elements.mode.forEach((input) => input.addEventListener("change", updateMode));
  form.addEventListener("submit", submit);
  newQuestionButton.addEventListener("click", resetConversation);
  document.querySelector("#close-dialog").addEventListener("click", () => sourceDialog.close());
  document.querySelector("#dialog-done").addEventListener("click", () => sourceDialog.close());
  sourceDialog.addEventListener("close", () => { if (previousDialogFocus && previousDialogFocus.isConnected) previousDialogFocus.focus(); });
  sourceDialog.addEventListener("click", (event) => { if (event.target === sourceDialog) sourceDialog.close(); });
  updateCount();
  updateMode();
})();
