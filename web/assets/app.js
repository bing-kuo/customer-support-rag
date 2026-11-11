(() => {
  "use strict";

  const form = document.querySelector("#ask-form");
  const productSelect = document.querySelector("#product");
  const questionInput = document.querySelector("#question");
  const characterCount = document.querySelector("#character-count");
  const submitButton = document.querySelector("#submit-button");
  const result = document.querySelector("#result");
  const resultCaption = document.querySelector("#result-caption");
  const resultContent = document.querySelector("#result-content");
  const sourceDialog = document.querySelector("#source-dialog");
  const sourceMetadata = document.querySelector("#source-metadata");
  const sourceText = document.querySelector("#source-text");
  const toast = document.querySelector("#toast");
  const safeCodes = {
    invalid_request: "問題格式不符合要求，請檢查內容後再試。",
    request_too_large: "這次請求內容過長，請縮短問題後再試。",
    ask_not_configured: "問答服務目前尚未設定完成。",
    dependency_unavailable: "問答服務暫時無法使用。",
    evidence_unavailable: "目前無法讀取回答依據。",
    paid_blocked: "目前無法完成這次回答。",
    wire_body_exceeded: "這次請求內容過長，請縮短問題後再試。",
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
  let previousDialogFocus = null;
  let toastTimer = null;

  function element(tag, className, value) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (value !== undefined) node.textContent = value;
    return node;
  }

  function characterLength(value) {
    return Array.from(value).length;
  }

  function updateCount() {
    const count = characterLength(questionInput.value);
    characterCount.textContent = `${count} / 1000`;
    characterCount.classList.toggle("over-limit", count > 1000);
  }

  function setBusy(value) {
    busy = value;
    submitButton.disabled = value;
    productSelect.disabled = value;
    questionInput.disabled = value;
    result.setAttribute("aria-busy", String(value));
    submitButton.querySelector(".button-label").textContent = value ? "整理中…" : "整理回答";
    resultCaption.textContent = value ? "正在整理這次問題，請稍候。" : "送出問題後，結果會顯示在這裡。";
  }

  function startResult(kind, label, caption) {
    resultCaption.textContent = caption;
    resultContent.replaceChildren();
    resultContent.append(element("span", `result-badge ${kind}`, label));
  }

  function addUsage(payload) {
    const box = element("section", "usage-box");
    box.append(element("p", "usage-title", payload.usage ? "本次用量" : "本次用量未知"));
    if (payload.usage) {
      const usage = payload.usage;
      box.append(element("p", "usage-detail", `輸入 ${usage.input_tokens} · 輸出 ${usage.output_tokens} · 合計 ${usage.total_tokens} tokens`));
    } else if (payload.usage_code) {
      box.append(element("p", "usage-detail", `服務未提供可確認的用量（${payload.usage_code}）。`));
    } else {
      box.append(element("p", "usage-detail", "本次沒有呼叫生成模型，因此沒有模型用量。"));
    }
    resultContent.append(box);
  }

  function addRequestId(payload) {
    if (typeof payload.request_id === "string") {
      resultContent.append(element("p", "request-id", `請求編號：${payload.request_id}`));
    }
  }

  function openSource(source, button) {
    previousDialogFocus = button;
    sourceMetadata.replaceChildren();
    const fields = [
      ["標題", source.title], ["版本", source.version], ["商品", source.product_id],
      ["文件 ID", source.document_id], ["Chunk", String(source.chunk_index)],
      ["段落", source.heading], ["原文字元位置", `${source.char_start}–${source.char_end}`],
    ];
    for (const [label, value] of fields) {
      const pair = element("div");
      pair.append(element("dt", "", label), element("dd", "", value));
      sourceMetadata.append(pair);
    }
    sourceText.textContent = source.text;
    sourceDialog.showModal();
    document.querySelector("#close-dialog").focus();
  }

  function addSources(sources) {
    if (!sources.length) return;
    resultContent.append(element("div", "result-divider"));
    resultContent.append(element("h3", "subheading", "回答引用的原文"));
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

  function showAnswer(payload) {
    startResult("", "回答草稿", "請先核對回答與引用原文，再決定是否使用。");
    resultContent.append(element("p", "answer-text", payload.answer));
    addSources(payload.sources);
    resultContent.append(element("div", "result-divider"));
    addUsage(payload);
    addRequestId(payload);
  }

  function showHandoff(payload, productId, question) {
    const reasons = {
      no_evidence: "目前找不到足夠的原文依據。",
      conflicting_evidence: "找到的資料內容彼此不一致。",
      refund_request: "退款問題需要由專人確認。",
    };
    startResult("handoff", "建議轉由專人處理", "這次沒有提供可直接使用的回答。");
    resultContent.append(element("p", "answer-text", reasons[payload.reason] || "這次需要由專人確認。"));
    const note = element("div", "notice-box");
    note.append(element("p", "notice-title", "提供給客服同仁的摘要"));
    note.append(element("p", "notice-detail", `商品：${productId.toUpperCase()}\n顧客問題（未驗證的使用者陳述）：${question}\n轉接原因：${reasons[payload.reason] || "需要專人確認。"}`));
    resultContent.append(note);
    const copy = element("button", "copy-button", "複製摘要");
    copy.type = "button";
    copy.addEventListener("click", () => copySummary(productId, question, reasons[payload.reason] || "需要專人確認。"));
    resultContent.append(copy);
    addUsage(payload);
    addRequestId(payload);
  }

  function showError(message, requestId, label = "這次沒有完成", code = null) {
    startResult("error", label, "輸入內容已保留；確認後可由你手動重新送出。");
    resultContent.append(element("p", "answer-text error-detail", message));
    if (code) resultContent.append(element("p", "request-id", `錯誤代碼：${code}`));
    if (requestId) resultContent.append(element("p", "request-id", `請求編號：${requestId}`));
  }

  async function copySummary(productId, question, reason) {
    const value = `商品：${productId.toUpperCase()}\n顧客問題（未驗證的使用者陳述）：${question}\n轉接原因：${reason}`;
    try {
      if (!navigator.clipboard || !navigator.clipboard.writeText) throw new Error("clipboard unavailable");
      await navigator.clipboard.writeText(value);
      announce("摘要已複製。請交由客服同仁確認；摘要不會自動寄出。", true);
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

  function validResponse(payload) {
    if (!payload || typeof payload !== "object" || typeof payload.request_id !== "string") return false;
    if (!Array.isArray(payload.sources) || !(payload.usage === null || validUsage(payload.usage))) return false;
    if (!(typeof payload.usage_code === "string" || payload.usage_code === null)) return false;
    if (!Array.isArray(payload.usage_fields)) return false;
    if (payload.status === "answer") {
      return typeof payload.answer === "string" && payload.reason === null && payload.sources.every((source) =>
        source && ["source_id", "artifact_key", "document_id", "product_id", "version", "title", "heading", "text"].every((key) => typeof source[key] === "string") &&
        ["chunk_index", "char_start", "char_end"].every((key) => Number.isInteger(source[key])) &&
        source.chunk_index >= 0 && source.char_start >= 0 && source.char_end >= source.char_start &&
        ["hp-a", "hp-b"].includes(source.product_id));
    }
    return payload.status === "handoff" && payload.answer === null && ["no_evidence", "conflicting_evidence", "refund_request"].includes(payload.reason) && payload.sources.length === 0;
  }

  function validUsage(usage) {
    const required = ["input_tokens", "output_tokens", "total_tokens"];
    const optional = ["cached_tokens", "cache_write_tokens", "ttl_30m_tokens", "ttl_12h_tokens", "reasoning_tokens"];
    return usage && typeof usage === "object" && required.every((key) => Number.isSafeInteger(usage[key]) && usage[key] >= 0) &&
      optional.every((key) => usage[key] === null || (Number.isSafeInteger(usage[key]) && usage[key] >= 0));
  }

  async function submit(event) {
    event.preventDefault();
    if (busy) return;
    const productId = productSelect.value;
    const question = questionInput.value;
    const length = characterLength(question);
    if (!question.trim()) {
      showError("請先輸入顧客問題。", null, "尚未送出");
      questionInput.focus();
      return;
    }
    if (length > 1000) {
      showError("問題最多 1000 字，請縮短後再送出。", null, "尚未送出");
      questionInput.focus();
      return;
    }
    setBusy(true);
    startResult("", "處理中", "正在整理這次問題，請稍候。");
    resultContent.append(element("p", "answer-text", "請稍候，正在確認回答與原文依據。"));
    try {
      const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ product_id: productId, question }),
      });
      let payload;
      try { payload = await response.json(); } catch { payload = null; }
      if (!response.ok) {
        const code = payload && typeof payload.code === "string" ? payload.code : "";
        const requestId = payload && typeof payload.request_id === "string" ? payload.request_id : null;
        const message = safeCodes[code] || (response.status === 422 ? "問題格式不符合要求，請檢查內容後再試。" : "服務暫時無法完成這次請求，請確認後再手動重試。");
        showError(message, requestId, `服務回應 ${response.status}`, code || null);
      } else if (!validResponse(payload)) {
        showError("服務回應格式無法確認，請稍後檢查服務狀態。", payload && typeof payload.request_id === "string" ? payload.request_id : null, "回應格式異常");
      } else if (payload.status === "answer") {
        showAnswer(payload);
      } else {
        showHandoff(payload, productId, question);
      }
    } catch {
      showError("無法連線到本機問答服務。確認服務狀態後，可由你手動重新送出。", null, "連線中斷");
    } finally {
      setBusy(false);
    }
  }

  questionInput.addEventListener("input", updateCount);
  form.addEventListener("submit", submit);
  document.querySelector("#close-dialog").addEventListener("click", () => sourceDialog.close());
  document.querySelector("#dialog-done").addEventListener("click", () => sourceDialog.close());
  sourceDialog.addEventListener("close", () => {
    if (previousDialogFocus && previousDialogFocus.isConnected) previousDialogFocus.focus();
  });
  sourceDialog.addEventListener("click", (event) => {
    if (event.target === sourceDialog) sourceDialog.close();
  });
  updateCount();
})();
