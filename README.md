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
uv run --locked uvicorn customer_support_rag.api:app --host 127.0.0.1 --port 8000
```

`GET http://127.0.0.1:8000/health` 回傳服務狀態、應用程式名稱與版本。這個端點只回報 API 程序狀態，不檢查尚未加入的模型、知識庫或儲存服務。

GitHub Actions 會使用 `uv.lock` 安裝依賴，在 Python 3.12 與 3.13 執行相同的格式、lint 與測試檢查，並以 Gitleaks 掃描 secrets。

目前包含最小 FastAPI app、健康檢查端點，以及以 TOML front matter 標記的少量合成知識文件範例（見 [`knowledge/README.md`](knowledge/README.md)）。文件讀取成功不等於人工審查、核准或啟用於線上 RAG。

本機 embedding 只做同步 adapter：既有 chunk 的原文送到固定的本機 Ollama 位址。呼叫者可以再把一份文件的來源與向量放進自己指定的 SQLite 檔，之後用同一個 key 讀回；store 不會自己呼叫模型。問句向量與離線 exact cosine 也在同一份說明。呼叫方式、HP-A mock 範例與保存格式見 [`knowledge/README.md`](knowledge/README.md)。這個 README 的範例沒有在本變更中對 Ollama 執行。

離線回答草稿把 caller 已經選好的 hits 收成有界 evidence request，再解析一份假的自然回答。固定 policy 與模型看得到的 JSON 資料分開。HP-A 的來源對照、mock HTTP 自然草稿與無證據 handoff 在 [`knowledge/README.md`](knowledge/README.md)。那是 mock，沒有呼叫 Luna，也還沒有核准啟用。
