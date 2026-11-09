# 客服 RAG

以合成資料驗證客服檢索增強生成（RAG）的 proof of concept。此 repository 目前處於早期開發階段，尚未核准用於正式環境。

## 開發環境

- CI 已驗證 Python 3.12 與 3.13
- `uv.lock` 固定應用程式與開發依賴的完整版本及雜湊
- FastAPI 提供 HTTP routing，Pydantic 定義並驗證健康檢查回應，Uvicorn 在本機啟動 ASGI app
- Starlette 固定在相容版本，讓開發依賴中的 HTTPX 支援 FastAPI `TestClient`；pytest 執行測試，Ruff 負責格式與 lint

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

目前只包含最小 FastAPI app 與健康檢查端點。RAG 行為、模型整合及合成語料會在後續 PR 分別提出 review。
